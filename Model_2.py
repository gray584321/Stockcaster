import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from torch.optim import Adamçç
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
from torch.cuda.amp import autocast, GradScaler

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from dataloader import StockDataLoader

class ProbSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        B, L, d_model = query.size()
        head_dim = d_model // self.num_heads
        local_scaling = 1.0 / math.sqrt(head_dim)
        
        def reshape(x):
            return x.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        
        q = reshape(query)
        k = reshape(key)
        v = reshape(value)
        scores = torch.matmul(q, k.transpose(-2, -1)) * local_scaling
        m1 = torch.logsumexp(scores, dim=-1) - math.log(L)
        m2 = scores.mean(dim=-1)
        M_measure = m1 - m2
        u = max(1, int(math.ceil(math.log(L + 1))))
        _, topk_query_indices = M_measure.topk(k=u, dim=-1)
        v_avg = v.mean(dim=2, keepdim=True)
        default_output = v_avg.expand(B, self.num_heads, L, head_dim).clone()
        q_active = torch.gather(q, dim=2, index=topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, head_dim))
        scores_active = torch.matmul(q_active, k.transpose(-2, -1)) * local_scaling
        if attn_mask is not None:
            causal_mask_active = attn_mask[topk_query_indices]
            scores_active = scores_active.masked_fill(causal_mask_active, float('-inf'))
        attn_weights_active = F.softmax(scores_active, dim=-1)
        attn_weights_active = self.dropout(attn_weights_active)
        attn_output_active = torch.matmul(attn_weights_active, v)
        attn_output = default_output
        attn_output.scatter_(2, topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, head_dim), attn_output_active)
        attn_weights_full = torch.zeros(B, self.num_heads, L, L, device=q.device)
        attn_weights_full.scatter_(2, topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, L), attn_weights_active)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_full

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TimeEmbedding(nn.Module):
    def __init__(self, d_model, year_range=(1900, 2100), month_size=12, week_size=53, day_size=32, hour_size=24, minute_size=60):
        super().__init__()
        self.year_offset = year_range[0]
        year_num = year_range[1] - year_range[0] + 1
        self.year_embedding = nn.Embedding(year_num, d_model)
        self.month_embedding = nn.Embedding(month_size, d_model)
        self.week_embedding = nn.Embedding(week_size, d_model)
        self.day_embedding = nn.Embedding(day_size, d_model)
        self.hour_embedding = nn.Embedding(hour_size, d_model)
        self.minute_embedding = nn.Embedding(minute_size, d_model)
        
    def forward(self, time_tensor):
        year = (time_tensor[:,:,0].long() - self.year_offset).clamp(min=0)
        month = (time_tensor[:,:,1].long() - 1).clamp(min=0)
        week = (time_tensor[:,:,2].long() - 1).clamp(min=0)
        day = (time_tensor[:,:,3].long() - 1).clamp(min=0)
        hour = time_tensor[:,:,4].long()
        minute = time_tensor[:,:,5].long()
        return (self.year_embedding(year) + self.month_embedding(month) +
                self.week_embedding(week) + self.day_embedding(day) +
                self.hour_embedding(hour) + self.minute_embedding(minute))

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, apply_distillation=True, use_prob_sparse=True):
        super().__init__()
        if use_prob_sparse:
            self.self_attn = ProbSparseAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.apply_distillation = apply_distillation
        if self.apply_distillation:
            self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.elu = nn.ELU()

    def forward(self, x):
        x_attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self.ffn(x))
        if self.apply_distillation:
            x = x.transpose(1, 2)
            x = self.elu(self.conv(x))
            x = self.pool(x)
            x = x.transpose(1, 2)
        return x

class InformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout, use_prob_sparse=True):
        super().__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, nhead, d_ff, dropout, True, use_prob_sparse) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, use_prob_sparse=True):
        super().__init__()
        if use_prob_sparse:
            self.self_attn = ProbSparseAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, memory):
        L = x.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        x_self, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + x_self)
        x_cross, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + x_cross)
        x = self.norm3(x + self.ffn(x))
        return x

class InformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout, use_prob_sparse=True):
        super().__init__()
        self.layers = nn.ModuleList([InformerDecoderLayer(d_model, nhead, d_ff, dropout, use_prob_sparse) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

class Informer(nn.Module):
    def __init__(self, input_dim=12, d_model=64, d_ff=256, nhead=8,
                 enc_layers=3, dec_layers=3, dropout=0.05,
                 encoder_length=96, decoder_length=48, prediction_length=24,
                 use_prob_sparse=True):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.prediction_length = prediction_length
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=encoder_length)
        self.pos_decoder = PositionalEncoding(d_model, max_len=decoder_length + prediction_length)
        self.encoder = InformerEncoder(num_layers=enc_layers, d_model=d_model,
                                       nhead=nhead, d_ff=d_ff, dropout=dropout, use_prob_sparse=use_prob_sparse)
        self.decoder = InformerDecoder(num_layers=dec_layers, d_model=d_model,
                                       nhead=nhead, d_ff=d_ff, dropout=dropout, use_prob_sparse=use_prob_sparse)
        self.output_projection = nn.Linear(d_model, input_dim)
        self.time_embed = TimeEmbedding(d_model)

    def forward(self, encoder_x, decoder_x, target=None, decoder_time=None):
        enc_input = self.input_projection(encoder_x)
        enc_input = self.pos_encoder(enc_input)
        memory = self.encoder(enc_input)
        if target is not None:
            dec_input = torch.cat([decoder_x, target], dim=1)
        else:
            placeholder = torch.zeros(decoder_x.size(0), self.prediction_length, self.input_dim, device=decoder_x.device)
            dec_input = torch.cat([decoder_x, placeholder], dim=1)
        dec_input = self.input_projection(dec_input)
        dec_input = self.pos_decoder(dec_input)
        if decoder_time is not None:
            time_emb = self.time_embed(decoder_time)
            dec_input = dec_input + time_emb
        dec_output = self.decoder(dec_input, memory)
        output = self.output_projection(dec_output)
        return output[:, -self.prediction_length:, :]

def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    scaler = GradScaler() if device.type == 'cuda' else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            encoder = batch['encoder'].to(device)
            decoder = batch['decoder'].to(device)
            target = batch['target'].to(device)
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    output = model(encoder, decoder, target)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(encoder, decoder, target)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        total_val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                encoder = batch['encoder'].to(device)
                decoder = batch['decoder'].to(device)
                target = batch['target'].to(device)
                output = model(encoder, decoder, target)
                loss = criterion(output, target)
                total_val_loss += loss.item()
                val_predictions.append(output)
                val_targets.append(target)
        avg_val_loss = total_val_loss / len(val_loader)
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        mae, rmse, mape = compute_metrics(val_predictions, val_targets)
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%, Duration: {epoch_duration:.2f}s, "
              f"Best Val Loss: {best_val_loss:.6f}, LR: {current_lr:.6e}")
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_informer_model.pt")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= 3:
            print("Early stopping triggered.")
            break

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_test_loss = 0.0
    predictions = []
    targets = []
    datetime_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            encoder = batch['encoder'].to(device)
            decoder = batch['decoder'].to(device)
            target = batch['target'].to(device)
            output = model(encoder, decoder, target)
            loss = criterion(output, target)
            total_test_loss += loss.item()
            predictions.append(output)
            targets.append(target)
            if 'datetime' in batch:
                datetime_list.append(batch['datetime'])
                
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    datetimes = [dt for sublist in datetime_list for dt in sublist] if datetime_list else None
    return predictions, targets, datetimes

def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    mae = np.mean(np.abs(pred_np - target_np))
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
    mape = np.mean(np.abs((pred_np - target_np) / (target_np + 1e-5))) * 100
    return mae, rmse, mape

if __name__ == "__main__":
    input_dim = 12
    d_model = 64
    d_ff = 256
    nhead = 8
    enc_layers = 3
    dec_layers = 3
    dropout = 0.05
    encoder_length = 96
    decoder_length = 48
    prediction_length = 24
    batch_size = 128
    num_epochs = 50
    learning_rate = 4e-5

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    seed = 69
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")
    print(f"Using device: {device}")
    
    csv_path = "data/processed/SPY.csv"
    stock_data_loader = StockDataLoader(csv_path, use_cyclical_encoding=True)
    stock_data_loader.prepare_datasets()
    train_loader, val_loader, test_loader = stock_data_loader.get_dataloaders(batch_size=batch_size)
    
    model = Informer(input_dim=input_dim, d_model=d_model, d_ff=d_ff, nhead=nhead,
                     enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout,
                     encoder_length=encoder_length, decoder_length=decoder_length,
                     prediction_length=prediction_length, use_prob_sparse=True)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device)
    predictions, targets, datetimes = evaluate_model(model, test_loader, criterion, device)
    mae, rmse, mape = compute_metrics(predictions, targets)
    print(f"Test Metrics - MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%")
    
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    close_price_mean = 100.0
    close_price_std = 20.0
    pred_np[:, :, 0] = pred_np[:, :, 0] * close_price_std + close_price_mean
    target_np[:, :, 0] = target_np[:, :, 0] * close_price_std + close_price_mean
    
    sample_idx = 0
    feature_idx = 0
    pred_series = pred_np[sample_idx, :, feature_idx]
    target_series = target_np[sample_idx, :, feature_idx]
    time_steps = range(1, len(pred_series) + 1)
    
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, target_series, label="Ground Truth", marker='o')
    plt.plot(time_steps, pred_series, label="Prediction", marker='x')
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.title(f"Ground Truth vs Prediction (Sample {sample_idx}, Feature {feature_idx})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(results_folder, "ground_truth_vs_prediction.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved prediction plot to {plot_path}")
    
    num_samples, pred_length, _ = pred_np.shape
    sample_ids = np.repeat(np.arange(num_samples), pred_length)
    time_steps_full = np.tile(np.arange(pred_length), num_samples)
    ground_truth_flat = target_np[:, :, feature_idx].flatten()
    prediction_flat = pred_np[:, :, feature_idx].flatten()
    
    df = pd.DataFrame({
        "sample": sample_ids,
        "time_step": time_steps_full,
        "ground_truth": ground_truth_flat,
        "prediction": prediction_flat
    })
    if datetimes is not None:
        if len(datetimes) == num_samples * pred_length:
            df["datetime"] = datetimes
        elif len(datetimes) == num_samples:
            df["datetime"] = np.repeat(datetimes, pred_length)
        else:
            df["datetime"] = None
    else:
        df["datetime"] = None
    
    csv_path = os.path.join(results_folder, "test_predictions_vs_ground_truth.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw predictions and ground truth data to {csv_path}")
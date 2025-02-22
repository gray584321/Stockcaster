import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt   # <-- new import for plotting
import os                        # <-- new import for creating folders and paths
import pandas as pd              # <-- new import for saving CSV files
import random  # added for random seed setting

# Define helper function for setting the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import the custom StockDataLoader defined in dataloader.py
from dataloader import StockDataLoader

# ------------------------------
# ProbSparse Self-Attention Module
# ------------------------------
class ProbSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(ProbSparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        # Note: we remove self.scaling here and compute per-head scaling in forward
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        # query, key, value: (B, L, d_model)
        B, L, d_model = query.size()
        head_dim = d_model // self.num_heads
        # Use per-head scaling to stabilize the exponentiation operation.
        local_scaling = 1.0 / math.sqrt(head_dim)
        
        def reshape(x):
            # (B, L, d_model) -> (B, num_heads, L, head_dim)
            return x.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        
        q = reshape(query)  # (B, num_heads, L, head_dim)
        k = reshape(key)
        v = reshape(value)
        
        # Compute raw scores using scaled dot-product for each query-key pair.
        scores = torch.matmul(q, k.transpose(-2, -1)) * local_scaling  # shape: (B, num_heads, L, L)
        
        # ------------------------------
        # Sparsity Measurement using an asymmetric exponential kernel
        # Compute:
        #   M(q_i, K) = ln((1/L) * sum(exp(raw_scores))) - (1/L) * sum(raw_scores)
        # where raw_scores = (q_i dot k_j)/sqrt(d_head)
        # ------------------------------
        m1 = torch.logsumexp(scores, dim=-1) - math.log(L)   # (B, num_heads, L)
        m2 = scores.mean(dim=-1)                               # (B, num_heads, L)
        M_measure = m1 - m2                                    # (B, num_heads, L)
        
        # Determine the number of active queries: u = ceil(log(L+1))
        u = max(1, int(math.ceil(math.log(L + 1))))
        
        # Select top-u queries based on the sparsity measure for each head
        _, topk_query_indices = M_measure.topk(k=u, dim=-1)  # (B, num_heads, u)
        
        # Compute a default output for all queries using the global average of v
        v_avg = v.mean(dim=2, keepdim=True)  # (B, num_heads, 1, head_dim)
        default_output = v_avg.expand(B, self.num_heads, L, head_dim).clone()  # (B, num_heads, L, head_dim)
        
        # Gather active queries based on selected indices
        q_active = torch.gather(q, dim=2, 
                                 index=topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, head_dim))
        
        # ------------------------------
        # Final Attention Calculation for active queries:
        #    Compute scores for active queries using the modified exponential kernel:
        #    scores_active = (q_active dot k^T)*local_scaling,
        #    then apply Softmax to obtain the probability distribution over keys.
        # ------------------------------
        scores_active = torch.matmul(q_active, k.transpose(-2, -1)) * local_scaling  # (B, num_heads, u, L)
        # --- Newly Added: Apply causal mask if provided ---
        if attn_mask is not None:
            # attn_mask is expected to be of shape (L, L) [boolean mask]
            # Gather the mask rows according to the active query indices:
            causal_mask_active = attn_mask[topk_query_indices]  # (B, num_heads, u, L)
            scores_active = scores_active.masked_fill(causal_mask_active, float('-inf'))
        
        attn_weights_active = F.softmax(scores_active, dim=-1)
        attn_weights_active = self.dropout(attn_weights_active)
        
        # Compute output for active queries
        attn_output_active = torch.matmul(attn_weights_active, v)  # (B, num_heads, u, head_dim)
        
        # Scatter the computed outputs for active queries back into the appropriate positions
        attn_output = default_output  # (B, num_heads, L, head_dim)
        attn_output.scatter_(2, topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, head_dim), 
                              attn_output_active)
        
        # (Optional) Build a corresponding full attention weights tensor with zeros for non-active queries
        attn_weights_full = torch.zeros(B, self.num_heads, L, L, device=q.device)
        attn_weights_full.scatter_(2, topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, L), 
                                    attn_weights_active)
        
        # Reshape output back to (B, L, d_model) and project to final dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_full

# ------------------------------
# Positional Encoding Module
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# ------------------------------
# Global Time Embedding Module
# ------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, d_model, year_range=(1900, 2100), month_size=12, week_size=53, day_size=32, hour_size=24, minute_size=60):
        super(TimeEmbedding, self).__init__()
        self.year_offset = year_range[0]
        year_num = year_range[1] - year_range[0] + 1
        self.year_embedding = nn.Embedding(year_num, d_model)
        self.month_embedding = nn.Embedding(month_size, d_model)
        self.week_embedding = nn.Embedding(week_size, d_model)
        self.day_embedding = nn.Embedding(day_size, d_model)
        self.hour_embedding = nn.Embedding(hour_size, d_model)
        self.minute_embedding = nn.Embedding(minute_size, d_model)

    def forward(self, time_tensor):
        # time_tensor: (B, L, 6) where columns are [year, month, week, day, hour, minute]
        year = (time_tensor[:,:,0].long() - self.year_offset).clamp(min=0)
        month = (time_tensor[:,:,1].long() - 1).clamp(min=0)
        week = (time_tensor[:,:,2].long() - 1).clamp(min=0)
        day = (time_tensor[:,:,3].long() - 1).clamp(min=0)
        hour = time_tensor[:,:,4].long()
        minute = time_tensor[:,:,5].long()
        e_year = self.year_embedding(year)
        e_month = self.month_embedding(month)
        e_week = self.week_embedding(week)
        e_day = self.day_embedding(day)
        e_hour = self.hour_embedding(hour)
        e_minute = self.minute_embedding(minute)
        return e_year + e_month + e_week + e_day + e_hour + e_minute

# ------------------------------
# Informer Encoder Layer with Distillation
# ------------------------------
class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, apply_distillation=True, use_prob_sparse=True):
        super(InformerEncoderLayer, self).__init__()
        if use_prob_sparse:
            self.self_attn = ProbSparseAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                                   dropout=dropout, batch_first=True)
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
            # --------------------------------------------------------------
            # Self-Attention Distilling Operation (Equation 4):
            # X₍ⱼ₊₁₎ = MaxPool( ELU( Conv1d( [Xⱼ]ᵀ ) ) )
            # 
            # This block first transposes the tensor from shape (B, L, d_model)
            # to (B, d_model, L) so that a 1D convolution is applied over the 
            # sequence (time-series) dimension. The ELU activation is then applied 
            # followed by a max pooling operation, which reduces the sequence length 
            # roughly by half.
            # --------------------------------------------------------------
            self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                  kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.elu = nn.ELU()

    def forward(self, x):
        # x: (B, L, d_model)
        x_attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x_attn)
        x_ffn = self.ffn(x)
        x = self.norm2(x + x_ffn)
        if self.apply_distillation:
            # Apply the self-attention distilling block (Equation 4)
            # Transpose x from (B, L, d_model) to (B, d_model, L)
            x = x.transpose(1, 2)
            # Apply convolution, ELU activation, and max-pooling
            x = self.elu(self.conv(x))
            x = self.pool(x)  # Reduces sequence length by about half
            # Transpose back to (B, new_L, d_model) where new_L ≈ L/2
            x = x.transpose(1, 2)
        return x

# ------------------------------
# Informer Encoder (Stacking Layers)
# ------------------------------
class InformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout, use_prob_sparse=True):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Apply distillation after every self-attention layer to reduce sequence length
            apply_distill = True
            layer = InformerEncoderLayer(d_model, nhead, d_ff, dropout,
                                         apply_distillation=apply_distill,
                                         use_prob_sparse=use_prob_sparse)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# ------------------------------
# Informer Decoder Layer
# ------------------------------
class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, use_prob_sparse=True):
        super(InformerDecoderLayer, self).__init__()
        if use_prob_sparse:
            self.self_attn = ProbSparseAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                                   dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                                dropout=dropout, batch_first=True)
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
        # ------------------------------
        # Newly Added: Generate a causal mask for masked self-attention in the decoder.
        # This mask prevents attending to future time steps.
        # ------------------------------
        L = x.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        # Pass the causal mask to self-attention.
        x_self, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + x_self)
        
        x_cross, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + x_cross)
        x_ffn = self.ffn(x)
        x = self.norm3(x + x_ffn)
        return x

# ------------------------------
# Informer Decoder (Stacking Layers)
# ------------------------------
class InformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout, use_prob_sparse=True):
        super(InformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [InformerDecoderLayer(d_model, nhead, d_ff, dropout, use_prob_sparse=use_prob_sparse) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

# ------------------------------
# Full Informer Network
# ------------------------------
class Informer(nn.Module):
    def __init__(self, input_dim=12, d_model=64, d_ff=256, nhead=8,
                 enc_layers=3, dec_layers=3, dropout=0.05,
                 encoder_length=96, decoder_length=48, prediction_length=24,
                 use_prob_sparse=True):
        super(Informer, self).__init__()
        self.input_dim = input_dim  # Needed to construct dummy tokens later
        self.d_model = d_model
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.prediction_length = prediction_length

        # Project input features (12-dim) to model space (d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        # Positional encoding for encoder (max length = encoder_length)
        self.pos_encoder = PositionalEncoding(d_model, max_len=encoder_length)
        # For decoder, we accept concatenated guiding + target seq (length = decoder_length + prediction_length)
        self.pos_decoder = PositionalEncoding(d_model, max_len=decoder_length + prediction_length)
        # Construct encoder and decoder stacks (passing along use_prob_sparse)
        self.encoder = InformerEncoder(num_layers=enc_layers, d_model=d_model,
                                       nhead=nhead, d_ff=d_ff, dropout=dropout, use_prob_sparse=use_prob_sparse)
        self.decoder = InformerDecoder(num_layers=dec_layers, d_model=d_model,
                                       nhead=nhead, d_ff=d_ff, dropout=dropout, use_prob_sparse=use_prob_sparse)
        # Final projection back to original feature dimension
        self.output_projection = nn.Linear(d_model, input_dim)
        # Global Time Embedding for decoder using discrete embeddings for each time component:
        self.time_embed = TimeEmbedding(d_model)

    def forward(self, encoder_x, decoder_x, target=None, decoder_time=None):
        """
        Forward pass of the Informer model implementing generative inference.

        The decoder input is constructed by concatenating:
          - X_token: a guiding (prompt) sequence (decoder_x) providing context.
          - X_0: a placeholder segment.

        During training (teacher forcing), X_0 is replaced with the ground-truth target.
        During inference, X_0 is initialized as zeros.

        The decoder operates on the full concatenated sequence without strict autoregressive masking,
        and only the outputs corresponding to X_0 (the prediction segment) are used.
        """
        # Encode the input sequence
        enc_input = self.input_projection(encoder_x)  # (B, encoder_length, d_model)
        enc_input = self.pos_encoder(enc_input)
        memory = self.encoder(enc_input)

        # Prepare decoder input for generative inference (one-shot decoding)
        if target is not None:
            dec_input = torch.cat([decoder_x, target], dim=1)  # (B, decoder_length + prediction_length, input_dim)
        else:
            placeholder = torch.zeros(decoder_x.size(0), self.prediction_length, self.input_dim, device=decoder_x.device)
            dec_input = torch.cat([decoder_x, placeholder], dim=1)

        dec_input = self.input_projection(dec_input)  # (B, L_dec, d_model)
        dec_input = self.pos_decoder(dec_input)
        if decoder_time is not None:
            # Add global time embeddings if available.
            # Assume decoder_time shape: (B, decoder_length + prediction_length, 6)
            time_emb = self.time_embed(decoder_time)  # (B, L_dec, d_model)
            dec_input = dec_input + time_emb

        # Decode with cross-attention using the encoder output.
        # Note: The decoder attends to the entire concatenated sequence without causal masking.
        dec_output = self.decoder(dec_input, memory)

        # Project the decoder output back to the original feature dimension.
        output = self.output_projection(dec_output)
        
        # Return only the segment corresponding to the prediction placeholders (X_0).
        return output[:, -self.prediction_length:, :]

# ------------------------------
# Training & Evaluation Functions
# ------------------------------
def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Record the start time of the epoch
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        # Create a progress bar for each epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            encoder = batch['encoder'].to(device)
            decoder = batch['decoder'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            # Forward with teacher forcing by providing the target
            output = model(encoder, decoder, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase with additional metric computation
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

        # Calculate additional epoch details
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%, Duration: {epoch_duration:.2f}s, Best Val Loss: {best_val_loss:.6f}, LR: {current_lr:.6e}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model state
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
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    
    # Concatenate batches together
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    return predictions, targets

def compute_metrics(pred, target):
    """
    Computes evaluation metrics: MAE, RMSE and MAPE.
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    mae = np.mean(np.abs(pred_np - target_np))
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
    mape = np.mean(np.abs((pred_np - target_np) / (target_np + 1e-5))) * 100
    return mae, rmse, mape

# ------------------------------
# Main Function: Integration with DataLoader and Training Loop
# ------------------------------
if __name__ == "__main__":
    # Hyperparameter configuration
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
    batch_size = 64
    num_epochs = 25
    learning_rate = 4e-5


    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Set random seed for reproducibility
    seed = 69
    set_random_seed(seed)
    print(f"Random seed set to: {seed}... hehe")

    print(f"Using device: {device}")

    # Initialize the custom data loader (CSV path should be adjusted as needed)
    csv_path = "data/processed/SPY.csv"
    stock_data_loader = StockDataLoader(csv_path, use_cyclical_encoding=True)
    stock_data_loader.prepare_datasets()
    train_loader, val_loader, test_loader = stock_data_loader.get_dataloaders(batch_size=batch_size)

    # Initialize the Informer model with ProbSparse self-attention and global time embedding enabled
    model = Informer(input_dim=input_dim, d_model=d_model, d_ff=d_ff, nhead=nhead,
                     enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout,
                     encoder_length=encoder_length, decoder_length=decoder_length,
                     prediction_length=prediction_length,
                     use_prob_sparse=True)        # enable ProbSparse self-attention
    model.to(device)

    # Define loss function and optimizer (MSE loss for regression)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device)

    # Evaluate on test set
    predictions, targets = evaluate_model(model, test_loader, criterion, device)
    mae, rmse, mape = compute_metrics(predictions, targets)
    print(f"Test Metrics - MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%") 

    # ------------------------------
    # New Code: Save Plot and CSV of Test Predictions vs Ground Truth
    # ------------------------------
    # Create a folder "results" if it doesn't exist
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Convert predictions and targets to NumPy arrays
    pred_np = predictions.cpu().numpy()   # shape: (num_samples, prediction_length, input_dim)
    target_np = targets.cpu().numpy()
    
    # For plotting, choose the first sample and the first feature for demonstration.
    sample_idx = 0
    feature_idx = 0
    pred_series = pred_np[sample_idx, :, feature_idx]
    target_series = target_np[sample_idx, :, feature_idx]
    time_steps = range(1, len(pred_series) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, target_series, label="Ground Truth", marker='o')
    plt.plot(time_steps, pred_series, label="Prediction", marker='x')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Ground Truth vs Prediction (Sample {sample_idx}, Feature {feature_idx})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(results_folder, "ground_truth_vs_prediction.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved prediction plot to {plot_path}")
    
    # Save raw predictions vs ground truth in a CSV file for the first feature across all test samples.
    num_samples, pred_length, _ = pred_np.shape
    # Flatten sample and time steps for the first feature...
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
    csv_path = os.path.join(results_folder, "test_predictions_vs_ground_truth.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw predictions and ground truth data to {csv_path}") 
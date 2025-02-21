import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
        self.scaling = 1 / math.sqrt(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        # query, key, value: (B, L, d_model)
        B, L, d_model = query.size()
        head_dim = d_model // self.num_heads
        
        def reshape(x):
            # (B, L, d_model) -> (B, num_heads, L, head_dim)
            return x.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        
        q = reshape(query)
        k = reshape(key)
        v = reshape(value)
        
        # Compute scaled dot-product attention scores: (B, num_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Determine top-k keys: using u = ceil(log(L+1))
        u = max(1, int(math.ceil(math.log(L + 1))))
        topk_scores, topk_indices = scores.topk(k=u, dim=-1)
        
        # Create a mask for top-k indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        
        # Set scores not in top-k to -infinity
        scores_masked = torch.where(mask, scores, torch.full_like(scores, float('-inf')))
        
        attn_weights = F.softmax(scores_masked, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, L, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

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
            # Convolutional distillation: reducing sequence length by half.
            # Input of shape (B, L, d_model) is transposed to (B, d_model, L)
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
            # Apply convolution, ELU activation, and max pooling to reduce L by half.
            x = x.transpose(1, 2)         # (B, d_model, L)
            x = self.elu(self.conv(x))
            x = self.pool(x)              # (B, d_model, L//2)
            x = x.transpose(1, 2)         # (B, L//2, d_model)
        return x

# ------------------------------
# Informer Encoder (Stacking Layers)
# ------------------------------
class InformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout, use_prob_sparse=True):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            apply_distill = True if i < num_layers - 1 else False
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
        x_self, _ = self.self_attn(x, x, x)
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
                 use_prob_sparse=True, time_feature_dim=5):
        super(Informer, self).__init__()
        self.input_dim = input_dim
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
        # Global Time Embedding for decoder (e.g., year, month, week, hour, minute)
        self.time_embedding = nn.Linear(time_feature_dim, d_model)

    def forward(self, encoder_x, decoder_x, target=None, decoder_time=None):
        """
        Forward pass of the Informer model.

        Parameters:
          - encoder_x: (B, encoder_length, input_dim)
          - decoder_x: (B, decoder_length, input_dim)  [guiding sequence]
          - target: (B, prediction_length, input_dim)   [ground truth; used in teacher forcing]
          - decoder_time: (B, decoder_length + prediction_length, time_feature_dim) [optional time features]

        Returns:
          - predictions: (B, prediction_length, input_dim)
        """
        # Encode the input sequence
        enc_input = self.input_projection(encoder_x)  # (B, encoder_length, d_model)
        enc_input = self.pos_encoder(enc_input)
        memory = self.encoder(enc_input)

        # Prepare decoder input
        if target is not None:
            # For training (teacher forcing): concatenate guiding + ground truth target
            dec_input = torch.cat([decoder_x, target], dim=1)  # (B, decoder_length + prediction_length, input_dim)
        else:
            # For inference: use only guiding sequence
            dec_input = decoder_x
        dec_input = self.input_projection(dec_input)  # (B, L_dec, d_model)
        dec_input = self.pos_decoder(dec_input)
        # Incorporate global time stamp embeddings if provided
        if decoder_time is not None:
            time_emb = self.time_embedding(decoder_time)  # (B, L_dec, d_model)
            dec_input = dec_input + time_emb
        # Decode with cross-attention using encoder output
        dec_output = self.decoder(dec_input, memory)
        # Project back to feature dimensionality
        output = self.output_projection(dec_output)
        
        # If teacher forcing used, only return the prediction part (last prediction_length tokens)
        if target is not None:
            return output[:, -self.prediction_length:, :]
        else:
            return output

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

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                encoder = batch['encoder'].to(device)
                decoder = batch['decoder'].to(device)
                target = batch['target'].to(device)
                output = model(encoder, decoder, target)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Calculate additional epoch details
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s, Best Val Loss: {best_val_loss:.6f}, LR: {current_lr:.6e}")
        
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
    batch_size = 32
    num_epochs = 20
    learning_rate = 4e-5

    # Device setup optimized for Mac ARM (MPS) and CUDA.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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
                     use_prob_sparse=True,      # enable ProbSparse self-attention
                     time_feature_dim=5)        # assuming 5 time features (e.g., year, month, week, hour, minute)
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
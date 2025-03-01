import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class PatchTST(nn.Module):
    def __init__(
        self,
        input_dim=7,
        output_dim=1,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        seq_len=100,
        pred_len=30,
        fc_dropout=0.1,
        head_dropout=0.1
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=n_layers,
            dropout=dropout
        )
        
        # Prediction head
        self.head = PredictionHead(
            d_model=d_model,
            output_dim=output_dim,
            pred_len=pred_len,
            head_dropout=head_dropout,
            fc_dropout=fc_dropout
        )

        # Reconstruction head for pretraining
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(d_model, patch_len * input_dim)
        )
        
    def forward(self, x):  # x: [batch_size, seq_len, input_dim]
        # Patch embedding with CLS token
        x = self.patch_embedding(x, add_cls=True)  # [batch_size, num_patches+1, d_model]
        
        # Transformer encoder
        x = self.encoder(x)  # [batch_size, num_patches+1, d_model]
        
        # Prediction
        x = self.head(x)  # [batch_size, pred_len, output_dim]
        
        return x

    def forward_pretrain(self, x):  # x: [batch_size, seq_len, input_dim]
        # Use patch embedding without adding CLS token
        x = self.patch_embedding(x, add_cls=False)  # [batch_size, num_patches, d_model]

        # Transformer encoder
        x = self.encoder(x)  # [batch_size, num_patches, d_model]

        # Reconstruction
        x = self.reconstruction_head(x)  # [batch_size, num_patches, patch_len * input_dim]
        x = x.view(x.shape[0], -1, self.patch_len, self.input_dim)  # [batch_size, num_patches, patch_len, input_dim]

        # Merge patch reconstruction back to sequence
        x = x.view(x.shape[0], -1, self.input_dim)  # [batch_size, reconstructed_seq_len, input_dim]

        # Trim to original sequence length if needed
        if x.shape[1] > self.seq_len:
            x = x[:, :self.seq_len, :]

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, input_dim, d_model, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Linear projection
        self.linear = nn.Linear(patch_len * input_dim, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x, add_cls=True):  # x: [batch_size, seq_len, input_dim]
        # Compute patches from raw input
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  
        patches = rearrange(patches, 'b np pl c -> b np (pl c)')

        # Linear projection of patches
        x = self.linear(patches)  # [batch_size, num_patches, d_model]

        if add_cls:
            # Add CLS token if required
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)

        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections and reshape
        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PredictionHead(nn.Module):
    def __init__(self, d_model, output_dim, pred_len, head_dropout, fc_dropout):
        super().__init__()
        
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(fc_dropout)
        self.linear = nn.Linear(d_model, pred_len * output_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.pred_len = pred_len
        self.output_dim = output_dim
        
    def forward(self, x):
        x = x[:, 0]  # Use CLS token
        x = self.dropout(x)
        x = self.linear(x)
        x = self.head_dropout(x)
        x = x.view(-1, self.pred_len)  # Changed to match target dimensions
        return x 
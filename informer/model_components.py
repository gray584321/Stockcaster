import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class MultiPathSparseAttention(nn.Module):
    """
    An improved sparse attention mechanism inspired by recent advances like DeepSeek's
    Native Sparse Attention (NSA), which uses multiple attention paths to better handle
    different aspects of the sequence data.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, sliding_window=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.sliding_window = sliding_window
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Token compressors for global path
        self.compressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.path_mixer = nn.Parameter(torch.ones(3) / 3)  # Learnable weights for each path
        
    def _reshape_for_multi_head(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, query, key, value, attn_mask=None):
        B, L, D = query.size()
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q_reshaped = self._reshape_for_multi_head(q)
        k_reshaped = self._reshape_for_multi_head(k)
        v_reshaped = self._reshape_for_multi_head(v)
        
        # 1. Global path (compressed attention)
        # Compress keys and values to reduce computation
        compression_ratio = min(4, max(1, L // 64))  # Adaptive compression ratio
        if compression_ratio > 1:
            # Apply compression to get fewer tokens
            k_compressed = k.view(B, -1, compression_ratio, D).mean(dim=2)
            v_compressed = v.view(B, -1, compression_ratio, D).mean(dim=2)
            k_global = self._reshape_for_multi_head(self.compressor(k_compressed))
            v_global = self._reshape_for_multi_head(v_compressed)
            
            # Global attention with compressed keys and values
            scores_global = torch.matmul(q_reshaped, k_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                # Need to adapt mask for compressed keys
                compressed_mask = attn_mask[:, :, ::compression_ratio]
                scores_global = scores_global.masked_fill(compressed_mask.unsqueeze(1), float('-inf'))
            attn_weights_global = F.softmax(scores_global, dim=-1)
            attn_weights_global = self.dropout(attn_weights_global)
            global_output = torch.matmul(attn_weights_global, v_global)
        else:
            # If sequence is short, use regular attention
            k_global = k_reshaped
            v_global = v_reshaped
            scores_global = torch.matmul(q_reshaped, k_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores_global = scores_global.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
            attn_weights_global = F.softmax(scores_global, dim=-1)
            attn_weights_global = self.dropout(attn_weights_global)
            global_output = torch.matmul(attn_weights_global, v_global)
        
        # 2. Local path (sliding window attention)
        # Create a sliding window mask to focus on nearby tokens
        window_size = min(self.sliding_window, L)
        local_mask = torch.ones(L, L, dtype=torch.bool, device=query.device)
        for i in range(L):
            start = max(0, i - window_size // 2)
            end = min(L, i + window_size // 2 + 1)
            local_mask[i, start:end] = False
        
        # Apply sliding window attention
        scores_local = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_local = scores_local.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        if attn_mask is not None:
            scores_local = scores_local.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn_weights_local = F.softmax(scores_local, dim=-1)
        attn_weights_local = self.dropout(attn_weights_local)
        local_output = torch.matmul(attn_weights_local, v_reshaped)
        
        # 3. Sparse path (top-k selection)
        # Use ProbSparse-like selection for the most important tokens
        scores_selection = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / math.sqrt(self.head_dim)
        m1 = torch.logsumexp(scores_selection, dim=-1) - math.log(L)
        m2 = scores_selection.mean(dim=-1)
        importance_scores = m1 - m2
        
        # Select top-k queries for each head
        u = max(1, int(math.ceil(math.log(L + 1))))
        _, topk_query_indices = importance_scores.topk(k=u, dim=-1)
        
        # Gather the selected queries
        batch_indices = torch.arange(B, device=query.device).view(B, 1, 1, 1).expand(B, self.num_heads, u, 1)
        head_indices = torch.arange(self.num_heads, device=query.device).view(1, self.num_heads, 1, 1).expand(B, self.num_heads, u, 1)
        gather_indices = torch.cat([batch_indices, head_indices, topk_query_indices.unsqueeze(-1)], dim=-1)
        
        # Compute attention for selected queries
        q_selected = q_reshaped.gather(2, topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, self.head_dim))
        scores_selected = torch.matmul(q_selected, k_reshaped.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # Create mask for selected queries
            selected_mask = torch.gather(attn_mask.unsqueeze(1).expand(B, self.num_heads, L, L), 2, 
                                       topk_query_indices.unsqueeze(-1).expand(B, self.num_heads, u, L))
            scores_selected = scores_selected.masked_fill(selected_mask, float('-inf'))
        
        attn_weights_selected = F.softmax(scores_selected, dim=-1)
        attn_weights_selected = self.dropout(attn_weights_selected)
        selected_output = torch.matmul(attn_weights_selected, v_reshaped)
        
        # Create a placeholder for the full output based on the sparse selection
        sparse_output = torch.zeros_like(q_reshaped)
        
        # Convert sparse_output to a list for scatter operation
        sparse_output_list = list(sparse_output.unbind(dim=2))
        selected_output_list = list(selected_output.unbind(dim=2))
        
        # Scatter the selected outputs back to their original positions
        for i, idx in enumerate(topk_query_indices.unbind(dim=2)):
            for b in range(B):
                for h in range(self.num_heads):
                    sparse_output_list[idx[b, h].item()][b, h] = selected_output_list[i][b, h]
        
        # Rebuild the sparse_output tensor
        sparse_output = torch.stack(sparse_output_list, dim=2)
        
        # Combine all paths with learnable weights
        path_weights = F.softmax(self.path_mixer, dim=0)
        combined_output = (
            path_weights[0] * global_output + 
            path_weights[1] * local_output + 
            path_weights[2] * sparse_output
        )
        
        # Reshape back to original dimensions
        combined_output = combined_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(combined_output)
        
        # For compatibility with the other attention modules
        attn_weights = (attn_weights_global + attn_weights_local) / 2
        
        return output, attn_weights


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
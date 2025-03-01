import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ProbSparseAttention(nn.Module):
    """
    ProbSparse attention mechanism from Informer for improved complexity with long sequences
    Reduces time complexity from O(L²) to O(L·log(L))
    """
    def __init__(self, embed_dim, num_heads, factor=5, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.factor = factor  # Factor determining sparsity level
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        
        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_K))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Find the Top_k query with maximum similarity
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], 
                     torch.arange(H)[None, :, None], 
                     M_top, :]  # factor*ln(L_Q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_Q)*L_K
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.training:
            # For inference, just use mean of V
            V_mean = V.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, D)
        else:
            # For training, initialize with zeros
            context = torch.zeros((B, H, L_Q, D), device=V.device)
        return context
    
    def _update_context(self, context_in, V, scores, index):
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1)  # [B, H, L_Q, L_K]
        
        context_in[torch.arange(B)[:, None, None], 
                  torch.arange(H)[None, :, None], 
                  index, :] = torch.matmul(attn, V)
        return context_in
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.q_proj(query)  # (B, L, E)
        k = self.k_proj(key)    # (B, L, E)
        v = self.v_proj(value)  # (B, L, E)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        
        # ProbSparse Attention
        L_Q, L_K = q.shape[2], k.shape[2]
        u = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = min(u, L_K)
        n_top = min(u, L_Q)
        
        # Calculate sparse attention scores
        scores, index = self._prob_QK(q, k, u, n_top)  # (B, H, n_top, L_K)
        
        # Apply mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        
        # Initialize context
        context = self._get_initial_context(v, L_Q)  # (B, H, L_Q, D)
        
        # Update context with sparse attention
        context = self._update_context(context, v, scores, index)
        
        # Combine heads and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        
        return output, None  # Return None for attn_weights to match interface


class StandardSelfAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism for time series
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.q_proj(query)  # (B, L, E)
        k = self.k_proj(key)    # (B, L, E)
        v = self.v_proj(value)  # (B, L, E)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)
        
        # Apply mask if provided
        if attn_mask is not None:
            # Fix: Ensure mask has the right shape for multi-head attention
            if attn_mask.dim() == 2:
                # If mask is 2D (seq_len x seq_len), expand it for batch and heads
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # (B, H, L, D)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (B, L, E)
        output = self.out_proj(context)  # (B, L, E)
        output = self.proj_dropout(output)
        
        return output, attn_weights


class SeriesDecomposition(nn.Module):
    """
    Series decomposition module from Autoformer for trend and seasonality modeling
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x):
        """
        Decompose the series into trend and seasonal components
        x: input series [B, L, D]
        returns: trend and seasonal components
        """
        # Extract trend component with moving average
        x_transposed = x.transpose(1, 2)  # [B, D, L]
        x_avg = self.avg(x_transposed)
        x_avg = x_avg.transpose(1, 2)  # [B, L, D]
        
        # Handle padding issues
        if x_avg.size(1) != x.size(1):
            padding = x.size(1) - x_avg.size(1)
            x_avg = F.pad(x_avg.transpose(1, 2), (0, padding)).transpose(1, 2)
        
        # Seasonal component is the residual
        x_seasonal = x - x_avg
        
        return x_avg, x_seasonal


class AutoCorrelationLayer(nn.Module):
    """
    Auto-correlation layer from Autoformer for capturing periodic patterns
    """
    def __init__(self, dim_model, n_heads, factor=1, dropout=0.1):
        super().__init__()
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.factor = factor
        self.dropout = dropout
        self.temperature = nn.Parameter(torch.ones(n_heads, 1, 1))
        
        self.query_projection = nn.Linear(dim_model, dim_model)
        self.key_projection = nn.Linear(dim_model, dim_model)
        self.value_projection = nn.Linear(dim_model, dim_model)
        self.out_projection = nn.Linear(dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout)
        
    def time_delay_aggregate(self, values, corr):
        """
        Time delay aggregation based on correlation
        """
        batch, head, channel, length = values.shape
        # Weighted aggregation
        corr = corr.transpose(-1, -2)
        output = torch.matmul(values, corr)
        return output
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        
        # Project inputs
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        
        # Compute auto-correlation
        queries = queries.permute(0, 2, 1, 3)  # [B, H, L, D]
        keys = keys.permute(0, 2, 1, 3)  # [B, H, S, D]
        values = values.permute(0, 2, 3, 1)  # [B, H, D, S]
        
        # Compute auto-correlation
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        corr = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(corr, dim=-1)
        
        # Apply softmax
        corr = corr / self.temperature
        corr = F.softmax(corr, dim=-1)
        corr = self.dropout_layer(corr)
        
        # Apply attention to values
        output = self.time_delay_aggregate(values, corr)  # [B, H, D, L]
        output = output.permute(0, 3, 1, 2).contiguous().view(B, L, -1)
        output = self.out_projection(output)
        
        return output, None


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


class TimeFeatureEmbedding(nn.Module):
    """Enhanced time embedding for time series that captures temporal patterns"""
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
        
        # Additional time features for financial time series
        self.is_weekend_embedding = nn.Embedding(2, d_model)  # 0=weekday, 1=weekend
        self.is_holiday_embedding = nn.Embedding(2, d_model)  # 0=regular day, 1=holiday
        self.quarter_embedding = nn.Embedding(4, d_model)     # 1-4 quarters
        
        # Linear layer to combine all features
        self.feature_combiner = nn.Linear(d_model, d_model)
        
    def forward(self, time_tensor):
        # Basic time features
        year = (time_tensor[:,:,0].long() - self.year_offset).clamp(min=0)
        month = (time_tensor[:,:,1].long() - 1).clamp(min=0)
        week = (time_tensor[:,:,2].long() - 1).clamp(min=0)
        day = (time_tensor[:,:,3].long() - 1).clamp(min=0)
        hour = time_tensor[:,:,4].long()
        minute = time_tensor[:,:,5].long()
        
        # Calculate additional features if available
        # (assuming weekday/holiday info might be in columns 6 and 7)
        quarter = ((month // 3) % 4).long()
        
        # If weekend and holiday info are available in the time tensor
        is_weekend = torch.zeros_like(day)
        is_holiday = torch.zeros_like(day)
        if time_tensor.size(2) > 6:
            is_weekend = time_tensor[:,:,6].long()
        if time_tensor.size(2) > 7:
            is_holiday = time_tensor[:,:,7].long()
        
        # Combine all embeddings
        combined = (self.year_embedding(year) + 
                   self.month_embedding(month) +
                   self.week_embedding(week) + 
                   self.day_embedding(day) +
                   self.hour_embedding(hour) + 
                   self.minute_embedding(minute) +
                   self.quarter_embedding(quarter) + 
                   self.is_weekend_embedding(is_weekend) +
                   self.is_holiday_embedding(is_holiday))
        
        # Final projection
        return self.feature_combiner(combined)


class ConvFeatureExtraction(nn.Module):
    """
    Convolutional feature extraction module for time series before transformer
    """
    def __init__(self, input_dim, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: [B, L, D]
        x_conv = x.transpose(1, 2)  # [B, D, L]
        x_conv = self.conv_layers(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, L, D]
        output = self.projection(x_conv)
        return output


class TSTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, attention_type='standard', dropout=0.1):
        super().__init__()
        if attention_type == 'standard':
            self.self_attn = StandardSelfAttention(d_model, nhead, dropout)
        elif attention_type == 'probsparse':
            self.self_attn = ProbSparseAttention(d_model, nhead, dropout=dropout)
        elif attention_type == 'autocorrelation':
            self.self_attn = AutoCorrelationLayer(d_model, nhead, dropout=dropout)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Decomposition for handling trend and seasonality
        self.decomp1 = SeriesDecomposition(kernel_size=25)
        self.decomp2 = SeriesDecomposition(kernel_size=25)
        
    def forward(self, src, src_mask=None):
        # Apply decomposition before self-attention
        src_trend, src_seasonal = self.decomp1(src)
        
        # Self attention block with residual - on seasonal component
        seasonal_out, _ = self.self_attn(src_seasonal, src_seasonal, src_seasonal, attn_mask=src_mask)
        seasonal_out = src_seasonal + self.dropout1(seasonal_out)
        seasonal_out = self.norm1(seasonal_out)
        
        # Feed forward block with residual
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(seasonal_out))))
        ff_out = seasonal_out + self.dropout2(ff_out)
        ff_out = self.norm2(ff_out)
        
        # Recombine with trend component
        trend_part, seasonal_part = self.decomp2(ff_out)
        out = trend_part + src_trend + seasonal_part
        
        return out


class TSTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, attention_type='standard', dropout=0.1):
        super().__init__()
        if attention_type == 'standard':
            self.self_attn = StandardSelfAttention(d_model, nhead, dropout)
            self.cross_attn = StandardSelfAttention(d_model, nhead, dropout)
        elif attention_type == 'probsparse':
            self.self_attn = ProbSparseAttention(d_model, nhead, dropout=dropout)
            self.cross_attn = ProbSparseAttention(d_model, nhead, dropout=dropout)
        elif attention_type == 'autocorrelation':
            self.self_attn = AutoCorrelationLayer(d_model, nhead, dropout=dropout)
            self.cross_attn = AutoCorrelationLayer(d_model, nhead, dropout=dropout)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Decomposition for handling trend and seasonality
        self.decomp1 = SeriesDecomposition(kernel_size=25)
        self.decomp2 = SeriesDecomposition(kernel_size=25)
        self.decomp3 = SeriesDecomposition(kernel_size=25)
        
    def forward(self, tgt, memory, tgt_mask=None):
        # Apply decomposition before self-attention
        tgt_trend, tgt_seasonal = self.decomp1(tgt)
        
        # Self attention block with residual - on seasonal component
        seasonal_out, _ = self.self_attn(tgt_seasonal, tgt_seasonal, tgt_seasonal, attn_mask=tgt_mask)
        seasonal_out = tgt_seasonal + self.dropout1(seasonal_out)
        seasonal_out = self.norm1(seasonal_out)
        
        # Recombine with trend
        post_sa_trend, post_sa_seasonal = self.decomp2(seasonal_out)
        seasonal_out = post_sa_trend + tgt_trend + post_sa_seasonal
        
        # Cross attention block with residual
        cross_trend, cross_seasonal = self.decomp2(seasonal_out)
        cross_out, _ = self.cross_attn(cross_seasonal, memory, memory)
        cross_out = cross_seasonal + self.dropout2(cross_out)
        cross_out = self.norm2(cross_out)
        
        # Recombine with trend
        post_ca_trend, post_ca_seasonal = self.decomp3(cross_out)
        cross_out = post_ca_trend + cross_trend + post_ca_seasonal
        
        # Feed forward block with residual
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(cross_out))))
        ff_out = cross_out + self.dropout3(ff_out)
        ff_out = self.norm3(ff_out)
        
        return ff_out


class TSTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, attention_type='standard', dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTransformerEncoderLayer(d_model, nhead, d_ff, attention_type, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)
        return self.norm(output)


class TSTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, attention_type='standard', dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTransformerDecoderLayer(d_model, nhead, d_ff, attention_type, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask)
        return self.norm(output)


class TimeSeriesTransformer(nn.Module):
    """
    Enhanced Time Series Transformer model for forecasting
    Includes ProbSparse attention, trend-cyclical decomposition,
    and convolutional feature extraction
    """
    def __init__(self, input_dim=12, d_model=64, d_ff=256, nhead=8,
                 enc_layers=3, dec_layers=3, dropout=0.1,
                 encoder_length=96, decoder_length=48, prediction_length=24,
                 use_time_features=True, attention_type='probsparse',
                 use_decomposition=True, use_convolutional=True):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.prediction_length = prediction_length
        self.use_decomposition = use_decomposition
        
        # Embedding layers
        if use_convolutional:
            self.feature_extraction = ConvFeatureExtraction(input_dim, d_model, dropout=dropout)
            self.input_projection = nn.Identity()  # ConvFeatureExtraction already projects to d_model
        else:
            self.feature_extraction = nn.Identity()
            self.input_projection = nn.Linear(input_dim, d_model)
            
        self.pos_encoder = PositionalEncoding(d_model, max_len=encoder_length)
        self.pos_decoder = PositionalEncoding(d_model, max_len=decoder_length + prediction_length)
        
        if use_time_features:
            self.time_embed = TimeFeatureEmbedding(d_model)
        else:
            self.time_embed = None
            
        # Series decomposition for trend and seasonality
        if use_decomposition:
            self.enc_decomp = SeriesDecomposition(kernel_size=25)
            self.dec_decomp = SeriesDecomposition(kernel_size=25)
        
        # Transformer modules with selected attention type
        self.encoder = TSTransformerEncoder(
            num_layers=enc_layers, 
            d_model=d_model,
            nhead=nhead, 
            d_ff=d_ff, 
            attention_type=attention_type,
            dropout=dropout
        )
        
        self.decoder = TSTransformerDecoder(
            num_layers=dec_layers, 
            d_model=d_model,
            nhead=nhead, 
            d_ff=d_ff, 
            attention_type=attention_type,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, encoder_x, decoder_x, target=None, decoder_time=None, scheduled_sampling_prob=0.0):
        # Apply convolutional feature extraction
        encoder_x = self.feature_extraction(encoder_x)
        decoder_x = self.feature_extraction(decoder_x)
        if target is not None:
            target = self.feature_extraction(target)
        
        # Process encoder inputs
        enc_input = self.input_projection(encoder_x)
        enc_input = self.pos_encoder(enc_input)
        
        # Add time features if available
        if decoder_time is not None and self.time_embed is not None:
            # Fix: Only use available time steps without exceeding dimension size
            available_length = min(decoder_time.size(1), self.encoder_length)
            encoder_time = decoder_time[:, :available_length, :]
            # If encoder_time is shorter than enc_input, only add time embeddings to available positions
            time_embed = self.time_embed(encoder_time)
            enc_input[:, :available_length, :] = enc_input[:, :available_length, :] + time_embed
        
        # Apply series decomposition if enabled
        if self.use_decomposition:
            enc_trend, enc_seasonal = self.enc_decomp(enc_input)
            # Pass only seasonal component through transformer
            memory_seasonal = self.encoder(enc_seasonal)
            # Recombine with trend
            memory = memory_seasonal + enc_trend
        else:
            # Standard processing without decomposition
            memory = self.encoder(enc_input)
        
        # Handle different decoding scenarios
        if scheduled_sampling_prob == 0.0 or target is None:
            # Teacher forcing or inference mode
            if target is not None:
                dec_input = torch.cat([decoder_x, target], dim=1)
            else:
                placeholder = torch.zeros(decoder_x.size(0), self.prediction_length, self.input_dim, device=decoder_x.device)
                dec_input = torch.cat([decoder_x, placeholder], dim=1)
            
            # Process decoder inputs
            dec_input = self.input_projection(dec_input)
            dec_input = self.pos_decoder(dec_input)
            
            # Add time features if available
            if decoder_time is not None and self.time_embed is not None:
                # Fix: Only use available time steps without exceeding dimension size
                available_length = min(decoder_time.size(1), dec_input.size(1))
                dec_time = decoder_time[:, :available_length, :]
                time_embed = self.time_embed(dec_time)
                dec_input[:, :available_length, :] = dec_input[:, :available_length, :] + time_embed
            
            # Create causal mask for autoregressive decoding
            tgt_mask = self._generate_square_subsequent_mask(dec_input.size(1)).to(dec_input.device)
            
            # Apply series decomposition if enabled
            if self.use_decomposition:
                dec_trend, dec_seasonal = self.dec_decomp(dec_input)
                # Pass seasonal component through transformer
                dec_output_seasonal = self.decoder(dec_seasonal, memory, tgt_mask=tgt_mask)
                # Recombine with trend
                dec_output = dec_output_seasonal + dec_trend
            else:
                # Standard processing without decomposition
                dec_output = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
                
            output = self.output_projection(dec_output)
            
            # Return only the prediction part
            return output[:, -self.prediction_length:, :]
        else:
            # Scheduled sampling: iterative decoding
            # Start with decoder input
            dec_seq = decoder_x  # shape: (B, decoder_length, input_dim)
            predictions = []
            B = decoder_x.size(0)
            
            for t in range(self.prediction_length):
                # Process current decoder sequence
                dec_seq_proj = self.input_projection(dec_seq)
                dec_seq_proj = self.pos_decoder(dec_seq_proj)
                
                # Add time features if available
                if decoder_time is not None and self.time_embed is not None:
                    # Fix: Only use available time steps without exceeding dimension size
                    available_length = min(decoder_time.size(1), dec_seq_proj.size(1))
                    dec_time = decoder_time[:, :available_length, :]
                    time_embed = self.time_embed(dec_time)
                    dec_seq_proj[:, :available_length, :] = dec_seq_proj[:, :available_length, :] + time_embed
                
                # Create causal mask
                tgt_mask = self._generate_square_subsequent_mask(dec_seq_proj.size(1)).to(dec_seq_proj.device)
                
                # Apply series decomposition if enabled
                if self.use_decomposition:
                    dec_trend, dec_seasonal = self.dec_decomp(dec_seq_proj)
                    # Pass seasonal component through transformer
                    dec_output_seasonal = self.decoder(dec_seasonal, memory, tgt_mask=tgt_mask)
                    # Recombine with trend
                    dec_output = dec_output_seasonal + dec_trend
                else:
                    # Standard processing without decomposition
                    dec_output = self.decoder(dec_seq_proj, memory, tgt_mask=tgt_mask)
                    
                full_output = self.output_projection(dec_output)
                
                # Get prediction for current timestep
                new_pred = full_output[:, -1:, :]  # (B, 1, input_dim)
                
                # Scheduled sampling decision
                if target is not None:
                    ground_truth = target[:, t:t+1, :]  # (B, 1, input_dim)
                    mask = (torch.rand(B, 1, 1, device=new_pred.device) < scheduled_sampling_prob).float()
                    input_step = mask * new_pred + (1 - mask) * ground_truth
                else:
                    input_step = new_pred
                
                # Append to decoder sequence
                dec_seq = torch.cat([dec_seq, input_step], dim=1)
                predictions.append(new_pred)
            
            # Concatenate all predictions
            predictions_seq = torch.cat(predictions, dim=1)  # (B, prediction_length, input_dim)
            return predictions_seq 


class PatchEmbedding(nn.Module):
    """
    Patch Embedding module for PatchTST model.
    Splits time series into patch segments as input tokens.
    
    Args:
        input_dim: Number of feature dimensions
        d_model: Hidden dimension size
        patch_len: Length of each patch
        stride: Stride between patches
        padding: Padding mode ('end' or None)
    """
    def __init__(self, input_dim, d_model, patch_len, stride, padding='end'):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        
        # Conv1d: (input_dim, patch_len) -> (d_model, 1)
        # Use Conv1d for patch embedding (kernel_size=patch_len)
        self.tokenizer = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=d_model, 
            kernel_size=patch_len,
            stride=stride, 
            padding=0
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.tokenizer.weight)
        
    def forward(self, x):
        # x: [B, L, D]  where L=sequence_length, D=input_dim
        B, L, D = x.shape
        
        # Handle padding if needed to ensure all patches fit
        if self.padding == 'end' and L % self.stride != 0:
            padding_len = self.stride - (L % self.stride)
            x = F.pad(x, (0, 0, 0, padding_len))
            L = L + padding_len
        
        # Reshape and prepare for 1D convolution: [B, D, L]
        x = x.transpose(1, 2)
        
        # Apply tokenization to create patches through convolution
        # Output shape: [B, d_model, num_patches]
        x = self.tokenizer(x)  
        
        # Rearrange to [B, num_patches, d_model]
        x = x.transpose(1, 2)
        
        return x


class ChannelIndependentMultiHeadAttention(nn.Module):
    """
    Channel-independent multi-head attention.
    Process each feature/channel separately to preserve channel-specific patterns.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.q_proj(query)  # (B, L, E)
        k = self.k_proj(key)    # (B, L, E)
        v = self.v_proj(value)  # (B, L, E)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        
        # Apply mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # (B, H, L, D)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (B, L, E)
        output = self.out_proj(context)  # (B, L, E)
        output = self.proj_dropout(output)
        
        return output, attn_weights


class ChannelIndependentEncoderLayer(nn.Module):
    """
    Channel-independent Transformer encoder layer for PatchTST.
    Each feature dimension is processed independently to preserve channel patterns.
    """
    def __init__(self, d_model, nhead, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = ChannelIndependentMultiHeadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, src, src_mask=None):
        # Multi-head self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class ChannelIndependentEncoder(nn.Module):
    """
    Channel-independent Transformer encoder for PatchTST.
    Stack of encoder layers that process each feature independently.
    """
    def __init__(self, num_layers, d_model, nhead, d_ff=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ChannelIndependentEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)
        
        return self.norm(output)


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer (PatchTST) model.
    Implements patch-wise token embedding and channel-independent processing
    for improved long-horizon forecasting.
    
    Based on: "PatchTST: A Time Series Transformer Based on Patch-wise Tokenization for Long-term Forecasting" 
    (Nie et al., 2023)
    """
    def __init__(self, input_dim=12, d_model=64, nhead=8, num_encoder_layers=3, d_ff=256, 
                dropout=0.1, encoder_length=96, prediction_length=24, patch_len=16, stride=8):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.patch_len = min(patch_len, encoder_length)
        self.stride = min(stride, patch_len)
        
        # Patch embedding: (batch_size, seq_len, input_dim) -> (batch_size, num_patches, d_model)
        self.patch_embedding = PatchEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding='end'
        )
        
        # Positional encoding for patches
        max_patches = math.ceil(encoder_length / self.stride)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_patches)
        
        # Channel-independent encoder layers
        self.encoder = ChannelIndependentEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            nhead=nhead,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Prediction head: project back to output dimensions
        self.projection = nn.Linear(d_model, prediction_length * input_dim)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, x_mark=None):
        # x: [batch_size, encoder_length, input_dim]
        
        # Create patch embeddings
        x = self.patch_embedding(x)  # [batch_size, num_patches, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through encoder
        x = self.encoder(x)  # [batch_size, num_patches, d_model]
        
        # Use mean of patch representations for prediction
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Project to output dimension
        output = self.projection(x)  # [batch_size, prediction_length * input_dim]
        
        # Reshape to [batch_size, prediction_length, input_dim]
        output = output.view(-1, self.prediction_length, self.input_dim)
        
        return output 
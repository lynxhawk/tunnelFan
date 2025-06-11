import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ImprovedPositionalEncoding(nn.Module):
    """改进的位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(ImprovedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimplifiedProbAttention(nn.Module):
    """简化版概率稀疏注意力 - 修复原版问题"""
    def __init__(self, factor=3, scale=None, attention_dropout=0.1):
        super(SimplifiedProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        
        # 转置以适配计算
        queries = queries.transpose(2, 1)  # B, H, L, D
        keys = keys.transpose(2, 1)        # B, H, S, D  
        values = values.transpose(2, 1)    # B, H, S, D
        
        # 如果序列长度较小，直接使用全注意力
        if L <= 256:
            scores = torch.matmul(queries, keys.transpose(-2, -1))
            scale = self.scale or 1./math.sqrt(D)
            scores = scores * scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, values)
            return out.transpose(2, 1), attn.mean(dim=1)
        
        # 对于长序列，使用简化的稀疏注意力
        # 计算采样数量
        U_part = max(1, L // self.factor)
        u = max(1, L // self.factor)
        
        # 随机采样key位置
        if S > U_part:
            sample_indices = torch.randperm(S)[:U_part]
            keys_sample = keys[:, :, sample_indices, :]
            values_sample = values[:, :, sample_indices, :]
        else:
            keys_sample = keys
            values_sample = values
            
        # 计算注意力分数
        scores = torch.matmul(queries, keys_sample.transpose(-2, -1))
        scale = self.scale or 1./math.sqrt(D)
        scores = scores * scale
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        out = torch.matmul(attn, values_sample)
        
        # 生成完整的注意力权重用于可视化
        full_attn = torch.zeros(B, H, L, S, device=queries.device)
        if S > U_part:
            full_attn[:, :, :, sample_indices] = attn
        else:
            full_attn = attn
            
        return out.transpose(2, 1), full_attn.mean(dim=1)


class FixedAttentionLayer(nn.Module):
    """修复版注意力层"""
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(FixedAttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        
    def forward(self, queries, keys, values, mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 投影
        queries = self.query_projection(queries).view(B, L, H, self.d_keys)
        keys = self.key_projection(keys).view(B, S, H, self.d_keys)
        values = self.value_projection(values).view(B, S, H, self.d_values)
        
        # 注意力计算
        out, attn = self.inner_attention(queries, keys, values, mask)
        
        # 输出投影
        out = out.contiguous().view(B, L, -1)
        out = self.out_projection(out)
        
        return out, attn


class FixedEncoderLayer(nn.Module):
    """修复版编码器层"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(FixedEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        
        # 使用更稳定的前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_out, attn = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attn


class ImprovedInformerEncoder(nn.Module):
    """改进版Informer编码器"""
    def __init__(self, d_model, n_heads, d_ff, depth, dropout=0.1, factor=3):
        super(ImprovedInformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.depth = depth
        
        # 位置编码
        self.pos_encoding = ImprovedPositionalEncoding(d_model, dropout=dropout)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            FixedEncoderLayer(
                FixedAttentionLayer(
                    SimplifiedProbAttention(factor=factor, attention_dropout=dropout),
                    d_model, n_heads
                ),
                d_model, d_ff, dropout
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # 位置编码
        x = self.pos_encoding(x)
        
        # 通过编码器层
        attns = []
        for layer in self.encoder_layers:
            x, attn = layer(x, mask)
            attns.append(attn)
            
        x = self.norm(x)
        
        return x, attns


class FixedDirectInformerClassifier(nn.Module):
    """修复版直接Informer分类器"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=4,
                 d_model=128, n_heads=8, d_ff=256, depth=2,
                 factor=3, dropout_rate=0.1):
        super(FixedDirectInformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # 输入投影 - 使用更小的维度
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 改进的Informer编码器
        self.informer_encoder = ImprovedInformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=depth,
            dropout=dropout_rate,
            factor=factor
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Informer编码
        x, attns = self.informer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_length)
        x = self.adaptive_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(x)
        
        # 返回最后一层的注意力权重
        final_attention = attns[-1] if attns else torch.ones(batch_size, self.seq_length, device=x.device)
        
        return output, final_attention
    
    def get_latent(self, x):
        """获取潜在表示"""
        x = self.input_projection(x)
        x, _ = self.informer_encoder(x)
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x).squeeze(-1)
        return x


class OptimizedTransformerClassifier(nn.Module):
    """针对轴承故障诊断优化的Transformer分类器"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=4,
                 d_model=96, n_heads=6, num_layers=3, d_ff=192,
                 dropout=0.1, use_cnn_preprocessing=True):
        super(OptimizedTransformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        self.use_cnn_preprocessing = use_cnn_preprocessing
        
        if use_cnn_preprocessing:
            # 轻量级CNN预处理
            self.cnn_prep = nn.Sequential(
                nn.Conv1d(input_channels, d_model//2, kernel_size=7, padding=3),
                nn.BatchNorm1d(d_model//2),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(d_model//2, d_model, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.MaxPool1d(2)
            )
            # 序列长度减少4倍
            self.effective_seq_len = seq_length // 4
        else:
            # 直接线性投影
            self.input_projection = nn.Linear(input_channels, d_model)
            self.effective_seq_len = seq_length
        
        # 位置编码
        self.pos_encoding = ImprovedPositionalEncoding(d_model, self.effective_seq_len * 2, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 注意力池化
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        batch_size = x.size(0)
        
        if self.use_cnn_preprocessing:
            # CNN预处理
            x = x.transpose(1, 2)  # (batch_size, input_channels, seq_length)
            x = self.cnn_prep(x)   # (batch_size, d_model, seq_length//4)
            x = x.transpose(1, 2)  # (batch_size, seq_length//4, d_model)
        else:
            # 直接投影
            x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # 注意力池化
        attention_weights = self.attention_pooling(x)  # (batch_size, seq_length, 1)
        context = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(context)
        
        return output, attention_weights.squeeze(-1)
    
    def get_latent(self, x):
        """获取潜在表示"""
        batch_size = x.size(0)
        
        if self.use_cnn_preprocessing:
            x = x.transpose(1, 2)
            x = self.cnn_prep(x)
            x = x.transpose(1, 2)
        else:
            x = self.input_projection(x)
        
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        context = torch.mean(x, dim=1)
        
        return context


class LightweightCNNTransformer(nn.Module):
    """轻量级CNN+Transformer混合模型 - 专为轴承故障诊断优化"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=4,
                 cnn_filters=32, d_model=64, n_heads=4, num_layers=2,
                 dropout=0.1):
        super(LightweightCNNTransformer, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # 轻量级CNN特征提取
        self.cnn_backbone = nn.Sequential(
            # 第一层：提取局部特征
            nn.Conv1d(input_channels, cnn_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(cnn_filters),
            nn.GELU(),
            nn.MaxPool1d(4),
            
            # 第二层：进一步抽象
            nn.Conv1d(cnn_filters, cnn_filters*2, kernel_size=9, padding=4),
            nn.BatchNorm1d(cnn_filters*2),
            nn.GELU(),
            nn.MaxPool1d(4),
            
            # 第三层：高层特征
            nn.Conv1d(cnn_filters*2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # 计算CNN后的序列长度
        self.cnn_seq_len = seq_length // 16  # 经过两次4倍下采样
        
        # 轻量级Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*2,  # 较小的前馈网络
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 位置编码
        self.pos_encoding = ImprovedPositionalEncoding(d_model, self.cnn_seq_len*2, dropout)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        batch_size = x.size(0)
        
        # CNN特征提取
        x = x.transpose(1, 2)  # (batch_size, input_channels, seq_length)
        cnn_features = self.cnn_backbone(x)  # (batch_size, d_model, reduced_seq_len)
        
        # 转换为Transformer输入格式
        x = cnn_features.transpose(1, 2)  # (batch_size, reduced_seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer处理
        x = self.transformer(x)  # (batch_size, reduced_seq_len, d_model)
        
        # 分类
        x = x.transpose(1, 2)  # (batch_size, d_model, reduced_seq_len)
        output = self.classifier(x)
        
        # 生成注意力权重（简化版）
        attention_weights = torch.ones(batch_size, self.cnn_seq_len, device=x.device) / self.cnn_seq_len
        
        return output, attention_weights
    
    def get_latent(self, x):
        """获取潜在表示"""
        x = x.transpose(1, 2)
        cnn_features = self.cnn_backbone(x)
        x = cnn_features.transpose(1, 2)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        
        # 全局平均池化
        context = torch.mean(x, dim=1)
        return context


# 工厂函数
def create_fixed_informer(input_channels=3, seq_length=1000, num_classes=4, memory_efficient=True):
    """创建修复版Informer模型"""
    if memory_efficient:
        return FixedDirectInformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            d_model=96,      # 更小的模型维度
            n_heads=6,       # 确保能被d_model整除
            d_ff=192,
            depth=2,         # 较少的层数
            factor=3,        # 较小的稀疏因子
            dropout_rate=0.1
        )
    else:
        return FixedDirectInformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            d_model=128,
            n_heads=8,
            d_ff=256,
            depth=3,
            factor=3,
            dropout_rate=0.1
        )


def create_optimized_transformer(input_channels=3, seq_length=1000, num_classes=4, 
                               use_cnn_preprocessing=True, memory_efficient=True):
    """创建优化的Transformer模型"""
    if memory_efficient:
        return OptimizedTransformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            d_model=96,
            n_heads=6,
            num_layers=3,
            d_ff=192,
            dropout=0.1,
            use_cnn_preprocessing=use_cnn_preprocessing
        )
    else:
        return OptimizedTransformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            d_model=128,
            n_heads=8,
            num_layers=4,
            d_ff=256,
            dropout=0.1,
            use_cnn_preprocessing=use_cnn_preprocessing
        )


def create_lightweight_cnn_transformer(input_channels=3, seq_length=1000, num_classes=4):
    """创建轻量级CNN+Transformer模型"""
    return LightweightCNNTransformer(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        cnn_filters=32,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1
    )
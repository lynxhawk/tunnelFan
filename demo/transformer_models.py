import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    位置编码模块 - 用于Transformer
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - 内存优化版本
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性变换并重塑为多头格式
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重新组织并通过输出层
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights.mean(dim=1)  # 返回平均注意力权重用于可视化


class FeedForward(nn.Module):
    """
    前馈网络 - 内存优化版本
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class VanillaTransformerClassifier(nn.Module):
    """
    原生Transformer分类器 - 8G显存优化版本
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=4,
                 d_model=128, n_heads=8, num_layers=4, d_ff=256, 
                 dropout=0.1, max_seq_len=2000):
        super(VanillaTransformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # 转置以适配位置编码
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, d_model)
        
        # 通过Transformer块
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x)
            attention_weights_list.append(attention_weights)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_length)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(x)
        
        # 返回最后一层的注意力权重用于可视化
        final_attention = attention_weights_list[-1] if attention_weights_list else None
        
        return output, final_attention
    
    def get_latent(self, x):
        """获取分类前的特征表示"""
        batch_size = x.size(0)
        
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)
        
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        return x


class LightweightCNN(nn.Module):
    """
    轻量级CNN特征提取器
    """
    def __init__(self, input_channels=3, filters=32, kernel_size=3, 
                 num_layers=2, dropout=0.1):
        super(LightweightCNN, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = filters * (2 ** i)
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
            
        self.cnn_layers = nn.Sequential(*layers)
        self.output_channels = in_channels
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        x = x.transpose(1, 2)  # (batch_size, input_channels, seq_length)
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # (batch_size, seq_length, output_channels)
        return x


class CNNTransformerClassifier(nn.Module):
    """
    CNN + Transformer混合模型 - 8G显存优化版本
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=4,
                 cnn_filters=32, cnn_kernel_size=3, cnn_layers=2,
                 d_model=128, n_heads=8, num_layers=3, d_ff=256,
                 dropout=0.1, max_seq_len=2000):
        super(CNNTransformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # CNN特征提取器
        self.cnn_extractor = LightweightCNN(
            input_channels, cnn_filters, cnn_kernel_size, 
            cnn_layers, dropout
        )
        
        # 计算CNN输出后的序列长度
        cnn_output_length = seq_length
        for _ in range(cnn_layers):
            cnn_output_length = cnn_output_length // 2
        
        # 特征投影层
        cnn_output_dim = cnn_filters * (2 ** (cnn_layers - 1))
        self.feature_projection = nn.Linear(cnn_output_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        batch_size = x.size(0)
        
        # CNN特征提取
        cnn_features = self.cnn_extractor(x)  # (batch_size, reduced_seq_len, cnn_output_dim)
        
        # 特征投影
        x = self.feature_projection(cnn_features)  # (batch_size, reduced_seq_len, d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # (reduced_seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, reduced_seq_len, d_model)
        
        # 通过Transformer块
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x)
            attention_weights_list.append(attention_weights)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # (batch_size, d_model, reduced_seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(x)
        
        # 返回最后一层的注意力权重用于可视化
        final_attention = attention_weights_list[-1] if attention_weights_list else None
        
        return output, final_attention
    
    def get_latent(self, x):
        """获取分类前的特征表示"""
        batch_size = x.size(0)
        
        cnn_features = self.cnn_extractor(x)
        x = self.feature_projection(cnn_features)
        
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)
        
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        return x


class EfficientTransformerClassifier(nn.Module):
    """
    高效Transformer分类器 - 使用线性注意力机制
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=4,
                 d_model=96, n_heads=6, num_layers=3, d_ff=192,
                 dropout=0.1, use_linear_attention=True):
        super(EfficientTransformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        self.use_linear_attention = use_linear_attention
        
        # 输入投影层
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, seq_length * 2, dropout)
        
        # Transformer编码器层
        if use_linear_attention:
            self.transformer_blocks = nn.ModuleList([
                LinearTransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # 通过Transformer块
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x)
            attention_weights_list.append(attention_weights)
        
        # 全局平均池化
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        # 分类
        output = self.classifier(x)
        
        final_attention = attention_weights_list[-1] if attention_weights_list else None
        
        return output, final_attention
    
    def get_latent(self, x):
        """获取分类前的特征表示"""
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)
        
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        return x


class LinearAttention(nn.Module):
    """
    线性注意力机制 - O(n)复杂度而非O(n^2)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(LinearAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 线性注意力计算 (ReLU激活)
        Q = F.relu(Q)
        K = F.relu(K)
        
        # 计算 K^T * V
        KV = torch.matmul(K.transpose(-2, -1), V)  # (batch, heads, d_k, d_k)
        
        # 计算 Q * (K^T * V)
        context = torch.matmul(Q, KV)  # (batch, heads, seq_len, d_k)
        
        # 归一化
        normalizer = torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1))
        context = context / (normalizer + 1e-6)
        
        # 重新组织
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # 生成虚拟注意力权重用于可视化
        dummy_attention = torch.ones(batch_size, seq_len, device=query.device) / seq_len
        
        return output, dummy_attention


class LinearTransformerBlock(nn.Module):
    """
    使用线性注意力的Transformer块
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(LinearTransformerBlock, self).__init__()
        self.attention = LinearAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 线性注意力 + 残差连接
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class ConvTransformerClassifier(nn.Module):
    """
    卷积+Transformer混合分类器 - 使用深度可分离卷积
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=4,
                 conv_channels=64, d_model=128, n_heads=8, num_layers=3,
                 d_ff=256, dropout=0.1):
        super(ConvTransformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # 深度可分离卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv1d(input_channels, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # 第二层 - 深度可分离卷积
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, 
                     groups=conv_channels, padding=2),  # 深度卷积
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=1),  # 逐点卷积
            nn.BatchNorm1d(conv_channels * 2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # 特征投影
        self.feature_projection = nn.Linear(conv_channels * 2, d_model)
        
        # 位置编码
        reduced_seq_len = seq_length // 4  # 经过两次maxpool
        self.pos_encoding = PositionalEncoding(d_model, reduced_seq_len * 2, dropout)
        
        # Transformer编码器
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        batch_size = x.size(0)
        
        # 转换为卷积格式
        x = x.transpose(1, 2)  # (batch_size, input_channels, seq_length)
        
        # 卷积特征提取
        conv_features = self.conv_layers(x)  # (batch_size, conv_channels*2, reduced_seq_len)
        
        # 转换回序列格式
        conv_features = conv_features.transpose(1, 2)  # (batch_size, reduced_seq_len, conv_channels*2)
        
        # 特征投影
        x = self.feature_projection(conv_features)  # (batch_size, reduced_seq_len, d_model)
        
        # 位置编码
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Transformer编码
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x)
            attention_weights_list.append(attention_weights)
        
        # 分类
        x = x.transpose(1, 2)  # (batch_size, d_model, reduced_seq_len)
        output = self.classifier(x)
        
        final_attention = attention_weights_list[-1] if attention_weights_list else None
        
        return output, final_attention
    
    def get_latent(self, x):
        """获取分类前的特征表示"""
        x = x.transpose(1, 2)
        conv_features = self.conv_layers(x)
        conv_features = conv_features.transpose(1, 2)
        x = self.feature_projection(conv_features)
        
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        return x


# 为了方便使用，创建预设的模型配置
def create_vanilla_transformer(input_channels=3, seq_length=1000, num_classes=4, memory_efficient=True):
    """创建原生Transformer模型"""
    if memory_efficient:
        # 8G显存优化配置
        return VanillaTransformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            d_model=96,      # 较小的模型维度
            n_heads=6,       # 较少的注意力头
            num_layers=3,    # 较少的层数
            d_ff=192,        # 较小的前馈网络
            dropout=0.1
        )
    else:
        # 标准配置
        return VanillaTransformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            d_model=128,
            n_heads=8,
            num_layers=4,
            d_ff=256,
            dropout=0.1
        )


def create_cnn_transformer(input_channels=3, seq_length=1000, num_classes=4, memory_efficient=True):
    """创建CNN+Transformer混合模型"""
    if memory_efficient:
        # 8G显存优化配置
        return CNNTransformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            cnn_filters=24,      # 较少的CNN滤波器
            cnn_kernel_size=3,
            cnn_layers=2,
            d_model=96,          # 较小的Transformer维度
            n_heads=6,
            num_layers=3,
            d_ff=192,
            dropout=0.1
        )
    else:
        # 标准配置
        return CNNTransformerClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            cnn_filters=32,
            cnn_kernel_size=3,
            cnn_layers=2,
            d_model=128,
            n_heads=8,
            num_layers=3,
            d_ff=256,
            dropout=0.1
        )


def create_efficient_transformer(input_channels=3, seq_length=1000, num_classes=4):
    """创建高效线性Transformer模型"""
    return EfficientTransformerClassifier(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        d_model=96,
        n_heads=6,
        num_layers=3,
        d_ff=192,
        dropout=0.1,
        use_linear_attention=True
    )


def create_conv_transformer(input_channels=3, seq_length=1000, num_classes=4):
    """创建卷积+Transformer模型"""
    return ConvTransformerClassifier(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        conv_channels=48,
        d_model=96,
        n_heads=6,
        num_layers=3,
        d_ff=192,
        dropout=0.1
    )
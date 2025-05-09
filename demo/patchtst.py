import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PatchTST(nn.Module):
    """
    PatchTST模型用于轴承故障分类
    基于论文 "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    """
    
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, n_heads=8, 
                 num_layers=3, dropout_rate=0.1):
        """
        初始化PatchTST模型
        
        参数:
        - input_channels: 输入通道数（通常是3，对应X、Y、Z轴）
        - seq_length: 序列长度（时间步数）
        - num_classes: 分类类别数
        - patch_size: 每个patch的长度
        - stride: patch提取的步长
        - d_model: Transformer模型的维度
        - n_heads: 多头注意力中的头数
        - num_layers: Transformer编码器层数
        - dropout_rate: Dropout比率
        """
        super(PatchTST, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        
        # 计算patch数量
        self.num_patches = (seq_length - patch_size) // stride + 1
        
        # 投影层：将每个patch映射到d_model维度
        self.patch_proj = nn.Linear(patch_size * input_channels, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate, self.num_patches)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - outputs: 分类输出，形状为 [batch_size, num_classes]
        - attention_weights: 注意力权重，用于可视化
        """
        batch_size = x.size(0)
        
        # 提取patches
        x_patches = self._extract_patches(x)  # [batch_size, num_patches, patch_size * input_channels]
        
        # 投影到d_model维度
        x_proj = self.patch_proj(x_patches)  # [batch_size, num_patches, d_model]
        
        # 添加位置编码
        x_pos = self.pos_encoding(x_proj)  # [batch_size, num_patches, d_model]
        
        # Transformer编码器
        attn_mask = None  # 可以添加自定义的注意力掩码
        outputs = self.transformer_encoder(x_pos, mask=attn_mask)  # [batch_size, num_patches, d_model]
        
        # 计算每个token的平均注意力权重，用于可视化
        # 注：此处生成假的注意力权重，因为Transformer内部的注意力权重难以直接获取
        # 实际应用中，您可能希望通过修改TransformerEncoder来保存真实的注意力权重
        attention_weights = torch.mean(outputs, dim=2)  # [batch_size, num_patches]
        
        # 进一步处理注意力以适配原始序列长度（用于可视化）
        attention_expanded = self._expand_attention(attention_weights)  # [batch_size, seq_length]
        
        # 全局平均池化
        x_pooled = torch.mean(outputs, dim=1)  # [batch_size, d_model]
        
        # 分类
        logits = self.classifier(x_pooled)  # [batch_size, num_classes]
        
        return logits, attention_expanded
    
    def _extract_patches(self, x):
        """
        从输入序列中提取patches
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - patches: 提取的patches，形状为 [batch_size, num_patches, patch_size * input_channels]
        """
        batch_size = x.size(0)
        patches = []
        
        for i in range(0, self.seq_length - self.patch_size + 1, self.stride):
            # 提取patch并展平通道维度
            patch = x[:, i:i+self.patch_size, :]  # [batch_size, patch_size, input_channels]
            patch_flat = patch.reshape(batch_size, -1)  # [batch_size, patch_size * input_channels]
            patches.append(patch_flat)
        
        # 堆叠所有patches
        return torch.stack(patches, dim=1)  # [batch_size, num_patches, patch_size * input_channels]
    
    def _expand_attention(self, attention_weights):
        """
        将注意力权重扩展到与原始序列相同的长度，用于可视化
        
        参数:
        - attention_weights: 注意力权重，形状为 [batch_size, num_patches]
        
        返回:
        - expanded: 扩展后的注意力权重，形状为 [batch_size, seq_length]
        """
        batch_size = attention_weights.size(0)
        expanded = torch.zeros(batch_size, self.seq_length, device=attention_weights.device)
        
        # 将每个patch的注意力分配给对应的时间步
        for i in range(self.num_patches):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.patch_size, self.seq_length)
            expanded[:, start_idx:end_idx] += attention_weights[:, i].unsqueeze(1)
        
        # 归一化
        expanded = F.normalize(expanded, p=1, dim=1)
        
        return expanded
    
    def get_latent(self, x):
        """
        获取潜在特征表示，用于t-SNE可视化等
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - latent: 潜在特征表示，形状为 [batch_size, d_model]
        """
        batch_size = x.size(0)
        
        # 提取patches
        x_patches = self._extract_patches(x)
        
        # 投影到d_model维度
        x_proj = self.patch_proj(x_patches)
        
        # 添加位置编码
        x_pos = self.pos_encoding(x_proj)
        
        # Transformer编码器
        outputs = self.transformer_encoder(x_pos)
        
        # 全局平均池化
        latent = torch.mean(outputs, dim=1)
        
        return latent


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer提供序列位置信息
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入张量
        
        参数:
        - x: 输入张量，形状为 [batch_size, seq_length, d_model]
        
        返回:
        - x: 添加位置编码后的张量，形状不变
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class PatchTSTClassifier(nn.Module):
    """
    封装PatchTST模型，提供与其他分类器兼容的接口
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, n_heads=8, 
                 num_layers=3, dropout_rate=0.1):
        super(PatchTSTClassifier, self).__init__()
        
        self.model = PatchTST(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_latent(self, x):
        return self.model.get_latent(x)
"""
Transformer和CNN-Transformer模型实现 - 修复版本
专为8GB显存优化，适用于轴承故障诊断
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码"""
    
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
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """原生Transformer分类器 - 8GB显存优化版"""
    
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10, 
                 d_model=128, nhead=8, num_layers=4, dim_feedforward=256, 
                 dropout_rate=0.1, max_len=5000):
        """
        Args:
            input_channels: 输入通道数
            seq_length: 序列长度
            num_classes: 分类数量
            d_model: 模型维度 (降低到128以节省显存)
            nhead: 注意力头数
            num_layers: Transformer层数 (降低到4层)
            dim_feedforward: 前馈网络维度 (降低以节省显存)
            dropout_rate: Dropout比率
            max_len: 最大序列长度
        """
        super(TransformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout_rate)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=False  # [seq_len, batch_size, d_model]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_length, input_channels] 或 [batch_size, input_channels, seq_length]
        Returns:
            logits: 分类输出
            features: 特征表示
        """
        batch_size = x.size(0)
        
        # 调整输入维度为 [batch_size, seq_length, input_channels]
        if x.dim() == 3 and x.size(1) == self.input_channels:
            x = x.transpose(1, 2)  # [batch_size, seq_length, input_channels]
        elif x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch_size, seq_length, 1]
        
        # 输入投影: [batch_size, seq_length, input_channels] -> [batch_size, seq_length, d_model]
        x = self.input_projection(x)
        
        # 转换为Transformer期望的格式: [seq_length, batch_size, d_model]
        x = x.transpose(0, 1)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(x)  # [seq_length, batch_size, d_model]
        
        # 转换回 [batch_size, seq_length, d_model]
        transformer_output = transformer_output.transpose(0, 1)
        
        # 全局平均池化: [batch_size, seq_length, d_model] -> [batch_size, d_model]
        features = transformer_output.mean(dim=1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features
    
    def get_latent(self, x):
        """获取潜在特征表示"""
        _, features = self.forward(x)
        return features


class CNNTransformerClassifier(nn.Module):
    """CNN-Transformer混合分类器 - 修复版本"""
    
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10,
                 cnn_filters=32, cnn_kernel_size=7, cnn_layers=2,
                 d_model=128, nhead=8, num_transformer_layers=3, 
                 dim_feedforward=256, dropout_rate=0.1, pool_size=4):
        """
        Args:
            input_channels: 输入通道数
            seq_length: 序列长度
            num_classes: 分类数量
            cnn_filters: CNN滤波器数量 (降低以节省显存)
            cnn_kernel_size: CNN核大小
            cnn_layers: CNN层数
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_transformer_layers: Transformer层数 (降低以节省显存)
            dim_feedforward: 前馈网络维度
            dropout_rate: Dropout比率
            pool_size: 池化大小，用于降低序列长度
        """
        super(CNNTransformerClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        self.pool_size = pool_size
        
        # 计算池化后的序列长度
        self.reduced_seq_length = seq_length // pool_size
        
        # CNN特征提取器
        cnn_layers_list = []
        current_channels = input_channels
        
        for i in range(cnn_layers):
            out_channels = cnn_filters * (2 ** i)  # 每层倍增滤波器数量
            out_channels = min(out_channels, 128)  # 限制最大滤波器数量
            
            cnn_layers_list.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=cnn_kernel_size, 
                         padding=cnn_kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_channels = out_channels
        
        # 添加池化层降低序列长度
        cnn_layers_list.append(nn.AdaptiveAvgPool1d(self.reduced_seq_length))
        
        self.cnn_feature_extractor = nn.Sequential(*cnn_layers_list)
        
        # 记录CNN输出的通道数
        self.cnn_output_channels = current_channels
        
        print(f"CNN输出通道数: {self.cnn_output_channels}, 池化后序列长度: {self.reduced_seq_length}")
        
        # CNN输出到Transformer输入的投影
        self.cnn_to_transformer = nn.Linear(self.cnn_output_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, self.reduced_seq_length + 100, dropout_rate)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_length, input_channels] 或 [batch_size, input_channels, seq_length]
        Returns:
            logits: 分类输出
            features: 特征表示
        """
        batch_size = x.size(0)
        
        # 调整输入维度为CNN期望的格式: [batch_size, input_channels, seq_length]
        if x.dim() == 3 and x.size(2) == self.input_channels:
            x = x.transpose(1, 2)  # [batch_size, input_channels, seq_length]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, seq_length]
        
        # Debug: 打印输入形状
        # print(f"CNN输入形状: {x.shape}")
        
        # CNN特征提取: [batch_size, input_channels, seq_length] -> [batch_size, cnn_output_channels, reduced_seq_length]
        cnn_features = self.cnn_feature_extractor(x)
        
        # Debug: 打印CNN输出形状
        # print(f"CNN输出形状: {cnn_features.shape}")
        
        # 转换为Transformer格式: [batch_size, reduced_seq_length, cnn_output_channels]
        cnn_features = cnn_features.transpose(1, 2)
        
        # Debug: 打印转置后形状
        # print(f"转置后形状: {cnn_features.shape}")
        
        # 投影到Transformer维度: [batch_size, reduced_seq_length, d_model]
        transformer_input = self.cnn_to_transformer(cnn_features)
        
        # Debug: 打印投影后形状
        # print(f"投影后形状: {transformer_input.shape}")
        
        # 转换为Transformer期望的格式: [reduced_seq_length, batch_size, d_model]
        transformer_input = transformer_input.transpose(0, 1)
        
        # 位置编码
        transformer_input = self.pos_encoder(transformer_input)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(transformer_input)
        
        # 转换回 [batch_size, reduced_seq_length, d_model]
        transformer_output = transformer_output.transpose(0, 1)
        
        # 全局平均池化: [batch_size, reduced_seq_length, d_model] -> [batch_size, d_model]
        features = transformer_output.mean(dim=1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features
    
    def get_latent(self, x):
        """获取潜在特征表示"""
        _, features = self.forward(x)
        return features


def create_transformer_model(input_channels=1, seq_length=1000, num_classes=10,
                           d_model=128, nhead=8, num_layers=4, 
                           dim_feedforward=256, dropout_rate=0.1):
    """创建Transformer模型的工厂函数"""
    return TransformerClassifier(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate
    )


def create_cnn_transformer_model(input_channels=1, seq_length=1000, num_classes=10,
                               cnn_filters=32, cnn_kernel_size=7, cnn_layers=2,
                               d_model=128, nhead=8, num_transformer_layers=3,
                               dim_feedforward=256, dropout_rate=0.1, pool_size=4):
    """创建CNN-Transformer模型的工厂函数"""
    return CNNTransformerClassifier(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        cnn_filters=cnn_filters,
        cnn_kernel_size=cnn_kernel_size,
        cnn_layers=cnn_layers,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        pool_size=pool_size
    )


# 测试代码
if __name__ == "__main__":
    # 测试Transformer模型
    print("测试Transformer模型...")
    transformer_model = create_transformer_model(
        input_channels=1,
        seq_length=1000,
        num_classes=10,
        d_model=128,
        nhead=8,
        num_layers=4
    )
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"Transformer模型参数量: {total_params:,}")
    
    # 测试输入
    test_input = torch.randn(16, 1000, 1)  # [batch_size, seq_length, channels]
    transformer_output, transformer_features = transformer_model(test_input)
    print(f"Transformer输出形状: {transformer_output.shape}, 特征形状: {transformer_features.shape}")
    
    # 测试CNN-Transformer模型
    print("\n测试CNN-Transformer模型...")
    cnn_transformer_model = create_cnn_transformer_model(
        input_channels=1,
        seq_length=1000,
        num_classes=10,
        cnn_filters=32,
        d_model=128,
        nhead=8,
        num_transformer_layers=3
    )
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in cnn_transformer_model.parameters())
    print(f"CNN-Transformer模型参数量: {total_params:,}")
    
    cnn_transformer_output, cnn_transformer_features = cnn_transformer_model(test_input)
    print(f"CNN-Transformer输出形状: {cnn_transformer_output.shape}, 特征形状: {cnn_transformer_features.shape}")
    
    # 显存使用估算
    print(f"\n显存使用估算 (batch_size=16):")
    print(f"CNN-Transformer: ~{total_params * 4 / 1024**2:.1f}MB 参数 + ~200MB 激活")
    print(f"总计: ~{total_params * 4 / 1024**2 + 200:.1f}MB")
    print("适合8GB显存训练！")
    
    print("\n所有模型测试完成！")
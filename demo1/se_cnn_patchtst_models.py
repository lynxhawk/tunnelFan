"""
SE-CNN-PatchTST模型实现 - 所有变体都带注意力机制
专为8GB显存优化，适用于轴承故障诊断
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    用于通道注意力的模块，增强特征表示
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch_size, channels, seq_len)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CNNFeatureExtractor(nn.Module):
    """
    标准CNN特征提取器 - 支持SE Block
    """
    def __init__(self, input_channels, base_filters=64, kernel_size=3, use_se=True, se_reduction=16):
        super(CNNFeatureExtractor, self).__init__()
        self.use_se = use_se
        
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(2)
        
        if use_se:
            self.se1 = SEBlock(base_filters, reduction=se_reduction)
        
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(base_filters*2)
        self.pool2 = nn.MaxPool1d(2)
        
        if use_se:
            self.se2 = SEBlock(base_filters*2, reduction=se_reduction)
        
        self.output_channels = base_filters*2
        
    def forward(self, x):
        # x: (batch_size, input_channels, seq_length)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_se:
            x = self.se1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.use_se:
            x = self.se2(x)
        x = self.pool2(x)
        
        return x


class OneLayerCNNFeatureExtractor(nn.Module):
    """
    一层卷积的CNN特征提取器
    """
    def __init__(self, input_channels, base_filters=64, kernel_size=3, use_se=True, se_reduction=16):
        super(OneLayerCNNFeatureExtractor, self).__init__()
        self.use_se = use_se
        
        self.conv1 = nn.Conv1d(input_channels, base_filters*2, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(base_filters*2)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(4)  # 使用更大的池化窗口，一次性缩小为原来的1/4
        
        if use_se:
            self.se1 = SEBlock(base_filters*2, reduction=se_reduction)
        
        self.output_channels = base_filters*2
        
    def forward(self, x):
        # x: (batch_size, input_channels, seq_length)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_se:
            x = self.se1(x)
        x = self.pool1(x)
        
        return x


class ThreeLayerCNNFeatureExtractor(nn.Module):
    """
    三层卷积的CNN特征提取器
    """
    def __init__(self, input_channels, base_filters=64, kernel_size=3, use_se=True, se_reduction=16):
        super(ThreeLayerCNNFeatureExtractor, self).__init__()
        self.use_se = use_se
        
        # 第一卷积层
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu1 = nn.ReLU(inplace=True)
        
        if use_se:
            self.se1 = SEBlock(base_filters, reduction=se_reduction)
            
        self.pool1 = nn.MaxPool1d(2)
        
        # 第二卷积层
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(base_filters*2)
        self.relu2 = nn.ReLU(inplace=True)
        
        if use_se:
            self.se2 = SEBlock(base_filters*2, reduction=se_reduction)
            
        # 第三卷积层
        self.conv3 = nn.Conv1d(base_filters*2, base_filters*2, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(base_filters*2)
        self.relu3 = nn.ReLU(inplace=True)
        
        if use_se:
            self.se3 = SEBlock(base_filters*2, reduction=se_reduction)
            
        self.pool2 = nn.MaxPool1d(2)
        
        self.output_channels = base_filters*2
        
    def forward(self, x):
        # x: (batch_size, input_channels, seq_length)
        
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.use_se:
            x = self.se1(x)
        x = self.pool1(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.use_se:
            x = self.se2(x)
        
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        if self.use_se:
            x = self.se3(x)
            
        x = self.pool2(x)
        
        return x


class PatchTSTAttention(nn.Module):
    """
    PatchTST模块 - 带注意力机制的版本
    """
    def __init__(self, d_model, patch_num, n_heads=8, num_layers=3, dropout_rate=0.1):
        super(PatchTSTAttention, self).__init__()
        
        # 位置编码
        self.patch_num = patch_num
        self.positional_encoding = nn.Parameter(torch.zeros(1, patch_num, d_model))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: [batch_size, patch_num, d_model]
        batch_size, actual_patch_num, d_model = x.size()
        
        # 处理位置编码维度不匹配的情况
        if actual_patch_num != self.patch_num:
            if actual_patch_num < self.patch_num:
                pos_enc = self.positional_encoding[:, :actual_patch_num, :]
            else:
                repeats = (actual_patch_num + self.patch_num - 1) // self.patch_num
                pos_enc = torch.cat([self.positional_encoding] * repeats, dim=1)[:, :actual_patch_num, :]
        else:
            pos_enc = self.positional_encoding
            
        # 添加位置编码
        x = x + pos_enc
        x = self.dropout(x)
        
        # 应用Transformer编码器层
        attns = []
        for encoder in self.encoder_layers:
            x = encoder(x)
            # 创建均匀注意力权重
            attn = torch.ones(batch_size, actual_patch_num, device=x.device) / actual_patch_num
            attns.append(attn)
        
        # 返回最后一层的attn作为注意力权重
        return x, attns[-1]


class SECNNPatchTSTBase(nn.Module):
    """
    SE-CNN-PatchTST 基础模型类
    """
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3, 
                 cnn_type='standard', base_filters=64, kernel_size=3, 
                 dropout_rate=0.1, use_se=True, se_reduction=16):
        super(SECNNPatchTSTBase, self).__init__()
        
        # 保存基本参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # 选择CNN特征提取器类型
        if cnn_type == 'standard':
            self.feature_extractor = CNNFeatureExtractor(
                input_channels=input_channels,
                base_filters=base_filters,
                kernel_size=kernel_size,
                use_se=use_se,
                se_reduction=se_reduction
            )
        elif cnn_type == 'one_layer':
            self.feature_extractor = OneLayerCNNFeatureExtractor(
                input_channels=input_channels,
                base_filters=base_filters,
                kernel_size=kernel_size,
                use_se=use_se,
                se_reduction=se_reduction
            )
        elif cnn_type == 'three_layer':
            self.feature_extractor = ThreeLayerCNNFeatureExtractor(
                input_channels=input_channels,
                base_filters=base_filters,
                kernel_size=kernel_size,
                use_se=use_se,
                se_reduction=se_reduction
            )
        else:
            raise ValueError(f"不支持的CNN类型: {cnn_type}")
        
        # 计算经过CNN后的序列长度
        self.seq_len_after_cnn = seq_length // 4
        
        # 计算patch数量
        self.patch_num = max(1, (self.seq_len_after_cnn - patch_size) // stride + 1)
        
        # Patch嵌入
        self.patch_embedding = nn.Conv1d(
            in_channels=self.feature_extractor.output_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride
        )
        
        # PatchTST模块（带注意力）
        self.patchtst = PatchTSTAttention(
            d_model=d_model,
            patch_num=self.patch_num,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_length, input_channels] 或 [batch_size, input_channels, seq_length]
        batch_size = x.size(0)
        
        # 调整输入维度为CNN期望的格式
        if x.dim() == 3 and x.size(2) == self.input_channels:
            x = x.transpose(1, 2)  # [batch_size, input_channels, seq_length]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, seq_length]
        
        # CNN特征提取
        x = self.feature_extractor(x)  # [batch_size, channels, seq_length/4]
        
        # 提取patches
        x = self.patch_embedding(x)  # [batch_size, d_model, num_patches]
        
        # 调整维度用于PatchTST
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, d_model]
        
        # 应用PatchTST模块
        x, attention_weights = self.patchtst(x)
        
        # 全局平均池化
        features = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # 分类
        outputs = self.classifier(features)
        
        return outputs, features
    
    def get_latent(self, x):
        """获取潜在特征表示"""
        _, features = self.forward(x)
        return features


class SECNNPatchTSTStandard(SECNNPatchTSTBase):
    """标准SE-CNN-PatchTST模型 - 8GB显存优化"""
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3,
                 base_filters=32, kernel_size=3, dropout_rate=0.1, 
                 use_se=True, se_reduction=8):
        super(SECNNPatchTSTStandard, self).__init__(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            cnn_type='standard',
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )


class SECNNPatchTSTOneLayer(SECNNPatchTSTBase):
    """一层CNN的SE-CNN-PatchTST模型 - 8GB显存优化"""
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3,
                 base_filters=32, kernel_size=3, dropout_rate=0.1, 
                 use_se=True, se_reduction=8):
        super(SECNNPatchTSTOneLayer, self).__init__(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            cnn_type='one_layer',
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )


class SECNNPatchTSTThreeLayer(SECNNPatchTSTBase):
    """三层CNN的SE-CNN-PatchTST模型 - 8GB显存优化"""
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3,
                 base_filters=32, kernel_size=3, dropout_rate=0.1, 
                 use_se=True, se_reduction=8):
        super(SECNNPatchTSTThreeLayer, self).__init__(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            cnn_type='three_layer',
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )


class StandardCNNPatchTST(nn.Module):
    """
    标准CNN-PatchTST模型（无SE Block，带注意力机制）
    """
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3,
                 base_filters=32, kernel_size=3, dropout_rate=0.1):
        super(StandardCNNPatchTST, self).__init__()
        
        # 保存基本参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # 标准CNN特征提取器（无SE Block）
        self.feature_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
            use_se=False,  # 关键：不使用SE Block
            se_reduction=16
        )
        
        # 计算经过CNN后的序列长度
        self.seq_len_after_cnn = seq_length // 4
        
        # 计算patch数量
        self.patch_num = max(1, (self.seq_len_after_cnn - patch_size) // stride + 1)
        
        # Patch嵌入
        self.patch_embedding = nn.Conv1d(
            in_channels=self.feature_extractor.output_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride
        )
        
        # PatchTST模块（带注意力）
        self.patchtst = PatchTSTAttention(
            d_model=d_model,
            patch_num=self.patch_num,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_length, input_channels] 或 [batch_size, input_channels, seq_length]
        batch_size = x.size(0)
        
        # 调整输入维度为CNN期望的格式
        if x.dim() == 3 and x.size(2) == self.input_channels:
            x = x.transpose(1, 2)  # [batch_size, input_channels, seq_length]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, seq_length]
        
        # 标准CNN特征提取（无SE Block）
        x = self.feature_extractor(x)  # [batch_size, channels, seq_length/4]
        
        # 提取patches
        x = self.patch_embedding(x)  # [batch_size, d_model, num_patches]
        
        # 调整维度用于PatchTST
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, d_model]
        
        # 应用PatchTST模块
        x, attention_weights = self.patchtst(x)
        
        # 全局平均池化
        features = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # 分类
        outputs = self.classifier(features)
        
        return outputs, features
    
    def get_latent(self, x):
        """获取潜在特征表示"""
        _, features = self.forward(x)
        return features


# 工厂函数
def create_se_cnn_patchtst_standard(input_channels=1, seq_length=1000, num_classes=10, **kwargs):
    """创建标准SE-CNN-PatchTST模型"""
    return SECNNPatchTSTStandard(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        **kwargs
    )


def create_se_cnn_patchtst_one_layer(input_channels=1, seq_length=1000, num_classes=10, **kwargs):
    """创建一层CNN的SE-CNN-PatchTST模型"""
    return SECNNPatchTSTOneLayer(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        **kwargs
    )


def create_se_cnn_patchtst_three_layer(input_channels=1, seq_length=1000, num_classes=10, **kwargs):
    """创建三层CNN的SE-CNN-PatchTST模型"""
    return SECNNPatchTSTThreeLayer(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        **kwargs
    )


def create_standard_cnn_patchtst(input_channels=1, seq_length=1000, num_classes=10, **kwargs):
    """创建标准CNN-PatchTST模型（无SE）"""
    return StandardCNNPatchTST(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        **kwargs
    )


# 测试代码
if __name__ == "__main__":
    # 测试所有模型变体
    models = {
        'SE-CNN-PatchTST-Standard': create_se_cnn_patchtst_standard,
        'SE-CNN-PatchTST-OneLayer': create_se_cnn_patchtst_one_layer,
        'SE-CNN-PatchTST-ThreeLayer': create_se_cnn_patchtst_three_layer,
        'Standard-CNN-PatchTST': create_standard_cnn_patchtst
    }
    
    # 测试输入
    test_input = torch.randn(16, 1000, 1)  # [batch_size, seq_length, channels]
    
    for model_name, create_func in models.items():
        print(f"\n测试 {model_name}...")
        
        model = create_func(
            input_channels=1,
            seq_length=1000,
            num_classes=10,
            d_model=128,
            n_heads=8,
            num_layers=3,
            base_filters=32
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        # 前向传播测试
        try:
            output, features = model(test_input)
            print(f"输出形状: {output.shape}, 特征形状: {features.shape}")
            print(f"显存估算: ~{total_params * 4 / 1024**2:.1f}MB")
        except Exception as e:
            print(f"测试失败: {e}")
    
    print(f"\n所有模型都适合8GB显存训练！")
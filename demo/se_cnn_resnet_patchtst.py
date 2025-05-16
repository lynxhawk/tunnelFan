import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    用于通道注意力的模块，可用于增强特征表示
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel = channel
        self.reduction = reduction
        self.fc = None
        self._build_fc(channel)
        
    def _build_fc(self, channel):
        """构建全连接层，适应输入通道数"""
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // self.reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // self.reduction), channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch_size, channels, seq_len)
        b, c, _ = x.size()
        
        # 检查通道数是否与预期不符
        if c != self.channel:
            print(f"通道数不匹配：预期 {self.channel}，实际 {c}，重建 fc 层")
            self.channel = c
            self._build_fc(c)
            self.fc = self.fc.to(x.device)
            
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    """
    ResNet基本块
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=False, se_reduction=16):
        super(BasicBlock, self).__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE Block
        if self.use_se:
            self.se = SEBlock(out_channels, reduction=se_reduction)
        
        # 如果输入和输出维度不匹配，使用1x1卷积进行调整
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CNNFeatureExtractor(nn.Module):
    """
    简单的CNN特征提取器
    """
    def __init__(self, input_channels, base_filters=64, kernel_size=3, use_se=False, se_reduction=16):
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
        
        self.conv3 = nn.Conv1d(base_filters*2, base_filters*4, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(base_filters*4)
        
        if use_se:
            self.se3 = SEBlock(base_filters*4, reduction=se_reduction)
        
        self.output_channels = base_filters*4
    
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
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.use_se:
            x = self.se3(x)
        
        return x


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet特征提取器
    """
    def __init__(self, input_channels, base_filters=64, kernel_size=3, use_se=False, se_reduction=16):
        super(ResNetFeatureExtractor, self).__init__()
        self.use_se = use_se
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=kernel_size, 
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)
        
        # ResNet层
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1, kernel_size=kernel_size, 
                                      use_se=use_se, se_reduction=se_reduction)
        self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2, kernel_size=kernel_size, 
                                      use_se=use_se, se_reduction=se_reduction)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=2, kernel_size=kernel_size, 
                                      use_se=use_se, se_reduction=se_reduction)
        
        self.output_channels = base_filters*4
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, kernel_size, use_se, se_reduction):
        layers = []
        
        # 第一个block可能需要downsample
        layers.append(BasicBlock(in_channels, out_channels, kernel_size, stride, use_se, se_reduction))
        
        # 后续block
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, kernel_size, 1, use_se, se_reduction))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch_size, input_channels, seq_length)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x


class PatchTST(nn.Module):
    """
    标准PatchTST模块（带注意力机制）
    """
    def __init__(self, d_model, patch_num, n_heads=8, num_layers=3, dropout_rate=0.1):
        super(PatchTST, self).__init__()
        
        # 位置编码
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
        
        # 添加位置编码
        x = x + self.positional_encoding
        x = self.dropout(x)
        
        # 应用Transformer编码器层
        attns = []
        for encoder in self.encoder_layers:
            x = encoder(x)
            # 创建虚拟注意力权重
            attn = torch.ones(x.size(0), x.size(1), device=x.device) / x.size(1)
            attns.append(attn)
        
        # 返回最后一层的attn作为注意力权重
        return x, attns[-1]


class PatchTSTNoAttention(nn.Module):
    """
    PatchTST模块（无注意力版本）
    """
    def __init__(self, d_model, patch_num, num_layers=3, dropout_rate=0.1, pooling_type='mean'):
        super(PatchTSTNoAttention, self).__init__()
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, patch_num, d_model))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        
        # 特征提取层（替代Transformer的编码器层）
        self.feature_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout_rate)
            )
            self.feature_layers.append(layer)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling_type = pooling_type
    
    def forward(self, x):
        # x: [batch_size, patch_num, d_model]
        
        # 添加位置编码
        x = x + self.positional_encoding
        x = self.dropout(x)
        
        # 应用特征提取层
        for layer in self.feature_layers:
            x_res = layer(x)
            x = x + x_res  # 残差连接
        
        # 创建虚拟注意力权重
        attn = torch.ones(x.size(0), x.size(1), device=x.device) / x.size(1)
        
        return x, attn
    
    def pool(self, x):
        """根据池化类型应用不同的池化策略"""
        if self.pooling_type == 'mean':
            return torch.mean(x, dim=1)  # [batch_size, d_model]
        elif self.pooling_type == 'max':
            return torch.max(x, dim=1)[0]  # [batch_size, d_model]
        elif self.pooling_type == 'last':
            return x[:, -1, :]  # [batch_size, d_model]
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")


class SECNNPatchTST_Base(nn.Module):
    """
    SE + CNN/ResNet + PatchTST 模型的基类
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3, 
                 use_attention=True, pooling_type='mean', cnn_type='cnn',
                 base_filters=64, kernel_size=3, dropout_rate=0.3, use_se=True, se_reduction=16):
        super(SECNNPatchTST_Base, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.pooling_type = pooling_type
        
        # 选择特征提取器类型
        if cnn_type.lower() == 'cnn':
            self.feature_extractor = CNNFeatureExtractor(
                input_channels=input_channels,
                base_filters=base_filters,
                kernel_size=kernel_size,
                use_se=use_se,
                se_reduction=se_reduction
            )
        elif cnn_type.lower() == 'resnet':
            self.feature_extractor = ResNetFeatureExtractor(
                input_channels=input_channels,
                base_filters=base_filters,
                kernel_size=kernel_size,
                use_se=use_se,
                se_reduction=se_reduction
            )
        else:
            raise ValueError(f"不支持的特征提取器类型: {cnn_type}")
        
        # 计算经过CNN/ResNet后的序列长度
        self.seq_len_after_cnn = seq_length // 4  # CNN经过两次池化，长度变为原来的1/4
        
        # 计算patch数量
        self.num_patches = (self.seq_len_after_cnn - patch_size) // stride + 1
        
        # Patch嵌入
        self.patch_embedding = nn.Conv1d(
            in_channels=self.feature_extractor.output_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride
        )
        
        # PatchTST模块（带注意力或无注意力）
        if use_attention:
            self.patchtst = PatchTST(
                d_model=d_model,
                patch_num=self.num_patches,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
        else:
            self.patchtst = PatchTSTNoAttention(
                d_model=d_model,
                patch_num=self.num_patches,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                pooling_type=pooling_type
            )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: [batch_size, seq_length, input_channels]
        
        # 调整维度用于CNN
        x = x.permute(0, 2, 1)  # [batch_size, input_channels, seq_length]
        
        # 应用CNN/ResNet特征提取
        x = self.feature_extractor(x)  # [batch_size, channels, seq_length/4]
        
        # 提取patches
        x = self.patch_embedding(x)  # [batch_size, d_model, num_patches]
        
        # 调整维度用于PatchTST
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, d_model]
        
        # 应用PatchTST模块
        x, attention_weights = self.patchtst(x)
        
        # 池化（如果是无注意力版本）
        if not self.use_attention:
            x = self.patchtst.pool(x)  # [batch_size, d_model]
        else:
            # 如果是有注意力版本，使用平均池化
            x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # 分类
        outputs = self.classifier(x)
        
        # 将注意力权重调整为与原始序列相同长度（用于可视化）
        # 这里简单地将注意力复制到全序列
        expanded_attention = torch.zeros(x.size(0), self.seq_length, device=x.device)
        patch_length = self.seq_length // self.num_patches
        for i in range(self.num_patches):
            start_idx = i * patch_length
            end_idx = min(start_idx + patch_length, self.seq_length)
            expanded_attention[:, start_idx:end_idx] = attention_weights[:, i].unsqueeze(1)
        
        # 归一化
        expanded_attention = F.normalize(expanded_attention, p=1, dim=1)
        
        return outputs, expanded_attention
    
    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 同forward逻辑，但只返回分类器前的特征
        
        # 调整维度用于CNN
        x = x.permute(0, 2, 1)  # [batch_size, input_channels, seq_length]
        
        # 应用CNN/ResNet特征提取
        x = self.feature_extractor(x)  # [batch_size, channels, seq_length/4]
        
        # 提取patches
        x = self.patch_embedding(x)  # [batch_size, d_model, num_patches]
        
        # 调整维度用于PatchTST
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, d_model]
        
        # 应用PatchTST模块
        x, _ = self.patchtst(x)
        
        # 池化
        if not self.use_attention:
            x = self.patchtst.pool(x)  # [batch_size, d_model]
        else:
            x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        return x


# 四种具体实现类
class SECNNPatchTSTNoAttention(SECNNPatchTST_Base):
    """SE + CNN + PatchTST (无注意力)"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, num_layers=3, 
                 pooling_type='mean', base_filters=64, kernel_size=3, 
                 dropout_rate=0.3, use_se=True, se_reduction=16):
        super(SECNNPatchTSTNoAttention, self).__init__(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=None,  # 无需注意力头
            num_layers=num_layers,
            use_attention=False,
            pooling_type=pooling_type,
            cnn_type='cnn',
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )


class SECNNPatchTSTAttention(SECNNPatchTST_Base):
    """SE + CNN + PatchTST (有注意力)"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3,
                 base_filters=64, kernel_size=3, dropout_rate=0.3, 
                 use_se=True, se_reduction=16):
        super(SECNNPatchTSTAttention, self).__init__(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            use_attention=True,
            pooling_type=None,  # 不需要池化类型
            cnn_type='cnn',
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )


class SEResNetPatchTSTNoAttention(SECNNPatchTST_Base):
    """SE + ResNet + PatchTST (无注意力)"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, num_layers=3, 
                 pooling_type='mean', base_filters=64, kernel_size=3, 
                 dropout_rate=0.3, use_se=True, se_reduction=16):
        super(SEResNetPatchTSTNoAttention, self).__init__(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=None,  # 无需注意力头
            num_layers=num_layers,
            use_attention=False,
            pooling_type=pooling_type,
            cnn_type='resnet',
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )


class SEResNetPatchTSTAttention(SECNNPatchTST_Base):
    """SE + ResNet + PatchTST (有注意力)"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3,
                 base_filters=64, kernel_size=3, dropout_rate=0.3, 
                 use_se=True, se_reduction=16):
        super(SEResNetPatchTSTAttention, self).__init__(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            use_attention=True,
            pooling_type=None,  # 不需要池化类型
            cnn_type='resnet',
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )
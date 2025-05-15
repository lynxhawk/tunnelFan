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
        # x: (batch_size, seq_len, channels) 或 (batch_size, channels, seq_len)
        # 根据输入形状调整操作
        if x.size(1) > x.size(2):  # (batch_size, seq_len, channels)
            x_permuted = x.permute(0, 2, 1)  # -> (batch_size, channels, seq_len)
            b, c, _ = x_permuted.size()
            
            # 检查通道数是否与预期不符
            if c != self.channel:
                print(f"通道数不匹配：预期 {self.channel}，实际 {c}，重建 fc 层")
                self.channel = c
                self._build_fc(c)
                self.fc = self.fc.to(x.device)
                
            y = self.avg_pool(x_permuted).view(b, c)
            y = self.fc(y).view(b, c, 1)
            x_scaled = x_permuted * y.expand_as(x_permuted)
            return x_scaled.permute(0, 2, 1)  # 恢复原始形状
        else:  # (batch_size, channels, seq_len)
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


class SEInformerEncoder(nn.Module):
    """
    添加SE Block的Informer编码器
    """
    def __init__(self, d_model, n_heads, d_ff, depth, dropout=0.1, factor=5, use_se=True, se_reduction=16):
        super(SEInformerEncoder, self).__init__()

        # 从原始代码导入所需的类
        from informer_models import PositionalEncoding, EncoderLayer, AttentionLayer, ProbAttention

        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.use_se = use_se

        # 各层定义
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # SE Block (如果启用)
        if self.use_se:
            self.se_blocks = nn.ModuleList([
                SEBlock(d_model, reduction=se_reduction) for _ in range(depth)
            ])

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(
                        False, factor, attention_dropout=dropout, output_attention=True),
                    d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout
            ) for _ in range(depth)
        ])

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]

        # 应用位置编码
        x = self.position_encoding(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # 应用编码器层和SE层
        attns = []
        for i, layer in enumerate(self.encoder_layers):
            x, attn = layer(x)
            if self.use_se:
                # 在每个编码器层后应用SE块
                x = self.se_blocks[i](x)
            attns.append(attn)

        return x, attns


class SEInformerClassifier(nn.Module):
    """
    带有SE Block的Informer分类器模型
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 d_model=256, n_heads=8, d_ff=512, depth=2,
                 factor=5, dropout_rate=0.3, use_se=True, se_reduction=16):
        super(SEInformerClassifier, self).__init__()
        
        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.d_model = d_model
        
        # 将原始信号映射到模型维度
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # 使用带SE Block的Informer编码器
        self.informer_encoder = SEInformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=depth,
            dropout=dropout_rate,
            factor=factor,
            use_se=use_se,
            se_reduction=se_reduction
        )
        
        # 全局注意力池化提取固定大小表示
        self.global_attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类头
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        
        # 投影到模型维度
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Informer编码器 (带SE Block)
        x, attns = self.informer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # 全局注意力池化
        attention_weights = self.global_attention(x)  # (batch_size, seq_length, 1)
        context = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        
        # 分类
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        # 返回分类输出和注意力权重
        return output, attention_weights.squeeze(2)
    
    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 投影到模型维度
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Informer编码器
        x, _ = self.informer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # 全局注意力池化
        attention_weights = self.global_attention(x)  # (batch_size, seq_length, 1)
        context = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        
        return context  # 返回上下文向量作为潜在表示


class SECNNInformerAttention(nn.Module):
    """
    融合SE Block的CNN + Informer + Attention模型
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38, 
                 filters=64, kernel_size=3, informer_d_model=256, 
                 informer_n_heads=8, informer_d_ff=512, informer_depth=2,
                 informer_factor=5, dropout_rate=0.3, use_se=True, se_reduction=16):
        super(SECNNInformerAttention, self).__init__()
        
        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.informer_d_model = informer_d_model
        self.use_se = use_se
        
        # CNN层
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.pool1 = nn.MaxPool1d(2)
        
        # SE Block 1 (CNN层后)
        if use_se:
            self.se1 = SEBlock(filters, reduction=se_reduction)
        
        self.conv2 = nn.Conv1d(filters, filters*2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters*2)
        self.pool2 = nn.MaxPool1d(2)
        
        # SE Block 2
        if use_se:
            self.se2 = SEBlock(filters*2, reduction=se_reduction)
        
        self.conv3 = nn.Conv1d(filters*2, filters*4, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(filters*4)
        
        # SE Block 3
        if use_se:
            self.se3 = SEBlock(filters*4, reduction=se_reduction)
        
        # 计算CNN层后的序列长度
        self.informer_seq_len = seq_length // 4
        
        # 线性层用于将CNN输出转换为Informer输入维度
        self.cnn_to_informer = nn.Linear(filters*4, informer_d_model)
        
        # 使用带SE Block的Informer编码器
        self.informer_encoder = SEInformerEncoder(
            d_model=informer_d_model,
            n_heads=informer_n_heads,
            d_ff=informer_d_ff,
            depth=informer_depth,
            dropout=dropout_rate,
            factor=informer_factor,
            use_se=use_se,
            se_reduction=se_reduction
        )
        
        # 全局注意力池化提取固定大小表示
        self.global_attention = nn.Sequential(
            nn.Linear(informer_d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类头
        self.fc1 = nn.Linear(informer_d_model, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        
        # 调整维度用于CNN
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)
        
        # CNN层
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_se:
            x = self.se1(x)
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_se:
            x = self.se2(x)
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        if self.use_se:
            x = self.se3(x)
        
        # 调整维度用于Informer
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/4, filters*4)
        
        # 转换为Informer输入维度
        x = self.cnn_to_informer(x)  # (batch_size, seq_length/4, informer_d_model)
        
        # Informer编码器 (带SE Block)
        x, attns = self.informer_encoder(x)  # (batch_size, seq_length/4, informer_d_model)
        
        # 全局注意力池化
        attention_weights = self.global_attention(x)  # (batch_size, seq_length/4, 1)
        context = torch.sum(x * attention_weights, dim=1)  # (batch_size, informer_d_model)
        
        # 分类
        x = F.relu(self.fc1(context))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        # 返回分类输出和注意力权重
        return output, attention_weights.squeeze(2)
    
    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 处理CNN
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_se:
            x = self.se1(x)
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_se:
            x = self.se2(x)
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        if self.use_se:
            x = self.se3(x)
        
        # 转换为Informer
        x = x.permute(0, 2, 1)
        x = self.cnn_to_informer(x)
        
        # 获取Informer表示
        x, _ = self.informer_encoder(x)
        
        # 全局注意力池化
        attention_weights = self.global_attention(x)
        context = torch.sum(x * attention_weights, dim=1)
        
        return context  # 返回上下文向量作为潜在表示


class SEPatchTSTClassifier(nn.Module):
    """
    添加SE Block的PatchTST分类器
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_size=16, stride=8, d_model=128, n_heads=8, num_layers=3,
                 dropout_rate=0.3, use_se=True, se_reduction=16):
        super(SEPatchTSTClassifier, self).__init__()
        
        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_se = use_se
        
        # 计算patch数量
        self.num_patches = (seq_length - patch_size) // stride + 1
        
        # Patch嵌入
        self.patch_embedding = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride
        )
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
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
        
        # SE Blocks (如果启用)
        if self.use_se:
            self.se_blocks = nn.ModuleList([
                SEBlock(d_model, reduction=se_reduction) for _ in range(num_layers)
            ])
        
        # 分类头
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_length, input_channels]
        
        # 调整维度用于卷积
        x = x.permute(0, 2, 1)  # [batch_size, input_channels, seq_length]
        
        # 提取patches
        x = self.patch_embedding(x)  # [batch_size, d_model, num_patches]
        
        # 调整维度用于Transformer
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, d_model]
        
        # 添加位置编码
        x = x + self.positional_encoding
        
        # 应用Transformer编码器层
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            if self.use_se:
                # 关键修复：在应用 SE Block 前调整维度
                # 当前 x 的形状是 [batch_size, num_patches, d_model]
                # 转置为 [batch_size, d_model, num_patches]
                x_for_se = x.transpose(1, 2)
                x_for_se = self.se_blocks[i](x_for_se)
                # 再转置回原来的形状
                x = x_for_se.transpose(1, 2)
        
        # 使用CLS token (第一个patch) 或全局平均池化
        # 这里选择全局平均池化
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # 保存注意力权重（这里使用均匀分布作为虚拟权重）
        batch_size = x.size(0)
        dummy_attention = torch.ones(batch_size, self.num_patches, device=x.device) / self.num_patches
        
        # 分类
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        return output, dummy_attention
    
    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 调整维度用于卷积
        x = x.permute(0, 2, 1)  # [batch_size, input_channels, seq_length]
        
        # 提取patches
        x = self.patch_embedding(x)  # [batch_size, d_model, num_patches]
        
        # 调整维度用于Transformer
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, d_model]
        
        # 添加位置编码
        x = x + self.positional_encoding
        
        # 应用Transformer编码器层
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            if self.use_se:
                # 关键修复：在应用 SE Block 前调整维度
                # 当前 x 的形状是 [batch_size, num_patches, d_model]
                # 转置为 [batch_size, d_model, num_patches]
                x_for_se = x.transpose(1, 2)
                x_for_se = self.se_blocks[i](x_for_se)
                # 再转置回原来的形状
                x = x_for_se.transpose(1, 2)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # 返回特征表示
        x = F.relu(self.fc1(x))
        
        return x  # 返回第一个全连接层的输出作为潜在表示
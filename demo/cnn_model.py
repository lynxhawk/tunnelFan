import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNClassifier(nn.Module):
    """
    简单的卷积神经网络(CNN)分类器，用于轴承故障诊断
    
    支持两种模式：
    1. 原始信号处理（use_features=False）：处理原始时间序列振动信号
    2. 特征数据处理（use_features=True）：处理提取的特征数据
    """

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 base_filters=64, kernel_sizes=[3, 5, 7], dropout_rate=0.3,
                 use_features=False, feature_input_dim=None):
        """
        初始化简单CNN分类器

        参数:
        - input_channels: 输入通道数，默认为3（X、Y、Z轴）
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - base_filters: 基础滤波器数量，默认为64
        - kernel_sizes: 卷积核大小列表，默认为[3, 5, 7]
        - dropout_rate: Dropout比率，默认为0.3
        - use_features: 是否使用特征数据模式，默认为False（使用原始信号）
        - feature_input_dim: 特征输入维度，仅在use_features=True时使用
        """
        super(SimpleCNNClassifier, self).__init__()

        self.use_features = use_features
        self.input_channels = input_channels
        self.seq_length = seq_length

        if not use_features:
            # 原始信号处理模式 - CNN架构
            # 第一层卷积块
            self.conv1 = nn.Conv1d(input_channels, base_filters,
                                   kernel_sizes[0], padding=kernel_sizes[0]//2)
            self.bn1 = nn.BatchNorm1d(base_filters)
            self.pool1 = nn.MaxPool1d(2)

            # 第二层卷积块
            self.conv2 = nn.Conv1d(base_filters, base_filters*2,
                                   kernel_sizes[1], padding=kernel_sizes[1]//2)
            self.bn2 = nn.BatchNorm1d(base_filters*2)
            self.pool2 = nn.MaxPool1d(2)

            # 第三层卷积块
            self.conv3 = nn.Conv1d(
                base_filters*2, base_filters*4, kernel_sizes[2], padding=kernel_sizes[2]//2)
            self.bn3 = nn.BatchNorm1d(base_filters*4)
            self.pool3 = nn.MaxPool1d(2)

            # 全局平均池化
            self.global_pool = nn.AdaptiveAvgPool1d(1)

            # 计算展平后的特征大小
            self.feature_size = base_filters*4
        else:
            # 特征数据处理模式 - 类MLP架构
            if feature_input_dim is None:
                raise ValueError("在特征模式下必须提供feature_input_dim参数")
                
            self.feature_input_dim = feature_input_dim
            
            # 特征处理层 - 使用类似MLP的结构但保持CNN风格
            self.feature_fc1 = nn.Linear(feature_input_dim, 256)
            self.feature_bn1 = nn.BatchNorm1d(256)
            self.feature_dropout1 = nn.Dropout(dropout_rate)
            
            self.feature_fc2 = nn.Linear(256, 128)
            self.feature_bn2 = nn.BatchNorm1d(128)
            self.feature_dropout2 = nn.Dropout(dropout_rate)
            
            # 设置特征大小与CNN分支一致，便于后续共享分类层
            self.feature_size = 128

        # 共享分类层
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        前向传播

        参数:
        - x: 输入数据
          - 当use_features=False时：形状为 (batch_size, seq_length, input_channels)
          - 当use_features=True时：形状为 (batch_size, feature_input_dim)

        返回:
        - logits: 类别预测的logits
        - attention_weights: 注意力权重（或虚拟注意力权重）
        """
        if not self.use_features:
            # 原始信号处理模式
            # 调整维度顺序以适应Conv1d
            if x.dim() == 3 and x.size(2) == self.input_channels:
                x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)

            # 第一层卷积块
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)

            # 第二层卷积块
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)

            # 第三层卷积块
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)

            # 全局平均池化
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)  # 展平特征
            
            # 用于虚拟注意力权重的序列长度
            seq_len = self.seq_length
        else:
            # 特征数据处理模式
            x = F.relu(self.feature_bn1(self.feature_fc1(x)))
            x = self.feature_dropout1(x)
            x = F.relu(self.feature_bn2(self.feature_fc2(x)))
            x = self.feature_dropout2(x)
            
            # 用于虚拟注意力权重的长度
            seq_len = 10  # 固定值，因为特征模式没有实际的序列长度

        # 共享分类层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)

        # 创建虚拟的注意力权重
        batch_size = x.size(0)
        dummy_attention = torch.ones(batch_size, seq_len, device=x.device) / seq_len

        return logits, dummy_attention

    def get_latent(self, x):
        """
        获取特征嵌入表示

        参数:
        - x: 输入数据

        返回:
        - 模型倒数第二层的输出作为特征嵌入
        """
        if not self.use_features:
            # 原始信号处理模式
            # 调整维度顺序以适应Conv1d
            if x.dim() == 3 and x.size(2) == self.input_channels:
                x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)

            # 前向传播到倒数第二层
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)

            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        else:
            # 特征数据处理模式
            x = F.relu(self.feature_bn1(self.feature_fc1(x)))
            x = self.feature_dropout1(x)
            x = F.relu(self.feature_bn2(self.feature_fc2(x)))
            x = self.feature_dropout2(x)

        # 共享层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        return x


# 测试代码示例（简化版）
if __name__ == "__main__":
    # 初始化模型
    model = SimpleCNNClassifier(input_channels=3, seq_length=1000, num_classes=38)
    
    # 测试输入
    x = torch.randn(16, 1000, 3)
    outputs, attention = model(x)
    print(f"输出形状: {outputs.shape}, 注意力权重形状: {attention.shape}")
    
    # 测试获取嵌入
    embedding = model.get_latent(x)
    print(f"嵌入形状: {embedding.shape}")
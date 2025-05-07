import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNClassifier(nn.Module):
    """
    简单的卷积神经网络(CNN)分类器，用于轴承故障诊断

    不包含注意力机制，专门用于处理原始的时间序列振动信号
    """

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 base_filters=64, kernel_sizes=[3, 5, 7], dropout_rate=0.3):
        """
        初始化简单CNN分类器

        参数:
        - input_channels: 输入通道数，默认为3（X、Y、Z轴）
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - base_filters: 基础滤波器数量，默认为64
        - kernel_sizes: 卷积核大小列表，默认为[3, 5, 7]
        - dropout_rate: Dropout比率，默认为0.3
        """
        super(SimpleCNNClassifier, self).__init__()

        self.input_channels = input_channels
        self.seq_length = seq_length

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

        # 全连接层
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        前向传播

        参数:
        - x: 输入数据，形状为 (batch_size, seq_length, input_channels)

        返回:
        - logits: 类别预测的logits
        - dummy_attention: 虚拟注意力权重（为了保持接口一致性）
        """
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

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)

        # 创建虚拟的注意力权重以保持接口一致性
        batch_size = x.size(0)
        dummy_attention = torch.ones(
            batch_size, self.seq_length, device=x.device) / self.seq_length

        return logits, dummy_attention

    def get_latent(self, x):
        """
        获取特征嵌入表示

        参数:
        - x: 输入数据

        返回:
        - 模型倒数第二层的输出作为特征嵌入
        """
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

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        return x


# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 16
    seq_length = 1000
    input_channels = 3
    num_classes = 38

    # 初始化模型
    model = SimpleCNNClassifier(input_channels, seq_length, num_classes)

    # 测试输入
    x = torch.randn(batch_size, seq_length, input_channels)
    outputs, attention = model(x)
    print(f"输出形状: {outputs.shape}, 注意力权重形状: {attention.shape}")

    # 测试获取嵌入
    embedding = model.get_latent(x)
    print(f"嵌入形状: {embedding.shape}")

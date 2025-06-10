import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    多层感知机(MLP)分类器，用于轴承故障诊断
    适用于特征提取后的数据
    """

    def __init__(self, input_dim, num_classes, hidden_layers=[512, 256, 128], dropout_rate=0.3):
        """
        初始化MLP分类器

        参数:
        - input_dim: 输入特征维度
        - num_classes: 分类类别数 (99个类别)
        - hidden_layers: 隐藏层维度列表
        - dropout_rate: Dropout比率
        """
        super(MLPClassifier, self).__init__()

        self.layers = nn.ModuleList()
        last_dim = input_dim

        # 添加隐藏层
        for dim in hidden_layers:
            self.layers.append(nn.Linear(last_dim, dim))
            self.layers.append(nn.BatchNorm1d(dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            last_dim = dim

        # 输出层
        self.output = nn.Linear(last_dim, num_classes)

    def forward(self, x):
        """前向传播"""
        # 如果输入是3D张量，展平为2D
        if x.dim() == 3:
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)

        # 通过所有层
        for layer in self.layers:
            x = layer(x)

        # 输出预测
        logits = self.output(x)

        # 创建虚拟注意力权重以保持接口一致性
        batch_size = logits.size(0)
        dummy_attention = torch.ones(batch_size, 10, device=x.device) / 10

        return logits, dummy_attention

    def get_latent(self, x):
        """获取特征嵌入表示"""
        if x.dim() == 3:
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)

        # 处理到倒数第二层
        for i in range(len(self.layers) - 4):  # 停在最后一个ReLU之前
            x = self.layers[i](x)

        return x


class CNNClassifier(nn.Module):
    """
    卷积神经网络(CNN)分类器，用于轴承故障诊断
    处理原始时间序列振动信号
    """

    def __init__(self, input_channels=1, seq_length=1000, num_classes=99,
                 base_filters=64, dropout_rate=0.3):
        """
        初始化CNN分类器

        参数:
        - input_channels: 输入通道数，默认为1
        - seq_length: 序列长度
        - num_classes: 分类类别数 (99个类别)
        - base_filters: 基础滤波器数量
        - dropout_rate: Dropout比率
        """
        super(CNNClassifier, self).__init__()

        self.input_channels = input_channels
        self.seq_length = seq_length

        # 第一层卷积块
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.pool1 = nn.MaxPool1d(2)

        # 第二层卷积块
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(base_filters*2)
        self.pool2 = nn.MaxPool1d(2)

        # 第三层卷积块
        self.conv3 = nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(base_filters*4)
        self.pool3 = nn.MaxPool1d(2)

        # 第四层卷积块
        self.conv4 = nn.Conv1d(base_filters*4, base_filters*8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(base_filters*8)
        self.pool4 = nn.MaxPool1d(2)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc1 = nn.Linear(base_filters*8, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """前向传播"""
        # 调整维度顺序以适应Conv1d: (batch_size, channels, seq_length)
        if x.dim() == 3 and x.size(2) == self.input_channels:
            x = x.permute(0, 2, 1)

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

        # 第四层卷积块
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 全连接分类层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)

        # 创建虚拟注意力权重
        batch_size = logits.size(0)
        dummy_attention = torch.ones(batch_size, self.seq_length//16, device=x.device) / (self.seq_length//16)

        return logits, dummy_attention

    def get_latent(self, x):
        """获取特征嵌入表示"""
        # 调整维度顺序
        if x.dim() == 3 and x.size(2) == self.input_channels:
            x = x.permute(0, 2, 1)

        # 卷积特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 处理到倒数第二层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        return x


class HybridClassifier(nn.Module):
    """
    混合分类器，结合CNN和MLP的优势
    可以同时处理原始信号和统计特征
    """

    def __init__(self, signal_input_channels=1, signal_seq_length=1000, 
                 feature_input_dim=15, num_classes=99, dropout_rate=0.3):
        """
        初始化混合分类器

        参数:
        - signal_input_channels: 信号输入通道数
        - signal_seq_length: 信号序列长度
        - feature_input_dim: 特征输入维度
        - num_classes: 分类类别数
        - dropout_rate: Dropout比率
        """
        super(HybridClassifier, self).__init__()

        # CNN分支处理原始信号
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(signal_input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # MLP分支处理统计特征
        self.mlp_branch = nn.Sequential(
            nn.Linear(feature_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),  # CNN特征256 + MLP特征256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, signal_input, feature_input):
        """前向传播"""
        # CNN分支
        if signal_input.dim() == 3 and signal_input.size(2) == 1:
            signal_input = signal_input.permute(0, 2, 1)
        
        cnn_features = self.cnn_branch(signal_input)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # MLP分支
        mlp_features = self.mlp_branch(feature_input)

        # 特征融合
        combined_features = torch.cat([cnn_features, mlp_features], dim=1)
        logits = self.fusion(combined_features)

        # 创建虚拟注意力权重
        batch_size = logits.size(0)
        dummy_attention = torch.ones(batch_size, 10, device=logits.device) / 10

        return logits, dummy_attention


def create_model(model_type, **kwargs):
    """模型工厂函数"""
    if model_type.upper() == 'MLP':
        return MLPClassifier(**kwargs)
    elif model_type.upper() == 'CNN':
        return CNNClassifier(**kwargs)
    elif model_type.upper() == 'HYBRID':
        return HybridClassifier(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


# 测试代码
if __name__ == "__main__":
    # 测试MLP模型
    print("测试MLP模型...")
    mlp_model = MLPClassifier(input_dim=15, num_classes=99)
    test_features = torch.randn(16, 15)
    mlp_out, mlp_att = mlp_model(test_features)
    print(f"MLP输出形状: {mlp_out.shape}")

    # 测试CNN模型
    print("\n测试CNN模型...")
    cnn_model = CNNClassifier(input_channels=1, seq_length=1000, num_classes=99)
    test_signals = torch.randn(16, 1000, 1)
    cnn_out, cnn_att = cnn_model(test_signals)
    print(f"CNN输出形状: {cnn_out.shape}")

    # 测试混合模型
    print("\n测试混合模型...")
    hybrid_model = HybridClassifier(
        signal_input_channels=1, signal_seq_length=1000,
        feature_input_dim=15, num_classes=99
    )
    hybrid_out, hybrid_att = hybrid_model(test_signals, test_features)
    print(f"混合模型输出形状: {hybrid_out.shape}")
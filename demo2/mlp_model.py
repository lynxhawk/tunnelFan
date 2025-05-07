import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    """
    简单的多层感知机(MLP)分类器，用于轴承故障诊断
    
    适用于特征提取后的数据或处理过的原始信号
    """
    def __init__(self, input_dim, num_classes, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        """
        初始化MLP分类器
        
        参数:
        - input_dim: 输入特征维度
        - num_classes: 分类类别数
        - hidden_layers: 隐藏层维度列表，默认为[256, 128, 64]
        - dropout_rate: Dropout比率，默认为0.3
        """
        super(MLPClassifier, self).__init__()
        
        # 网络层定义
        self.layers = nn.ModuleList()
        
        # 输入层
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
        """
        前向传播
        
        参数:
        - x: 输入数据，形状为 (batch_size, input_dim) 或 (batch_size, seq_length, input_channels)
        
        返回:
        - logits: 类别预测的logits
        - 为了与其他模型兼容，返回一个简单的占位注意力权重
        """
        # 处理输入
        if x.dim() == 3:
            # 如果输入是原始信号数据 (batch_size, seq_length, channels)
            # 将其展平为 (batch_size, seq_length * channels)
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
        
        # 通过隐藏层
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        logits = self.output(x)
        
        # 创建一个虚拟的注意力权重，以保持接口一致性
        batch_size = x.size(0)
        dummy_attention = torch.ones(batch_size, 10, device=x.device) / 10  # 任意长度10
        
        return logits, dummy_attention
    
    def get_latent(self, x):
        """
        获取特征嵌入表示
        
        参数:
        - x: 输入数据
        
        返回:
        - 模型倒数第二层的输出作为特征嵌入
        """
        if x.dim() == 3:
            # 如果输入是原始信号数据 (batch_size, seq_length, channels)
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
        
        # 只处理到倒数第二层
        for i in range(len(self.layers) - 4):  # -4是为了在最后一个隐藏层的ReLU处停止
            x = self.layers[i](x)
        
        return x


# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 16
    seq_length = 1000
    input_channels = 3
    num_classes = 38
    
    # 测试特征输入
    input_dim = 100
    mlp_feature = MLPClassifier(input_dim, num_classes)
    x_features = torch.randn(batch_size, input_dim)
    outputs, attention = mlp_feature(x_features)
    print(f"特征输入 - 输出形状: {outputs.shape}, 注意力权重形状: {attention.shape}")
    
    # 测试原始信号输入
    mlp_signal = MLPClassifier(seq_length * input_channels, num_classes)
    x_raw = torch.randn(batch_size, seq_length, input_channels)
    outputs, attention = mlp_signal(x_raw)
    print(f"原始信号输入 - 输出形状: {outputs.shape}, 注意力权重形状: {attention.shape}")
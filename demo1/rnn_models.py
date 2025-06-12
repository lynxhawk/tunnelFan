"""
LSTM和GRU模型实现
用于轴承故障诊断的循环神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMClassifier(nn.Module):
    """LSTM分类器"""
    
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10, 
                 hidden_dim=128, num_layers=2, dropout_rate=0.3, 
                 bidirectional=False, attention=False):
        """
        Args:
            input_channels: 输入通道数
            seq_length: 序列长度
            num_classes: 分类数量
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout_rate: Dropout比率
            bidirectional: 是否使用双向LSTM
            attention: 是否使用注意力机制
        """
        super(LSTMClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # 注意力机制
        if attention:
            self.attention_layer = AttentionLayer(lstm_output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    
    def forward(self, x):
        """前向传播
        Args:
            x: 输入数据 [batch_size, seq_length, input_channels] 或 [batch_size, input_channels, seq_length]
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
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch_size, seq_length, hidden_dim * directions]
        
        if self.attention:
            # 使用注意力机制
            features = self.attention_layer(lstm_out)  # [batch_size, hidden_dim * directions]
        else:
            # 使用最后一个时间步的输出
            if self.bidirectional:
                # 双向LSTM: 取前向和后向的最后输出
                features = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_dim * 2]
            else:
                features = hidden[-1]  # [batch_size, hidden_dim]
        
        # Dropout
        features = self.dropout(features)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features
    
    def get_latent(self, x):
        """获取潜在特征表示"""
        _, features = self.forward(x)
        return features


class GRUClassifier(nn.Module):
    """GRU分类器"""
    
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10, 
                 hidden_dim=128, num_layers=2, dropout_rate=0.3, 
                 bidirectional=False, attention=False):
        """
        Args:
            input_channels: 输入通道数
            seq_length: 序列长度
            num_classes: 分类数量
            hidden_dim: GRU隐藏层维度
            num_layers: GRU层数
            dropout_rate: Dropout比率
            bidirectional: 是否使用双向GRU
            attention: 是否使用注意力机制
        """
        super(GRUClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算GRU输出维度
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # 注意力机制
        if attention:
            self.attention_layer = AttentionLayer(gru_output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_output_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    
    def forward(self, x):
        """前向传播
        Args:
            x: 输入数据 [batch_size, seq_length, input_channels] 或 [batch_size, input_channels, seq_length]
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
        
        # GRU前向传播
        gru_out, hidden = self.gru(x)  # gru_out: [batch_size, seq_length, hidden_dim * directions]
        
        if self.attention:
            # 使用注意力机制
            features = self.attention_layer(gru_out)  # [batch_size, hidden_dim * directions]
        else:
            # 使用最后一个时间步的输出
            if self.bidirectional:
                # 双向GRU: 取前向和后向的最后输出
                features = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_dim * 2]
            else:
                features = hidden[-1]  # [batch_size, hidden_dim]
        
        # Dropout
        features = self.dropout(features)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features
    
    def get_latent(self, x):
        """获取潜在特征表示"""
        _, features = self.forward(x)
        return features


class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: [batch_size, seq_length, hidden_dim]
        Returns:
            attended_output: [batch_size, hidden_dim]
        """
        # 计算注意力分数
        attention_scores = self.attention_weights(lstm_output)  # [batch_size, seq_length, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_length, 1]
        
        # 应用注意力权重
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, hidden_dim]
        
        return attended_output


class StackedLSTMClassifier(nn.Module):
    """堆叠LSTM分类器 - 多尺度特征提取"""
    
    def __init__(self, input_channels=1, seq_length=1000, num_classes=10, 
                 hidden_dims=[64, 128, 64], dropout_rate=0.3):
        """
        Args:
            input_channels: 输入通道数
            seq_length: 序列长度
            num_classes: 分类数量
            hidden_dims: 每层LSTM的隐藏维度列表
            dropout_rate: Dropout比率
        """
        super(StackedLSTMClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_dims = hidden_dims
        
        # 构建堆叠LSTM层
        self.lstm_layers = nn.ModuleList()
        input_dim = input_channels
        
        for i, hidden_dim in enumerate(hidden_dims):
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0  # 在层间单独添加dropout
            )
            self.lstm_layers.append(lstm)
            input_dim = hidden_dim
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类层
        total_hidden_dim = sum(hidden_dims)  # 连接所有层的输出
        self.classifier = nn.Sequential(
            nn.Linear(total_hidden_dim, total_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(total_hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)
        
        # 调整输入维度
        if x.dim() == 3 and x.size(1) == self.input_channels:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # 收集所有层的特征
        layer_features = []
        current_input = x
        
        for lstm in self.lstm_layers:
            lstm_out, (hidden, _) = lstm(current_input)
            
            # 使用全局平均池化获取层特征
            layer_feature = self.global_avg_pool(lstm_out.transpose(1, 2)).squeeze(-1)
            layer_features.append(layer_feature)
            
            # 添加残差连接（如果维度匹配）
            if current_input.size(-1) == lstm_out.size(-1):
                current_input = lstm_out + current_input
            else:
                current_input = lstm_out
            
            current_input = self.dropout(current_input)
        
        # 连接所有层的特征
        features = torch.cat(layer_features, dim=1)
        features = self.dropout(features)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features
    
    def get_latent(self, x):
        """获取潜在特征表示"""
        _, features = self.forward(x)
        return features


def create_lstm_model(input_channels=1, seq_length=1000, num_classes=10, 
                     hidden_dim=128, num_layers=2, dropout_rate=0.3, 
                     bidirectional=False, attention=False, model_variant='standard'):
    """创建LSTM模型的工厂函数
    
    Args:
        model_variant: 模型变体 ['standard', 'stacked']
    """
    if model_variant == 'stacked':
        hidden_dims = [hidden_dim // 2, hidden_dim, hidden_dim // 2]
        return StackedLSTMClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
    else:
        return LSTMClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            attention=attention
        )


def create_gru_model(input_channels=1, seq_length=1000, num_classes=10, 
                    hidden_dim=128, num_layers=2, dropout_rate=0.3, 
                    bidirectional=False, attention=False):
    """创建GRU模型的工厂函数"""
    return GRUClassifier(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        attention=attention
    )


# 测试代码
if __name__ == "__main__":
    # 测试LSTM模型
    print("测试LSTM模型...")
    lstm_model = create_lstm_model(
        input_channels=1,
        seq_length=1000,
        num_classes=10,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        attention=True
    )
    
    # 测试输入
    test_input = torch.randn(32, 1000, 1)  # [batch_size, seq_length, channels]
    lstm_output, lstm_features = lstm_model(test_input)
    print(f"LSTM输出形状: {lstm_output.shape}, 特征形状: {lstm_features.shape}")
    
    # 测试GRU模型
    print("\n测试GRU模型...")
    gru_model = create_gru_model(
        input_channels=1,
        seq_length=1000,
        num_classes=10,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        attention=True
    )
    
    gru_output, gru_features = gru_model(test_input)
    print(f"GRU输出形状: {gru_output.shape}, 特征形状: {gru_features.shape}")
    
    # 测试堆叠LSTM模型
    print("\n测试堆叠LSTM模型...")
    stacked_lstm = create_lstm_model(
        input_channels=1,
        seq_length=1000,
        num_classes=10,
        hidden_dim=128,
        model_variant='stacked'
    )
    
    stacked_output, stacked_features = stacked_lstm(test_input)
    print(f"堆叠LSTM输出形状: {stacked_output.shape}, 特征形状: {stacked_features.shape}")
    
    print("\n所有模型测试完成！")
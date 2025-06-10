import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PureLSTMClassifier(nn.Module):
    """
    纯LSTM分类器（不包含CNN层）
    """
    
    def __init__(self, input_channels, seq_length, num_classes, 
                 lstm_hidden=128, num_layers=2, dropout_rate=0.3, 
                 bidirectional=True, pooling_type='mean'):
        """
        参数:
        - input_channels: 输入通道数（例如3个轴的振动信号）
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - lstm_hidden: LSTM隐藏单元数
        - num_layers: LSTM层数
        - dropout_rate: Dropout比率
        - bidirectional: 是否使用双向LSTM
        - pooling_type: 池化类型 ('mean', 'max', 'last')
        """
        super(PureLSTMClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling_type = pooling_type
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
        
        # 用于获取潜在表示
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, input_channels)
        
        返回:
        - outputs: 分类输出
        - pooled_output: 池化后的输出（作为注意力权重的替代）
        """
        batch_size = x.size(0)
        
        # LSTM前向传播
        # x: (batch_size, seq_length, input_channels)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch_size, seq_length, lstm_hidden * num_directions)
        
        # 根据池化类型处理LSTM输出
        if self.pooling_type == 'mean':
            # 平均池化
            pooled_output = torch.mean(lstm_out, dim=1)
        elif self.pooling_type == 'max':
            # 最大池化
            pooled_output, _ = torch.max(lstm_out, dim=1)
        elif self.pooling_type == 'last':
            # 使用最后一个时间步的输出
            if self.bidirectional:
                # 对于双向LSTM，取前向和后向的最后输出
                forward_out = lstm_out[:, -1, :self.lstm_hidden]
                backward_out = lstm_out[:, 0, self.lstm_hidden:]
                pooled_output = torch.cat([forward_out, backward_out], dim=1)
            else:
                pooled_output = lstm_out[:, -1, :]
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        outputs = self.classifier(pooled_output)
        
        # 创建注意力权重的近似（基于LSTM输出的方差）
        attention_weights = torch.var(lstm_out, dim=2)  # (batch_size, seq_length)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return outputs, attention_weights
    
    def get_latent(self, x):
        """
        获取潜在表示
        """
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 池化
        if self.pooling_type == 'mean':
            pooled_output = torch.mean(lstm_out, dim=1)
        elif self.pooling_type == 'max':
            pooled_output, _ = torch.max(lstm_out, dim=1)
        elif self.pooling_type == 'last':
            if self.bidirectional:
                forward_out = lstm_out[:, -1, :self.lstm_hidden]
                backward_out = lstm_out[:, 0, self.lstm_hidden:]
                pooled_output = torch.cat([forward_out, backward_out], dim=1)
            else:
                pooled_output = lstm_out[:, -1, :]
        
        # 通过特征提取器
        return self.feature_extractor(pooled_output)


class PureGRUClassifier(nn.Module):
    """
    纯GRU分类器（不包含CNN层）
    """
    
    def __init__(self, input_channels, seq_length, num_classes, 
                 gru_hidden=128, num_layers=2, dropout_rate=0.3, 
                 bidirectional=True, pooling_type='mean'):
        """
        参数:
        - input_channels: 输入通道数（例如3个轴的振动信号）
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - gru_hidden: GRU隐藏单元数
        - num_layers: GRU层数
        - dropout_rate: Dropout比率
        - bidirectional: 是否使用双向GRU
        - pooling_type: 池化类型 ('mean', 'max', 'last')
        """
        super(PureGRUClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.gru_hidden = gru_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling_type = pooling_type
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算GRU输出维度
        gru_output_dim = gru_hidden * 2 if bidirectional else gru_hidden
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_output_dim // 2, num_classes)
        )
        
        # 用于获取潜在表示
        self.feature_extractor = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, input_channels)
        
        返回:
        - outputs: 分类输出
        - pooled_output: 池化后的输出（作为注意力权重的替代）
        """
        batch_size = x.size(0)
        
        # GRU前向传播
        # x: (batch_size, seq_length, input_channels)
        gru_out, h_n = self.gru(x)
        # gru_out: (batch_size, seq_length, gru_hidden * num_directions)
        
        # 根据池化类型处理GRU输出
        if self.pooling_type == 'mean':
            # 平均池化
            pooled_output = torch.mean(gru_out, dim=1)
        elif self.pooling_type == 'max':
            # 最大池化
            pooled_output, _ = torch.max(gru_out, dim=1)
        elif self.pooling_type == 'last':
            # 使用最后一个时间步的输出
            if self.bidirectional:
                # 对于双向GRU，取前向和后向的最后输出
                forward_out = gru_out[:, -1, :self.gru_hidden]
                backward_out = gru_out[:, 0, self.gru_hidden:]
                pooled_output = torch.cat([forward_out, backward_out], dim=1)
            else:
                pooled_output = gru_out[:, -1, :]
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        outputs = self.classifier(pooled_output)
        
        # 创建注意力权重的近似（基于GRU输出的方差）
        attention_weights = torch.var(gru_out, dim=2)  # (batch_size, seq_length)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return outputs, attention_weights
    
    def get_latent(self, x):
        """
        获取潜在表示
        """
        batch_size = x.size(0)
        
        # GRU前向传播
        gru_out, h_n = self.gru(x)
        
        # 池化
        if self.pooling_type == 'mean':
            pooled_output = torch.mean(gru_out, dim=1)
        elif self.pooling_type == 'max':
            pooled_output, _ = torch.max(gru_out, dim=1)
        elif self.pooling_type == 'last':
            if self.bidirectional:
                forward_out = gru_out[:, -1, :self.gru_hidden]
                backward_out = gru_out[:, 0, self.gru_hidden:]
                pooled_output = torch.cat([forward_out, backward_out], dim=1)
            else:
                pooled_output = gru_out[:, -1, :]
        
        # 通过特征提取器
        return self.feature_extractor(pooled_output)


class LSTMWithAttention(nn.Module):
    """
    带注意力机制的LSTM分类器
    """
    
    def __init__(self, input_channels, seq_length, num_classes, 
                 lstm_hidden=128, num_layers=2, dropout_rate=0.3, 
                 bidirectional=True, attention_dim=64):
        """
        参数:
        - input_channels: 输入通道数
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - lstm_hidden: LSTM隐藏单元数
        - num_layers: LSTM层数
        - dropout_rate: Dropout比率
        - bidirectional: 是否使用双向LSTM
        - attention_dim: 注意力机制的维度
        """
        super(LSTMWithAttention, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
        
        # 用于获取潜在表示
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        """
        前向传播
        """
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch_size, seq_length, lstm_hidden * num_directions)
        
        # 计算注意力权重
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch_size, seq_length)
        
        # 应用注意力权重
        attended_output = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        # attended_output: (batch_size, lstm_hidden * num_directions)
        
        # Dropout
        attended_output = self.dropout(attended_output)
        
        # 分类
        outputs = self.classifier(attended_output)
        
        return outputs, attention_weights
    
    def get_latent(self, x):
        """
        获取潜在表示
        """
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 计算注意力权重
        attention_scores = self.attention(lstm_out)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)
        
        # 应用注意力权重
        attended_output = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        # 通过特征提取器
        return self.feature_extractor(attended_output)


class GRUWithAttention(nn.Module):
    """
    带注意力机制的GRU分类器
    """
    
    def __init__(self, input_channels, seq_length, num_classes, 
                 gru_hidden=128, num_layers=2, dropout_rate=0.3, 
                 bidirectional=True, attention_dim=64):
        """
        参数:
        - input_channels: 输入通道数
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - gru_hidden: GRU隐藏单元数
        - num_layers: GRU层数
        - dropout_rate: Dropout比率
        - bidirectional: 是否使用双向GRU
        - attention_dim: 注意力机制的维度
        """
        super(GRUWithAttention, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.gru_hidden = gru_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算GRU输出维度
        gru_output_dim = gru_hidden * 2 if bidirectional else gru_hidden
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(gru_output_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_output_dim // 2, num_classes)
        )
        
        # 用于获取潜在表示
        self.feature_extractor = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        """
        前向传播
        """
        batch_size = x.size(0)
        
        # GRU前向传播
        gru_out, h_n = self.gru(x)
        # gru_out: (batch_size, seq_length, gru_hidden * num_directions)
        
        # 计算注意力权重
        attention_scores = self.attention(gru_out)  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch_size, seq_length)
        
        # 应用注意力权重
        attended_output = torch.sum(gru_out * attention_weights.unsqueeze(-1), dim=1)
        # attended_output: (batch_size, gru_hidden * num_directions)
        
        # Dropout
        attended_output = self.dropout(attended_output)
        
        # 分类
        outputs = self.classifier(attended_output)
        
        return outputs, attention_weights
    
    def get_latent(self, x):
        """
        获取潜在表示
        """
        batch_size = x.size(0)
        
        # GRU前向传播
        gru_out, h_n = self.gru(x)
        
        # 计算注意力权重
        attention_scores = self.attention(gru_out)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)
        
        # 应用注意力权重
        attended_output = torch.sum(gru_out * attention_weights.unsqueeze(-1), dim=1)
        
        # 通过特征提取器
        return self.feature_extractor(attended_output)
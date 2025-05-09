import torch
import torch.nn as nn
import torch.nn.functional as F
from enhanced_attention import EnhancedSelfAttention, MultiHeadSelfAttention, HybridAttention

class CNNLSTM_EnhancedAttention(nn.Module):
    """CNN + LSTM + 增强注意力机制模型"""

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 filters=64, kernel_size=3, lstm_hidden=100, dropout_rate=0.3,
                 attention_type='enhanced', attention_params=None):
        super(CNNLSTM_EnhancedAttention, self).__init__()

        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.attention_type = attention_type

        # CNN层 - 第一层
        self.conv1 = nn.Conv1d(input_channels, filters,
                               kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.pool1 = nn.MaxPool1d(2)

        # CNN层 - 第二层
        self.conv2 = nn.Conv1d(
            filters, filters*2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters*2)
        self.pool2 = nn.MaxPool1d(2)

        # CNN层 - 第三层
        self.conv3 = nn.Conv1d(filters*2, filters*4,
                               kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(filters*4)

        # 计算LSTM输入序列长度（经过池化后）
        self.lstm_seq_len = seq_length // 4

        # 双向LSTM层
        self.lstm = nn.LSTM(filters*4, lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 设置默认注意力参数
        if attention_params is None:
            attention_params = {}

        # 根据指定的类型选择注意力机制
        if attention_type == 'enhanced':
            # 增强版自注意力
            self.attention = EnhancedSelfAttention(
                input_dim=lstm_hidden*2,
                attention_dim=attention_params.get('attention_dim', 64),
                temperature=attention_params.get('temperature', 1.0),
                dropout=attention_params.get('dropout', dropout_rate)
            )
        elif attention_type == 'multihead':
            # 多头注意力
            self.attention = MultiHeadSelfAttention(
                input_dim=lstm_hidden*2,
                num_heads=attention_params.get('num_heads', 4),
                head_dim=attention_params.get('head_dim', lstm_hidden//2),
                dropout=attention_params.get('dropout', dropout_rate)
            )
        elif attention_type == 'hybrid':
            # 混合注意力
            self.attention = HybridAttention(
                input_dim=lstm_hidden*2,
                attention_dim=attention_params.get('attention_dim', 64),
                position_bias=attention_params.get('position_bias', True)
            )
        else:
            # 默认使用原始自注意力（来自CNNLSTM_Attention）
            from pytorch_cnn_bilstm_attention import SelfAttention
            self.attention = SelfAttention(lstm_hidden*2)

        # 全连接层
        self.fc1 = nn.Linear(lstm_hidden*2, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)

        # 调整输入维度以适应Conv1d
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)

        # CNN层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        # 调整维度以适应LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/4, filters*4)

        # LSTM层
        x, _ = self.lstm(x)
        x = self.dropout1(x)

        # 注意力层
        context, attention_weights = self.attention(x)

        # 全连接层
        x = F.relu(self.fc1(context))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, attention_weights

    def get_latent(self, x):
        """获取特征嵌入表示用于可视化"""
        # 调整输入维度以适应Conv1d
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)

        # CNN层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        # 调整维度以适应LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/4, filters*4)

        # LSTM层
        x, _ = self.lstm(x)
        x = self.dropout1(x)

        # 注意力层
        context, _ = self.attention(x)

        # 全连接层
        x = F.relu(self.fc1(context))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))

        return x
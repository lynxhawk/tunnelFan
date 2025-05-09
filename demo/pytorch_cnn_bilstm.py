import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM_NoAttention(nn.Module):
    """CNN + LSTM 模型，无注意力机制版本"""

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 filters=64, kernel_size=3, lstm_hidden=100, dropout_rate=0.3,
                 pooling_type='last'):
        """
        初始化CNN-LSTM模型（无注意力机制）
        
        参数:
        - input_channels: 输入通道数，默认为3（X、Y、Z轴）
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - filters: CNN滤波器数量
        - kernel_size: 卷积核大小
        - lstm_hidden: LSTM隐藏单元数量
        - dropout_rate: Dropout比率
        - pooling_type: 池化类型，'last'表示使用LSTM最后一个时间步的输出，
                        'mean'表示使用所有时间步的平均值
        """
        super(CNNLSTM_NoAttention, self).__init__()

        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.pooling_type = pooling_type

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
        x, (h_n, c_n) = self.lstm(x)
        x = self.dropout1(x)

        # 特征提取 - 不使用注意力机制
        if self.pooling_type == 'last':
            # 使用最后一个时间步的输出
            batch_size = x.size(0)
            # 获取最后一个时间步的输出（双向LSTM，需要连接两个方向）
            forward_output = x[:, -1, :self.lstm_hidden]
            backward_output = x[:, 0, self.lstm_hidden:]
            context = torch.cat([forward_output, backward_output], dim=1)
        else:  # 'mean'
            # 使用所有时间步的平均值
            context = torch.mean(x, dim=1)

        # 全连接层
        x = F.relu(self.fc1(context))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # 创建虚拟注意力权重以保持接口一致性
        batch_size = context.size(0)
        dummy_attention = torch.ones(
            batch_size, self.lstm_seq_len, device=x.device) / self.lstm_seq_len

        return x, dummy_attention

    def get_latent(self, x):
        """
        获取特征嵌入表示

        参数:
        - x: 输入数据

        返回:
        - 模型倒数第二层的输出作为特征嵌入
        """
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
        x, (h_n, c_n) = self.lstm(x)

        # 特征提取 - 不使用注意力机制
        if self.pooling_type == 'last':
            # 使用最后一个时间步的输出
            batch_size = x.size(0)
            # 获取最后一个时间步的输出（双向LSTM，需要连接两个方向）
            forward_output = x[:, -1, :self.lstm_hidden]
            backward_output = x[:, 0, self.lstm_hidden:]
            context = torch.cat([forward_output, backward_output], dim=1)
        else:  # 'mean'
            # 使用所有时间步的平均值
            context = torch.mean(x, dim=1)

        # 处理到全连接层
        x = F.relu(self.fc1(context))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))

        return x
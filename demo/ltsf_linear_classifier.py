import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, input_channels, seq_length, output_dim=64, 
                 individual=False, kernel_size=25):
        super(DLinear, self).__init__()
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.individual = individual
        
        # Decomposition
        self.decomposition = SeriesDecomp(kernel_size)
        
        # Individual linear layers for each channel
        if self.individual:
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Seasonal = nn.ModuleList()
            
            for i in range(input_channels):
                self.Linear_Trend.append(nn.Linear(seq_length, output_dim))
                self.Linear_Seasonal.append(nn.Linear(seq_length, output_dim))
        else:
            self.Linear_Trend = nn.Linear(seq_length, output_dim)
            self.Linear_Seasonal = nn.Linear(seq_length, output_dim)

    def forward(self, x):
        # x: [Batch, seq_length, channels]
        
        # First decompose the time series
        trend, seasonal = self.decomposition(x)
        
        if self.individual:
            trend_output = torch.zeros([x.size(0), self.output_dim, x.size(2)], 
                                       dtype=x.dtype, device=x.device)
            seasonal_output = torch.zeros([x.size(0), self.output_dim, x.size(2)], 
                                          dtype=x.dtype, device=x.device)
            
            for i in range(x.size(2)):
                trend_output[:, :, i] = self.Linear_Trend[i](trend[:, :, i])
                seasonal_output[:, :, i] = self.Linear_Seasonal[i](seasonal[:, :, i])
        else:
            # Reshape for the linear layers
            batch_size, seq_len, channels = x.size()
            trend_flat = trend.reshape(batch_size, seq_len, channels)
            seasonal_flat = seasonal.reshape(batch_size, seq_len, channels)
            
            # Apply linear transformation
            trend_output = self.Linear_Trend(trend_flat.transpose(1, 2)).transpose(1, 2)
            seasonal_output = self.Linear_Seasonal(seasonal_flat.transpose(1, 2)).transpose(1, 2)
        
        # Final output is the sum of trend and seasonality components
        x = trend_output + seasonal_output
        return x


class LTSFLinearClassifier(nn.Module):
    """
    LTSF-Linear model adapted for classification
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 hidden_dim=64, kernel_size=25, individual=False, dropout_rate=0.3):
        super(LTSFLinearClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # DLinear model for feature extraction
        self.dlinear = DLinear(
            input_channels=input_channels,
            seq_length=seq_length,
            output_dim=hidden_dim,
            individual=individual,
            kernel_size=kernel_size
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim * input_channels, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: [Batch, seq_length, channels]
        
        # Apply DLinear module
        x = self.dlinear(x)  # [Batch, hidden_dim, channels]
        
        # Reshape for classification
        batch_size = x.size(0)
        
        # Transpose to [Batch, channels, hidden_dim]
        x = x.transpose(1, 2)
        
        # Flatten features for classification
        x = x.reshape(batch_size, -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        # For compatibility with other models that return attention weights
        # Create dummy attention weights
        dummy_attention = torch.ones(batch_size, self.seq_length // 4, device=x.device)
        dummy_attention = dummy_attention / (self.seq_length // 4)
        
        return output, dummy_attention

    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # Apply DLinear module
        x = self.dlinear(x)  # [Batch, hidden_dim, channels]
        
        # Reshape for classification
        batch_size = x.size(0)
        
        # Transpose to [Batch, channels, hidden_dim]
        x = x.transpose(1, 2)
        
        # Flatten features
        x = x.reshape(batch_size, -1)
        
        # Get representation from first fully-connected layer
        x = F.relu(self.fc1(x))
        
        return x
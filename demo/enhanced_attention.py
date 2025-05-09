import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSelfAttention(nn.Module):
    """增强版自注意力层，带有可调整参数"""

    def __init__(self, input_dim, attention_dim=64, temperature=1.0, dropout=0.1):
        super(EnhancedSelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.temperature = temperature  # 控制softmax的锐度

        # 注意力层的权重矩阵
        self.W = nn.Linear(input_dim, attention_dim)
        self.u = nn.Linear(attention_dim, 1, bias=False)
        
        # 添加层归一化和dropout
        self.layernorm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 应用层归一化
        x_norm = self.layernorm(x)
        
        # 计算注意力得分
        u = torch.tanh(self.W(x_norm))  # (batch_size, seq_len, attention_dim)
        u = self.dropout(u)  # 应用dropout
        scores = self.u(u)  # (batch_size, seq_len, 1)

        # 应用温度系数并获取注意力权重
        attention_weights = F.softmax(scores / self.temperature, dim=1)

        # 计算加权和
        context = torch.sum(x * attention_weights, dim=1)

        return context, attention_weights.squeeze(-1)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力层"""

    def __init__(self, input_dim, num_heads=4, head_dim=32, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        
        # 确保输入维度能被头数整除，否则进行调整
        if self.total_dim != input_dim:
            self.input_projection = nn.Linear(input_dim, self.total_dim)
        else:
            self.input_projection = nn.Identity()
            
        # 多头注意力投影矩阵
        self.q_linear = nn.Linear(self.total_dim, self.total_dim)
        self.k_linear = nn.Linear(self.total_dim, self.total_dim)
        self.v_linear = nn.Linear(self.total_dim, self.total_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(self.total_dim, input_dim)
        
        # 层归一化
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.layernorm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = head_dim ** -0.5

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 层归一化和残差连接准备
        residual = x
        x = self.layernorm1(x)
        
        # 投影输入到多头维度空间
        x = self.input_projection(x)
        
        # 线性投影得到查询、键、值
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # 将输出拆分为多个头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力加权上下文
        context = torch.matmul(attention_weights, v)
        
        # 转置并重塑以合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.total_dim)
        
        # 输出投影
        output = self.output_projection(context)
        
        # 残差连接和最终层归一化
        output = residual + self.dropout(output)
        output = self.layernorm2(output)
        
        # 为了与其他注意力机制兼容，我们需要返回一个上下文向量和注意力权重
        # 我们使用平均池化得到上下文向量
        global_context = torch.mean(output, dim=1)
        
        # 将多头注意力权重平均为单个注意力权重 (用于可视化)
        avg_attention = torch.mean(attention_weights, dim=1)[:, 0, :]  # 取第一个头的权重作为示例
        
        return global_context, avg_attention


# 定义混合注意力机制，组合自注意力和循环注意力
class HybridAttention(nn.Module):
    """混合注意力机制：结合自注意力和基于位置的加权"""

    def __init__(self, input_dim, attention_dim=64, position_bias=True):
        super(HybridAttention, self).__init__()
        self.attention_dim = attention_dim
        self.position_bias = position_bias

        # 主注意力层的权重矩阵
        self.W = nn.Linear(input_dim, attention_dim)
        self.u = nn.Linear(attention_dim, 1, bias=False)
        
        # 位置编码偏置，如果启用
        if position_bias:
            self.position_bias = nn.Parameter(torch.zeros(1, 1000, 1))  # 假设最大序列长度为1000
            nn.init.normal_(self.position_bias, mean=0.0, std=0.02)
        
        # 层归一化
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 应用层归一化
        x_norm = self.layernorm(x)
        
        # 计算注意力得分
        u = torch.tanh(self.W(x_norm))  # (batch_size, seq_len, attention_dim)
        scores = self.u(u)  # (batch_size, seq_len, 1)
        
        # 如果启用位置偏置，添加到分数中
        if hasattr(self, 'position_bias'):
            position_bias = self.position_bias[:, :seq_len, :]
            scores = scores + position_bias
            
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=1)
        
        # 计算加权和
        context = torch.sum(x * attention_weights, dim=1)
        
        return context, attention_weights.squeeze(-1)
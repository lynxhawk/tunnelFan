import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ProbAttention(nn.Module):
    """Informer的概率稀疏注意力机制"""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # 计算采样的Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # 找出稀疏度量下的Top_k查询
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # 使用简化的Q计算Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
        else:  # 使用掩码
            assert (L_Q == L_V)  # 仅用于自注意力
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        
        # 创建上下文的副本以避免原地操作错误
        context_in = context_in.clone()
        
        context_in[torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index, :] = torch.matmul(attn, V)
                
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, u, U_part)

        # 添加缩放因子
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # 获取上下文
        context = self._get_initial_context(values, L_Q)
        # 使用选定的top_k查询更新上下文
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q)

        return context.transpose(2, 1), attn


class AttentionLayer(nn.Module):
    """注意力层"""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 使用reshape而不是view，以处理可能的非连续张量
        queries = self.query_projection(queries).reshape(B, L, H, -1)
        keys = self.key_projection(keys).reshape(B, S, H, -1)
        values = self.value_projection(values).reshape(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        
        # 使用reshape而不是view
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        new_x, attn = self.attention(
            x, x, x
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x+y), attn


class InformerEncoder(nn.Module):
    """Informer编码器"""

    def __init__(self, d_model, n_heads, d_ff, depth, dropout=0.1, factor=5):
        super(InformerEncoder, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth

        # 各层定义
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(
                        False, factor, attention_dropout=dropout, output_attention=True),
                    d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout
            ) for _ in range(depth)
        ])

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]

        # 应用位置编码
        x = self.position_encoding(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # 应用编码器层
        attns = []
        for layer in self.encoder_layers:
            x, attn = layer(x)
            attns.append(attn)

        return x, attns


class DirectInformerClassifier(nn.Module):
    """直接使用原始信号的Informer分类器"""

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 d_model=256, n_heads=8, d_ff=512, depth=2,
                 factor=5, dropout_rate=0.3):
        super(DirectInformerClassifier, self).__init__()

        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.d_model = d_model

        # 将原始信号映射到模型维度
        self.input_projection = nn.Linear(input_channels, d_model)

        # Informer编码器
        self.informer_encoder = InformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=depth,
            dropout=dropout_rate,
            factor=factor
        )

        # 全局注意力池化提取固定大小表示
        self.global_attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # 分类头
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)

        # 投影到模型维度
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)

        # Informer编码器
        # (batch_size, seq_length, d_model)
        x, attns = self.informer_encoder(x)

        # 全局注意力池化
        attention_weights = self.global_attention(
            x)  # (batch_size, seq_length, 1)
        # (batch_size, d_model)
        context = torch.sum(x * attention_weights, dim=1)

        # 分类
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # 返回分类输出和注意力权重
        return output, attention_weights.squeeze(2)

    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 投影到模型维度
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)

        # Informer编码器
        x, _ = self.informer_encoder(x)  # (batch_size, seq_length, d_model)

        # 全局注意力池化
        attention_weights = self.global_attention(
            x)  # (batch_size, seq_length, 1)
        # (batch_size, d_model)
        context = torch.sum(x * attention_weights, dim=1)

        return context  # 返回上下文向量作为潜在表示
# 使用特征作为输入的Informer模型


class FeatureInformerClassifier(nn.Module):
    """使用手动提取特征的Informer分类器"""

    def __init__(self, feature_dim, num_classes=38,
                 d_model=256, n_heads=8, d_ff=512, depth=2,
                 factor=5, dropout_rate=0.3):
        super(FeatureInformerClassifier, self).__init__()

        # 保存参数
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.d_model = d_model

        # 特征映射层
        self.feature_mapping = nn.Linear(feature_dim, d_model)

        # Informer编码器 - 对于特征输入，我们将其视为序列长度为1的序列
        self.informer_encoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model)
        )

        # 分类头
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, feature_dim)

        # 映射到模型维度
        x = self.feature_mapping(x)  # (batch_size, d_model)

        # 通过简化的编码器 - 没有自注意力，因为这里我们处理的是单个特征向量
        x = self.informer_encoder(x)  # (batch_size, d_model)

        # 分类
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # 对于特征输入，没有真正的注意力权重，所以返回一个虚拟的权重
        dummy_attention = torch.ones(x.size(0), 1, device=x.device)

        return output, dummy_attention

    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 映射到模型维度
        x = self.feature_mapping(x)  # (batch_size, d_model)

        # 通过简化的编码器
        x = self.informer_encoder(x)  # (batch_size, d_model)

        return x  # 返回编码器输出作为潜在表示


# 轻量级CNN + Informer的模型
class LightCNNInformerClassifier(nn.Module):
    """使用轻量级CNN + Informer的分类器"""

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 filters=32, kernel_size=3, d_model=256, n_heads=8,
                 d_ff=512, depth=2, factor=5, dropout_rate=0.3):
        super(LightCNNInformerClassifier, self).__init__()

        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.d_model = d_model

        # 轻量级CNN - 只做一层卷积和池化，主要目的是降维
        self.conv = nn.Conv1d(input_channels, filters,
                              kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.MaxPool1d(4)  # 进行4倍下采样

        # 计算CNN后的序列长度
        self.cnn_seq_len = seq_length // 4

        # CNN输出映射到模型维度
        self.cnn_to_informer = nn.Linear(filters, d_model)

        # Informer编码器
        self.informer_encoder = InformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=depth,
            dropout=dropout_rate,
            factor=factor
        )

        # 全局注意力池化
        self.global_attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # 分类头
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)

        # 调整维度用于CNN
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)

        # 应用轻量级CNN
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)

        # 调整维度用于Informer
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/4, filters)

        # 映射到模型维度
        x = self.cnn_to_informer(x)  # (batch_size, seq_length/4, d_model)

        # Informer编码器
        # (batch_size, seq_length/4, d_model)
        x, attns = self.informer_encoder(x)

        # 全局注意力池化
        attention_weights = self.global_attention(
            x)  # (batch_size, seq_length/4, 1)
        # (batch_size, d_model)
        context = torch.sum(x * attention_weights, dim=1)

        # 分类
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # 返回分类输出和注意力权重
        return output, attention_weights.squeeze(2)

    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 调整维度用于CNN
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)

        # 应用轻量级CNN
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)

        # 调整维度用于Informer
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/4, filters)

        # 映射到模型维度
        x = self.cnn_to_informer(x)  # (batch_size, seq_length/4, d_model)

        # Informer编码器
        x, _ = self.informer_encoder(x)  # (batch_size, seq_length/4, d_model)

        # 全局注意力池化
        attention_weights = self.global_attention(
            x)  # (batch_size, seq_length/4, 1)
        # (batch_size, d_model)
        context = torch.sum(x * attention_weights, dim=1)

        return context  # 返回上下文向量作为潜在表示

class CNNInformerAttention(nn.Module):
    """CNN + Informer + Attention模型用于轴承故障诊断"""
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38, 
                 filters=64, kernel_size=3, informer_d_model=256, 
                 informer_n_heads=8, informer_d_ff=512, informer_depth=2,
                 informer_factor=5, dropout_rate=0.3):
        super(CNNInformerAttention, self).__init__()
        
        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.informer_d_model = informer_d_model
        
        # CNN层 - 类似于原始模型
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(filters, filters*2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters*2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(filters*2, filters*4, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(filters*4)
        
        # 计算CNN层后的序列长度
        self.informer_seq_len = seq_length // 4
        
        # 线性层用于将CNN输出转换为Informer输入维度
        self.cnn_to_informer = nn.Linear(filters*4, informer_d_model)
        
        # Informer编码器
        self.informer_encoder = InformerEncoder(
            d_model=informer_d_model,
            n_heads=informer_n_heads,
            d_ff=informer_d_ff,
            depth=informer_depth,
            dropout=dropout_rate,
            factor=informer_factor
        )
        
        # 全局注意力池化提取固定大小表示
        self.global_attention = nn.Sequential(
            nn.Linear(informer_d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类头
        self.fc1 = nn.Linear(informer_d_model, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)
        
        # 调整维度用于CNN
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)
        
        # CNN层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 调整维度用于Informer
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/4, filters*4)
        
        # 转换为Informer输入维度
        x = self.cnn_to_informer(x)  # (batch_size, seq_length/4, informer_d_model)
        
        # Informer编码器
        x, attns = self.informer_encoder(x)  # (batch_size, seq_length/4, informer_d_model)
        
        # 全局注意力池化
        attention_weights = self.global_attention(x)  # (batch_size, seq_length/4, 1)
        context = torch.sum(x * attention_weights, dim=1)  # (batch_size, informer_d_model)
        
        # 分类
        x = F.relu(self.fc1(context))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        # 返回分类输出和注意力权重
        return output, attention_weights.squeeze(2)
    
    def get_latent(self, x):
        """获取潜在表示用于可视化"""
        # 处理CNN
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 转换为Informer
        x = x.permute(0, 2, 1)
        x = self.cnn_to_informer(x)
        
        # 获取Informer表示
        x, _ = self.informer_encoder(x)
        
        # 全局注意力池化
        attention_weights = self.global_attention(x)
        context = torch.sum(x * attention_weights, dim=1)
        
        return context  # 返回上下文向量作为潜在表示
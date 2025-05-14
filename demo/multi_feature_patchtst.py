import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiFeaturePatchTST(nn.Module):
    """
    多特征融合的PatchTST模型用于轴承故障分类
    整合了原始信号处理、手工特征提取和多尺度patch分析
    """
    
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_sizes=[16, 32, 64], strides=[8, 16, 32], d_model=128, n_heads=8, 
                 num_layers=3, dropout_rate=0.1, use_fft=True, use_wavelet=True, 
                 use_time_features=True, sampling_rate=10000):
        """
        初始化多特征融合PatchTST模型
        
        参数:
        - input_channels: 输入通道数（通常是3，对应X、Y、Z轴）
        - seq_length: 序列长度（时间步数）
        - num_classes: 分类类别数
        - patch_sizes: 每个patch的长度列表，用于多尺度特征提取
        - strides: patch提取的步长列表，对应每个patch_size
        - d_model: Transformer模型的维度
        - n_heads: 多头注意力中的头数
        - num_layers: Transformer编码器层数
        - dropout_rate: Dropout比率
        - use_fft: 是否使用FFT特征
        - use_wavelet: 是否使用小波特征
        - use_time_features: 是否使用时域特征
        - sampling_rate: 采样率，用于频域特征提取
        """
        super(MultiFeaturePatchTST, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.use_fft = use_fft
        self.use_wavelet = use_wavelet
        self.use_time_features = use_time_features
        self.sampling_rate = sampling_rate
        
        # 计算每种尺度的patch数量
        self.num_patches = []
        for p_size, stride in zip(patch_sizes, strides):
            self.num_patches.append((seq_length - p_size) // stride + 1)
            
        total_patches = sum(self.num_patches)
        
        # 多尺度patch投影
        self.patch_projs = nn.ModuleList([
            nn.Linear(p_size * input_channels, d_model)
            for p_size in patch_sizes
        ])
        
        # 手工特征提取网络
        if use_time_features:
            # 时域特征: 每个轴8个特征，共24个特征
            self.time_features_proj = nn.Linear(24, d_model)
        
        if use_fft:
            # 频域特征: 每个轴7个特征，共21个特征
            self.freq_features_proj = nn.Linear(21, d_model)
            
            # 额外的FFT特征提取（前100个频率分量）
            self.fft_proj = nn.Linear(100 * input_channels * 2, d_model)
        
        if use_wavelet:
            # 小波特征: 每个轴对应level+1个子带，每个子带2个特征(能量和标准差)
            # 对于level=4的情况，每个轴有10个特征，共30个特征
            self.wavelet_features_proj = nn.Linear(30, d_model)
        
        # 特征融合层
        num_feature_types = sum([1, use_time_features, 
                                use_fft, bool(use_fft and False),  # FFT特征投影 + 原始FFT分析
                                use_wavelet])
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * num_feature_types, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
            
        # 计算最大位置编码长度
        max_patches = total_patches + num_feature_types - 1  # 减去原始patch特征，加上融合后特征
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate, max_patches)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # 残差投影
        self.res_proj = nn.Linear(d_model, d_model)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 分类头 - 更复杂的分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - outputs: 分类输出，形状为 [batch_size, num_classes]
        - attention_weights: 注意力权重，用于可视化
        - uncertainty: 不确定性估计（如果启用）
        """
        batch_size = x.size(0)
        
        # 提取多尺度patches
        patches_list = []
        for i, (patch_size, stride, proj) in enumerate(zip(self.patch_sizes, self.strides, self.patch_projs)):
            patches = self._extract_patches(x, patch_size, stride)
            patches_proj = proj(patches)
            patches_list.append(patches_proj)
        
        # 提取手工特征
        features_list = []
        
        # 1. 时域特征
        if self.use_time_features:
            time_features = self.extract_time_domain_features(x)
            time_features_proj = self.time_features_proj(time_features)
            # 添加批次维度
            time_features_proj = time_features_proj.unsqueeze(1)
            features_list.append(time_features_proj)
        
        # 2. 频域特征
        if self.use_fft:
            # 标准频域特征
            freq_features = self.extract_frequency_domain_features(x)
            freq_features_proj = self.freq_features_proj(freq_features)
            freq_features_proj = freq_features_proj.unsqueeze(1)
            features_list.append(freq_features_proj)
            
            # 额外的FFT频谱特征
            fft_features = self.extract_fft_features(x)
            fft_features_proj = self.fft_proj(fft_features)
            fft_features_proj = fft_features_proj.unsqueeze(1)
            features_list.append(fft_features_proj)
        
        # 3. 小波特征
        if self.use_wavelet:
            wavelet_features = self.extract_wavelet_features(x)
            wavelet_features_proj = self.wavelet_features_proj(wavelet_features)
            wavelet_features_proj = wavelet_features_proj.unsqueeze(1)
            features_list.append(wavelet_features_proj)
        
        # 首先处理原始patch特征
        combined_patches = torch.cat(patches_list, dim=1)
        
        # 所有特征列表，包括原始patch和手工特征
        all_features = [combined_patches] + features_list
        
        # 特征融合
        if len(features_list) > 0:
            # 手工特征融合
            fused_features = torch.cat(features_list, dim=2)
            fused_features = self.feature_fusion(fused_features)
            
            # 将融合后的特征添加到序列
            combined_features = torch.cat([combined_patches, fused_features], dim=1)
        else:
            combined_features = combined_patches
            
        # 添加位置编码
        pos_enc = self.pos_encoding(combined_features)
        
        # 残差连接和层归一化
        x_res = pos_enc + self.res_proj(combined_features)
        x_norm = self.layer_norm1(x_res)
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(x_norm)
        
        # 另一个残差连接和层归一化
        transformer_out = transformer_out + x_res
        transformer_out = self.layer_norm2(transformer_out)
        
        # 计算注意力权重，用于可视化
        attention_weights = torch.mean(transformer_out, dim=2)
        attention_expanded = self._expand_attention(attention_weights)
        
        # 全局池化
        pooled = torch.mean(transformer_out, dim=1)
        
        # 分类
        logits = self.classifier(pooled)

        return logits, attention_expanded
    
    def _extract_patches(self, x, patch_size, stride):
        """
        从输入序列中提取指定大小的patches
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        - patch_size: 单个patch的大小
        - stride: 提取patch的步长
        
        返回:
        - patches: 提取的patches，形状为 [batch_size, num_patches, patch_size * input_channels]
        """
        batch_size = x.size(0)
        patches = []
        
        for i in range(0, self.seq_length - patch_size + 1, stride):
            # 提取patch并展平通道维度
            patch = x[:, i:i+patch_size, :]  # [batch_size, patch_size, input_channels]
            patch_flat = patch.reshape(batch_size, -1)  # [batch_size, patch_size * input_channels]
            patches.append(patch_flat)
        
        # 堆叠所有patches
        return torch.stack(patches, dim=1)  # [batch_size, num_patches, patch_size * input_channels]
    
    def extract_time_domain_features(self, x):
        """
        提取时域特征，与BearingDataProcessor中的方法类似
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - features: 时域特征，形状为 [batch_size, 24]（每个轴8个特征，共3个轴）
        """
        batch_size, seq_length, channels = x.shape
        features = []
        
        for batch_idx in range(batch_size):
            batch_features = []
            
            for axis in range(channels):
                axis_data = x[batch_idx, :, axis]
                
                # 计算统计特征
                mean = torch.mean(axis_data)
                std = torch.std(axis_data)
                rms = torch.sqrt(torch.mean(torch.square(axis_data)))
                peak = torch.max(torch.abs(axis_data))
                peak_to_peak = torch.max(axis_data) - torch.min(axis_data)
                crest = peak / rms if rms != 0 else torch.tensor(0.0, device=x.device)
                
                # 计算峰度和偏度
                centered_data = axis_data - mean
                if std != 0:
                    kurtosis = torch.mean(torch.pow(centered_data / std, 4))
                    skewness = torch.mean(torch.pow(centered_data / std, 3))
                else:
                    kurtosis = torch.tensor(0.0, device=x.device)
                    skewness = torch.tensor(0.0, device=x.device)
                
                # 添加到特征列表
                batch_features.extend([mean, std, rms, peak, peak_to_peak, crest, kurtosis, skewness])
            
            features.append(batch_features)
        
        return torch.tensor(features, device=x.device)
    
    def extract_frequency_domain_features(self, x):
        """
        提取频域特征，与BearingDataProcessor中的方法类似
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - features: 频域特征，形状为 [batch_size, 21]（每个轴7个特征，共3个轴）
        """
        batch_size, seq_length, channels = x.shape
        features = []
        fs = self.sampling_rate
        
        for batch_idx in range(batch_size):
            batch_features = []
            
            for axis in range(channels):
                axis_data = x[batch_idx, :, axis].cpu().numpy()  # 转到CPU上进行FFT
                
                # 计算FFT
                spectrum = np.abs(np.fft.fft(axis_data))
                freq = np.fft.fftfreq(len(axis_data), d=1/fs)
                
                # 只保留正频率部分
                positive_freq_idx = np.where(freq > 0)[0]
                spectrum = spectrum[positive_freq_idx]
                freq = freq[positive_freq_idx]
                
                # 计算频域特征
                dominant_freq = freq[np.argmax(spectrum)]
                mean_freq = np.sum(freq * spectrum) / np.sum(spectrum) if np.sum(spectrum) != 0 else 0
                
                # 计算中值频率
                cumsum = np.cumsum(spectrum)
                median_freq_idx = np.argmax(cumsum >= np.sum(spectrum)/2)
                median_freq = freq[median_freq_idx]
                
                # 频带能量
                freq_bands = [0, 500, 1000, 2000]
                band_energy = []
                
                for i in range(len(freq_bands)-1):
                    lower = freq_bands[i]
                    upper = freq_bands[i+1]
                    band_idx = np.where((freq >= lower) & (freq < upper))[0]
                    band_energy.append(np.sum(spectrum[band_idx]))
                
                # 添加到特征列表
                batch_features.extend([dominant_freq, mean_freq, median_freq] + band_energy)
            
            features.append(batch_features)
        
        return torch.tensor(features, device=x.device)
    
    def extract_fft_features(self, x):
        """
        提取原始FFT特征，用于深度特征学习
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - fft_features: FFT特征，形状为 [batch_size, 100 * input_channels * 2]
        """
        batch_size, seq_length, channels = x.shape
        
        # 转置以适应FFT (确保在时间维度上应用FFT)
        x_cpu = x.cpu().numpy()  # 转到CPU上进行FFT
        
        # 对每个样本和每个通道进行FFT
        all_features = []
        for b in range(batch_size):
            sample_features = []
            
            for c in range(channels):
                # 应用FFT
                spectrum = np.fft.rfft(x_cpu[b, :, c])
                
                # 提取幅度和相位
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)
                
                # 选择前k个频率成分
                k = min(100, len(magnitude))
                magnitude = magnitude[:k]
                phase = phase[:k]
                
                # 合并特征
                sample_features.extend(magnitude)
                sample_features.extend(phase)
            
            all_features.append(sample_features)
        
        return torch.tensor(all_features, device=x.device)
    
    def extract_wavelet_features(self, x, wavelet='db4', level=4):
        """
        提取小波特征，与BearingDataProcessor中的方法类似
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        - wavelet: 小波类型
        - level: 分解级别
        
        返回:
        - features: 小波特征，形状为 [batch_size, 30]（每个轴10个特征，共3个轴）
        """
        import pywt
        batch_size, seq_length, channels = x.shape
        features = []
        
        for batch_idx in range(batch_size):
            batch_features = []
            
            for axis in range(channels):
                axis_data = x[batch_idx, :, axis].cpu().numpy()  # 转到CPU上进行小波分解
                
                # 小波分解
                coeffs = pywt.wavedec(axis_data, wavelet, level=level)
                
                # 计算每个子带的能量
                energies = [np.sum(np.square(coeff)) for coeff in coeffs]
                
                # 计算每个子带的标准差
                stds = [np.std(coeff) for coeff in coeffs]
                
                # 添加到特征列表
                batch_features.extend(energies + stds)
            
            features.append(batch_features)
        
        return torch.tensor(features, device=x.device)
    
    def _expand_attention(self, attention_weights):
        """
        将注意力权重扩展到与原始序列相同的长度，用于可视化
        
        参数:
        - attention_weights: 注意力权重，形状为 [batch_size, num_patches_total + num_feature_types]
        
        返回:
        - expanded: 扩展后的注意力权重，形状为 [batch_size, seq_length]
        """
        batch_size = attention_weights.size(0)
        expanded = torch.zeros(batch_size, self.seq_length, device=attention_weights.device)
        
        # 跟踪当前patch索引
        current_patch_idx = 0
        
        # 对每种patch尺度分别处理
        for scale_idx, (patch_size, stride, num_patches) in enumerate(zip(self.patch_sizes, self.strides, self.num_patches)):
            for i in range(num_patches):
                start_idx = i * stride
                end_idx = min(start_idx + patch_size, self.seq_length)
                # 将当前patch的注意力分配给对应的时间步
                expanded[:, start_idx:end_idx] += attention_weights[:, current_patch_idx].unsqueeze(1)
                current_patch_idx += 1
        
        # 忽略手工特征的注意力权重，因为它们不直接对应于时间步
        
        # 归一化
        expanded = F.normalize(expanded, p=1, dim=1)
        
        return expanded
    
    def get_latent(self, x):
        """
        获取潜在特征表示，用于t-SNE可视化等
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_length, input_channels]
        
        返回:
        - latent: 潜在特征表示，形状为 [batch_size, d_model]
        """
        batch_size = x.size(0)
        
        # 提取多尺度patches
        patches_list = []
        for i, (patch_size, stride, proj) in enumerate(zip(self.patch_sizes, self.strides, self.patch_projs)):
            patches = self._extract_patches(x, patch_size, stride)
            patches_proj = proj(patches)
            patches_list.append(patches_proj)
        
        # 提取手工特征
        features_list = []
        
        # 1. 时域特征
        if self.use_time_features:
            time_features = self.extract_time_domain_features(x)
            time_features_proj = self.time_features_proj(time_features)
            time_features_proj = time_features_proj.unsqueeze(1)
            features_list.append(time_features_proj)
        
        # 2. 频域特征
        if self.use_fft:
            # 标准频域特征
            freq_features = self.extract_frequency_domain_features(x)
            freq_features_proj = self.freq_features_proj(freq_features)
            freq_features_proj = freq_features_proj.unsqueeze(1)
            features_list.append(freq_features_proj)
            
            # 额外的FFT频谱特征
            fft_features = self.extract_fft_features(x)
            fft_features_proj = self.fft_proj(fft_features)
            fft_features_proj = fft_features_proj.unsqueeze(1)
            features_list.append(fft_features_proj)
        
        # 3. 小波特征
        if self.use_wavelet:
            wavelet_features = self.extract_wavelet_features(x)
            wavelet_features_proj = self.wavelet_features_proj(wavelet_features)
            wavelet_features_proj = wavelet_features_proj.unsqueeze(1)
            features_list.append(wavelet_features_proj)
        
        # 首先处理原始patch特征
        combined_patches = torch.cat(patches_list, dim=1)
        
        # 特征融合
        if len(features_list) > 0:
            # 手工特征融合
            fused_features = torch.cat(features_list, dim=2)
            fused_features = self.feature_fusion(fused_features)
            
            # 将融合后的特征添加到序列
            combined_features = torch.cat([combined_patches, fused_features], dim=1)
        else:
            combined_features = combined_patches
            
        # 添加位置编码
        pos_enc = self.pos_encoding(combined_features)
        
        # 残差连接和层归一化
        x_res = pos_enc + self.res_proj(combined_features)
        x_norm = self.layer_norm1(x_res)
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(x_norm)
        
        # 全局池化
        latent = torch.mean(transformer_out, dim=1)
        
        return latent


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer提供序列位置信息
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入张量
        
        参数:
        - x: 输入张量，形状为 [batch_size, seq_length, d_model]
        
        返回:
        - x: 添加位置编码后的张量，形状不变
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class MultiFeaturePatchTSTClassifier(nn.Module):
    """
    封装MultiFeaturePatchTST模型，提供与其他分类器兼容的接口
    """
    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 patch_sizes=[16, 32, 64], strides=[8, 16, 32], d_model=128, n_heads=8, 
                 num_layers=3, dropout_rate=0.1, use_fft=True, use_wavelet=True, 
                 use_time_features=True, sampling_rate=10000):
        super(MultiFeaturePatchTSTClassifier, self).__init__()
        
        self.model = MultiFeaturePatchTST(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_sizes=patch_sizes,
            strides=strides,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_fft=use_fft,
            use_wavelet=use_wavelet,
            use_time_features=use_time_features,
            sampling_rate=sampling_rate
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_latent(self, x):
        return self.model.get_latent(x)
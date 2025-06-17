import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt
import librosa
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# =================== 方案1: VAE (变分自编码器) ===================

class VAEAnomalyDetector(nn.Module):
    """基于VAE的异常检测器"""
    
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE的关键：输出均值和方差
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """编码器：输出均值和对数方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧：从N(μ,σ)采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码器"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def vae_loss(self, x, reconstructed, mu, logvar, beta=1.0):
        """VAE损失函数"""
        # 重构损失
        recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss


class VAEFaultDetector:
    """VAE故障检测器"""
    
    def __init__(self, latent_dim=64, epochs=100, batch_size=64, lr=0.001, beta=1.0):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta  # KL散度权重
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, normal_data):
        """训练VAE"""
        print(f"🤖 训练VAE异常检测器 (设备: {self.device})")
        
        # 数据预处理
        if normal_data.ndim == 3:
            normal_data = normal_data.reshape(normal_data.shape[0], -1)
        
        normal_scaled = self.scaler.fit_transform(normal_data)
        normal_tensor = torch.FloatTensor(normal_scaled).to(self.device)
        
        # 创建VAE模型
        input_dim = normal_scaled.shape[1]
        self.model = VAEAnomalyDetector(input_dim, self.latent_dim).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            
            # 随机打乱数据
            indices = torch.randperm(len(normal_tensor))
            
            for i in range(0, len(normal_tensor), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_data = normal_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # VAE前向传播
                reconstructed, mu, logvar = self.model(batch_data)
                loss = self.model.vae_loss(batch_data, reconstructed, mu, logvar, self.beta)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"   Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        # 计算阈值
        self._calculate_vae_threshold(normal_tensor)
        self.is_fitted = True
        
        print(f"✅ VAE训练完成")
        return self
    
    def _calculate_vae_threshold(self, normal_tensor):
        """计算VAE异常检测阈值"""
        self.model.eval()
        anomaly_scores = []
        
        with torch.no_grad():
            for i in range(0, len(normal_tensor), self.batch_size):
                batch = normal_tensor[i:i+self.batch_size]
                reconstructed, mu, logvar = self.model(batch)
                
                # 方法1: 重构误差
                recon_error = torch.mean((batch - reconstructed) ** 2, dim=1)
                
                # 方法2: 概率密度 (更适合VAE)
                # 计算在潜在空间中的概率密度
                z = self.model.reparameterize(mu, logvar)
                log_likelihood = -0.5 * torch.sum(z ** 2, dim=1)  # 假设先验是标准正态分布
                
                # 综合异常分数 (重构误差 - 对数似然)
                combined_score = recon_error - log_likelihood
                anomaly_scores.extend(combined_score.cpu().numpy())
        
        # 设置阈值
        self.threshold = np.percentile(anomaly_scores, 95)
        print(f"🎯 VAE阈值: {self.threshold:.6f}")
    
    def predict(self, test_data):
        """VAE异常检测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        if test_data.ndim == 3:
            test_data = test_data.reshape(test_data.shape[0], -1)
        
        test_scaled = self.scaler.transform(test_data)
        test_tensor = torch.FloatTensor(test_scaled).to(self.device)
        
        self.model.eval()
        anomaly_scores = []
        
        with torch.no_grad():
            for i in range(0, len(test_tensor), self.batch_size):
                batch = test_tensor[i:i+self.batch_size]
                reconstructed, mu, logvar = self.model(batch)
                
                # 计算综合异常分数
                recon_error = torch.mean((batch - reconstructed) ** 2, dim=1)
                z = self.model.reparameterize(mu, logvar)
                log_likelihood = -0.5 * torch.sum(z ** 2, dim=1)
                combined_score = recon_error - log_likelihood
                
                anomaly_scores.extend(combined_score.cpu().numpy())
        
        anomaly_scores = np.array(anomaly_scores)
        predictions = (anomaly_scores > self.threshold).astype(int)
        
        return predictions, anomaly_scores


# =================== 方案2: 手工特征提取器 ===================

class BearingFeatureExtractor:
    """轴承振动信号特征提取器"""
    
    def __init__(self, sampling_rate=25600):
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, signal):
        """时域特征提取"""
        features = []
        
        # 基本统计特征
        features.extend([
            np.mean(signal),           # 均值
            np.std(signal),            # 标准差
            np.var(signal),            # 方差
            np.max(signal),            # 最大值
            np.min(signal),            # 最小值
            np.ptp(signal),            # 峰峰值
            np.median(signal),         # 中位数
            np.mean(np.abs(signal)),   # 平均绝对值
            np.sqrt(np.mean(signal**2)) # RMS
        ])
        
        # 高阶统计特征
        features.extend([
            skew(signal),              # 偏度
            kurtosis(signal),          # 峭度
        ])
        
        # 波形因子和峰值因子
        rms = np.sqrt(np.mean(signal**2))
        if rms > 1e-10:
            mean_abs = np.mean(np.abs(signal))
            peak = np.max(np.abs(signal))
            features.extend([
                rms / mean_abs,        # 波形因子
                peak / rms,            # 峰值因子
                peak / mean_abs,       # 脉冲因子
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, signal):
        """频域特征提取"""
        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        power_spectrum = np.abs(fft[:len(signal)//2])**2
        freqs = freqs[:len(signal)//2]
        
        features = []
        
        # 功率谱特征
        features.extend([
            np.sum(power_spectrum),                    # 总功率
            np.mean(power_spectrum),                   # 平均功率
            np.std(power_spectrum),                    # 功率标准差
            np.max(power_spectrum),                    # 峰值功率
            np.argmax(power_spectrum),                 # 主频率索引
        ])
        
        # 频率重心和带宽
        total_power = np.sum(power_spectrum)
        if total_power > 1e-10:
            freq_centroid = np.sum(freqs * power_spectrum) / total_power
            freq_variance = np.sum(((freqs - freq_centroid) ** 2) * power_spectrum) / total_power
            features.extend([
                freq_centroid,                         # 频率重心
                np.sqrt(freq_variance),                # 频率带宽
            ])
        else:
            features.extend([0, 0])
        
        # 频段能量分布 (将频谱分为4个频段)
        n_bands = 4
        band_size = len(power_spectrum) // n_bands
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < n_bands - 1 else len(power_spectrum)
            band_energy = np.sum(power_spectrum[start_idx:end_idx])
            features.append(band_energy)
        
        return np.array(features)
    
    def extract_wavelet_features(self, signal):
        """小波域特征提取"""
        features = []
        
        # 多尺度小波分解
        wavelet = 'db4'
        levels = 4
        
        try:
            coeffs = pywt.wavedec(signal, wavelet, level=levels)
            
            # 每层小波系数的能量
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    energy = np.sum(coeff ** 2)
                    features.append(energy)
                else:
                    features.append(0)
            
            # 相对能量
            total_energy = sum(features)
            if total_energy > 1e-10:
                relative_energies = [e / total_energy for e in features]
                features.extend(relative_energies)
            else:
                features.extend([0] * len(features))
                
        except Exception as e:
            print(f"⚠️ 小波变换失败: {e}")
            features = [0] * (levels + 1) * 2  # 填充零值
        
        return np.array(features)
    
    def extract_envelope_features(self, signal):
        """包络谱特征提取 (轴承故障的重要特征)"""
        features = []
        
        try:
            # 希尔伯特变换获取包络
            analytic_signal = signal + 1j * np.imag(signal)  # 简化版
            envelope = np.abs(analytic_signal)
            
            # 包络的统计特征
            features.extend([
                np.mean(envelope),
                np.std(envelope),
                np.max(envelope),
                kurtosis(envelope),
                skew(envelope)
            ])
            
            # 包络谱
            envelope_fft = np.fft.fft(envelope)
            envelope_spectrum = np.abs(envelope_fft[:len(envelope)//2])**2
            
            features.extend([
                np.sum(envelope_spectrum),
                np.mean(envelope_spectrum),
                np.max(envelope_spectrum),
                np.argmax(envelope_spectrum)
            ])
            
        except Exception as e:
            print(f"⚠️ 包络分析失败: {e}")
            features = [0] * 9
        
        return np.array(features)
    
    def extract_all_features(self, signal):
        """提取所有特征"""
        features = []
        
        # 时域特征
        time_features = self.extract_time_domain_features(signal)
        features.extend(time_features)
        
        # 频域特征
        freq_features = self.extract_frequency_domain_features(signal)
        features.extend(freq_features)
        
        # 小波特征
        wavelet_features = self.extract_wavelet_features(signal)
        features.extend(wavelet_features)
        
        # 包络特征
        envelope_features = self.extract_envelope_features(signal)
        features.extend(envelope_features)
        
        return np.array(features)


# =================== 方案3: 特征+自编码器 ===================

class FeatureBasedAutoencoder(nn.Module):
    """基于手工特征的自编码器"""
    
    def __init__(self, feature_dim, latent_dim=32):
        super().__init__()
        
        # 由于特征维度较低，使用较小的网络
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class FeatureBasedAnomalyDetector:
    """基于手工特征的异常检测器"""
    
    def __init__(self, latent_dim=32, epochs=100, batch_size=32, lr=0.001):
        self.feature_extractor = BearingFeatureExtractor()
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features_from_signals(self, signals):
        """从信号中提取特征"""
        print("🔍 提取手工特征...")
        
        if signals.ndim == 3:
            signals = signals.reshape(signals.shape[0], -1)
        
        features_list = []
        for i, signal in enumerate(signals):
            if (i + 1) % 1000 == 0:
                print(f"   处理进度: {i+1}/{len(signals)}")
            
            features = self.feature_extractor.extract_all_features(signal)
            features_list.append(features)
        
        features_array = np.array(features_list)
        print(f"✅ 特征提取完成，特征维度: {features_array.shape[1]}")
        
        return features_array
    
    def fit(self, normal_signals):
        """训练基于特征的异常检测器"""
        print(f"🤖 训练特征自编码器 (设备: {self.device})")
        
        # 提取特征
        normal_features = self.extract_features_from_signals(normal_signals)
        
        # 标准化特征
        normal_scaled = self.scaler.fit_transform(normal_features)
        normal_tensor = torch.FloatTensor(normal_scaled).to(self.device)
        
        # 创建模型
        feature_dim = normal_scaled.shape[1]
        self.model = FeatureBasedAutoencoder(feature_dim, self.latent_dim).to(self.device)
        
        # 训练
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            
            indices = torch.randperm(len(normal_tensor))
            for i in range(0, len(normal_tensor), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_data = normal_tensor[batch_indices]
                
                optimizer.zero_grad()
                _, decoded = self.model(batch_data)
                loss = criterion(decoded, batch_data)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"   Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        # 计算阈值
        self.model.eval()
        with torch.no_grad():
            _, decoded = self.model(normal_tensor)
            reconstruction_errors = torch.mean((normal_tensor - decoded) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        self.threshold = np.percentile(reconstruction_errors, 95)
        self.is_fitted = True
        
        print(f"✅ 特征自编码器训练完成，阈值: {self.threshold:.6f}")
        return self
    
    def predict(self, test_signals):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 提取特征
        test_features = self.extract_features_from_signals(test_signals)
        test_scaled = self.scaler.transform(test_features)
        test_tensor = torch.FloatTensor(test_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            _, decoded = self.model(test_tensor)
            reconstruction_errors = torch.mean((test_tensor - decoded) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        predictions = (reconstruction_errors > self.threshold).astype(int)
        return predictions, reconstruction_errors


# =================== 集成版本 ===================

class AdvancedFaultDiagnosis:
    """高级故障诊断系统"""
    
    def __init__(self):
        self.methods = {
            'vae': VAEFaultDetector(latent_dim=64, epochs=50),
            'feature_ae': FeatureBasedAnomalyDetector(latent_dim=32, epochs=50)
        }
        
    def run_advanced_methods(self, normal_data, test_data, test_labels):
        """运行高级方法"""
        print("🚀 开始高级异常检测方法测试")
        print("=" * 80)
        
        results = {}
        
        for method_name, detector in self.methods.items():
            print(f"\n🔬 测试方法: {method_name}")
            
            try:
                # 训练
                detector.fit(normal_data)
                
                # 预测
                predictions, scores = detector.predict(test_data)
                
                # 评估 (简化版)
                from sklearn.metrics import roc_auc_score, classification_report
                auc = roc_auc_score(test_labels, scores)
                
                print(f"📊 {method_name} 结果:")
                print(f"   AUC: {auc:.4f}")
                print(classification_report(test_labels, predictions, 
                                          target_names=['正常', '故障']))
                
                results[method_name] = auc
                
            except Exception as e:
                print(f"⚠️ 方法 {method_name} 失败: {e}")
                results[method_name] = 0
        
        return results


# 使用示例
if __name__ == "__main__":
    print("🎯 高级异常检测方法演示")
    print("=" * 50)
    
    # 生成模拟数据
    np.random.seed(42)
    
    # 正常数据
    normal_data = []
    for i in range(500):
        t = np.linspace(0, 1, 1000)
        signal = 0.1 * np.sin(2 * np.pi * 10 * t) + 0.02 * np.random.randn(1000)
        normal_data.append(signal)
    
    # 故障数据
    fault_data = []
    for i in range(100):
        t = np.linspace(0, 1, 1000)
        signal = 0.1 * np.sin(2 * np.pi * 10 * t) + 0.02 * np.random.randn(1000)
        # 添加故障特征（冲击）
        signal[500:510] += 0.5 * np.random.randn(10)
        fault_data.append(signal)
    
    normal_data = np.array(normal_data)
    fault_data = np.array(fault_data)
    
    # 准备测试数据
    test_data = np.vstack([normal_data[400:], fault_data])
    test_labels = np.hstack([np.zeros(100), np.ones(100)])
    
    # 运行高级方法
    advanced_system = AdvancedFaultDiagnosis()
    results = advanced_system.run_advanced_methods(
        normal_data[:400], test_data, test_labels
    )
    
    print(f"\n🏆 高级方法结果总结:")
    for method, auc in results.items():
        print(f"  {method:15s}: AUC = {auc:.4f}")
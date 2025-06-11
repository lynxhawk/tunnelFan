import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import psutil
import warnings
warnings.filterwarnings('ignore')


class OptimizedBearingDataset(Dataset):
    """优化版轴承数据集类"""
    
    def __init__(self, data, labels, seq_length=1000, mode='signal'):
        """
        初始化数据集
        
        参数:
        - data: 输入数据
        - labels: 标签
        - seq_length: 序列长度（仅对signal模式有效）
        - mode: 'signal'表示使用原始信号，'feature'表示使用统计特征
        """
        # 预先转换为tensor，避免每次__getitem__时重复转换
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = torch.FloatTensor(data)
            
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = torch.LongTensor(labels)
            
        self.seq_length = seq_length
        self.mode = mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 直接返回tensor，避免重复转换
        return self.data[idx], self.labels[idx]


class OptimizedBearingDataProcessor:
    """优化版轴承数据处理器"""
    
    def __init__(self, data_dir='bearing_dataset', seq_length=1000, overlap_ratio=0.5):
        """
        初始化数据处理器
        
        参数:
        - data_dir: 数据集目录
        - seq_length: 序列长度
        - overlap_ratio: 窗口重叠比例
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.overlap_ratio = overlap_ratio
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # 性能监控
        self.processing_stats = {
            'load_time': 0,
            'window_time': 0,
            'feature_time': 0,
            'normalize_time': 0
        }
        
    def load_raw_data(self):
        """优化版加载原始数据"""
        import time
        start_time = time.time()
        
        all_data = []
        all_labels = []
        
        print("🚀 正在加载数据...")
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        csv_files.sort()  # 确保顺序一致
        
        # 使用更高效的数据读取
        for i, file_name in enumerate(csv_files):
            print(f"\r处理文件 {i+1}/{len(csv_files)}: {file_name}", end='', flush=True)
            file_path = os.path.join(self.data_dir, file_name)
            
            # 从文件名提取类别标签
            class_name = file_name.replace('.csv', '')
            
            # 读取CSV数据 - 优化读取参数
            try:
                # 使用更快的读取方式
                df = pd.read_csv(file_path, header=None, dtype=np.float32, 
                               engine='c')  # 使用C引擎更快
                data = df.values.astype(np.float32)
                
                # 确保数据是一维的
                if data.ndim > 1:
                    data = data.flatten()
                
                all_data.append(data)
                all_labels.append(class_name)
                
            except Exception as e:
                print(f"\n  - 错误: 无法加载文件 {file_name}: {e}")
                continue
        
        print(f"\n✅ 总共加载了 {len(all_data)} 个类别的数据")
        
        self.processing_stats['load_time'] = time.time() - start_time
        print(f"📊 数据加载耗时: {self.processing_stats['load_time']:.2f}秒")
        
        return all_data, all_labels
    
    def create_sliding_windows_vectorized(self, data, step_size=None):
        """向量化版本的滑动窗口创建 - 显著提速"""
        if step_size is None:
            step_size = int(self.seq_length * (1 - self.overlap_ratio))
        
        # 计算窗口数量
        num_windows = (len(data) - self.seq_length) // step_size + 1
        
        if num_windows <= 0:
            return np.array([])
        
        # 使用向量化操作创建所有窗口索引
        start_indices = np.arange(num_windows) * step_size
        indices = start_indices[:, np.newaxis] + np.arange(self.seq_length)
        
        # 一次性提取所有窗口
        windows = data[indices]
        
        return windows
    
    def extract_statistical_features_vectorized(self, signals):
        """向量化版本的特征提取 - 批量处理"""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        features_list = []
        
        # 批量计算时域特征
        means = np.mean(signals, axis=1)
        stds = np.std(signals, axis=1)
        vars_vals = np.var(signals, axis=1)
        maxs = np.max(signals, axis=1)
        mins = np.min(signals, axis=1)
        medians = np.median(signals, axis=1)
        ptps = np.ptp(signals, axis=1)
        mean_abs = np.mean(np.abs(signals), axis=1)
        rms = np.sqrt(np.mean(signals**2, axis=1))
        
        # 峰度和偏度（避免除零）
        kurtosis = np.zeros(len(signals))
        skewness = np.zeros(len(signals))
        
        valid_mask = stds > 1e-8
        if np.any(valid_mask):
            kurtosis[valid_mask] = np.mean(signals[valid_mask]**4, axis=1) / (stds[valid_mask]**4)
            centered = signals[valid_mask] - means[valid_mask][:, np.newaxis]
            skewness[valid_mask] = np.mean(centered**3, axis=1) / (stds[valid_mask]**3)
        
        # 频域特征（批量FFT）
        try:
            fft_signals = np.fft.fft(signals, axis=1)
            power_spectra = np.abs(fft_signals)**2
            
            fft_means = np.mean(power_spectra, axis=1)
            fft_stds = np.std(power_spectra, axis=1)
            fft_maxs = np.max(power_spectra, axis=1)
            fft_argmaxs = np.argmax(power_spectra, axis=1)
        except:
            fft_means = np.zeros(len(signals))
            fft_stds = np.zeros(len(signals))
            fft_maxs = np.zeros(len(signals))
            fft_argmaxs = np.zeros(len(signals))
        
        # 组合所有特征
        features = np.column_stack([
            means, stds, vars_vals, maxs, mins, medians, ptps, mean_abs, rms,
            kurtosis, skewness, fft_means, fft_stds, fft_maxs, fft_argmaxs
        ])
        
        return features.astype(np.float32)
    
    def prepare_signal_data(self, test_size=0.2, random_state=42):
        """优化版准备原始信号数据"""
        import time
        
        raw_data, raw_labels = self.load_raw_data()
        
        all_sequences = []
        all_sequence_labels = []
        
        print("\n🔄 正在创建序列数据...")
        window_start = time.time()
        
        for i, (data, label) in enumerate(zip(raw_data, raw_labels)):
            print(f"\r处理类别 {i+1}/{len(raw_data)}: {label[:20]}...", end='', flush=True)
            
            # 使用向量化版本创建滑动窗口
            windows = self.create_sliding_windows_vectorized(data)
            
            if len(windows) > 0:
                all_sequences.append(windows)
                all_sequence_labels.extend([label] * len(windows))
        
        print(f"\n")
        
        # 高效合并所有序列
        if all_sequences:
            X = np.vstack(all_sequences)
        else:
            raise ValueError("没有生成任何序列数据")
            
        y = np.array(all_sequence_labels)
        
        self.processing_stats['window_time'] = time.time() - window_start
        print(f"📊 窗口创建耗时: {self.processing_stats['window_time']:.2f}秒")
        
        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 批量数据标准化
        normalize_start = time.time()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # 调整维度为 (batch_size, seq_length, 1) 用于CNN
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        self.processing_stats['normalize_time'] = time.time() - normalize_start
        print(f"📊 标准化耗时: {self.processing_stats['normalize_time']:.2f}秒")
        
        print(f"\n✅ 数据准备完成:")
        print(f"  - 总序列数: {len(X_scaled):,}")
        print(f"  - 序列形状: {X_scaled.shape}")
        print(f"  - 类别数: {len(np.unique(y_encoded))}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_feature_data(self, test_size=0.2, random_state=42):
        """优化版准备特征数据"""
        import time
        
        raw_data, raw_labels = self.load_raw_data()
        
        all_features = []
        all_feature_labels = []
        
        print("\n🔬 正在提取特征...")
        feature_start = time.time()
        
        for i, (data, label) in enumerate(zip(raw_data, raw_labels)):
            print(f"\r处理类别 {i+1}/{len(raw_data)}: {label[:20]}...", end='', flush=True)
            
            # 创建滑动窗口
            windows = self.create_sliding_windows_vectorized(data)
            
            if len(windows) > 0:
                # 批量提取特征
                features = self.extract_statistical_features_vectorized(windows)
                all_features.append(features)
                all_feature_labels.extend([label] * len(features))
        
        print(f"\n")
        
        # 高效合并所有特征
        if all_features:
            X = np.vstack(all_features)
        else:
            raise ValueError("没有提取到任何特征")
            
        y = np.array(all_feature_labels)
        
        self.processing_stats['feature_time'] = time.time() - feature_start
        print(f"📊 特征提取耗时: {self.processing_stats['feature_time']:.2f}秒")
        
        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 特征标准化
        normalize_start = time.time()
        X_scaled = self.scaler.fit_transform(X)
        self.processing_stats['normalize_time'] = time.time() - normalize_start
        print(f"📊 标准化耗时: {self.processing_stats['normalize_time']:.2f}秒")
        
        print(f"\n✅ 特征数据准备完成:")
        print(f"  - 总样本数: {len(X_scaled):,}")
        print(f"  - 特征维度: {X_scaled.shape[1]}")
        print(f"  - 类别数: {len(np.unique(y_encoded))}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_optimized_data_loaders(self, mode='signal', batch_size=32, test_size=0.4, random_state=42):
        """获取优化的数据加载器"""
        
        # 自动确定最佳worker数量
        cpu_count = psutil.cpu_count()
        
        # 根据系统资源调整worker数量
        if cpu_count >= 16:
            num_workers = 8  # 高端系统
        elif cpu_count >= 8:
            num_workers = 6  # 中端系统
        elif cpu_count >= 4:
            num_workers = 4  # 普通系统
        else:
            num_workers = 2  # 低端系统
        
        # Windows系统特殊处理
        if os.name == 'nt':
            num_workers = min(num_workers, 4)  # Windows上限制worker数量
        
        # 检查可用内存
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            num_workers = min(num_workers, 2)  # 内存不足时减少worker
        
        print(f"\n⚙️  数据加载器优化配置:")
        print(f"   CPU核心数: {cpu_count}")
        print(f"   可用内存: {memory_gb:.1f}GB")
        print(f"   Workers数量: {num_workers}")
        print(f"   Pin Memory: {torch.cuda.is_available()}")
        
        # 准备数据
        if mode == 'signal':
            X_train, X_temp, y_train, y_temp = self.prepare_signal_data(test_size, random_state)
        else:
            X_train, X_temp, y_train, y_temp = self.prepare_feature_data(test_size, random_state)
        
        # 进一步分割临时数据为验证集和测试集
        val_size = 0.5
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"\n📊 数据分割结果:")
        total_samples = len(X_train) + len(X_val) + len(X_test)
        print(f"  - 训练集: {len(X_train):,} 样本 ({len(X_train)/total_samples*100:.1f}%)")
        print(f"  - 验证集: {len(X_val):,} 样本 ({len(X_val)/total_samples*100:.1f}%)")
        print(f"  - 测试集: {len(X_test):,} 样本 ({len(X_test)/total_samples*100:.1f}%)")
        
        # 创建优化的数据集
        train_dataset = OptimizedBearingDataset(X_train, y_train, self.seq_length, mode)
        val_dataset = OptimizedBearingDataset(X_val, y_val, self.seq_length, mode)
        test_dataset = OptimizedBearingDataset(X_test, y_test, self.seq_length, mode)
        
        # 创建优化的数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # GPU时启用
            persistent_workers=True if num_workers > 0 else False,  # 保持worker进程
            drop_last=False,
            prefetch_factor=4 if num_workers > 0 else 2  # 预取更多batch
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False,
            prefetch_factor=4 if num_workers > 0 else 2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False,
            prefetch_factor=4 if num_workers > 0 else 2
        )
        
        print(f"✅ 数据加载器创建完成")
        
        return train_loader, val_loader, test_loader
    
    # 保持兼容性的方法
    def get_data_loaders(self, mode='signal', batch_size=32, test_size=0.4, random_state=42):
        """兼容原接口的方法"""
        return self.get_optimized_data_loaders(mode, batch_size, test_size, random_state)
    
    def get_class_info(self):
        """获取类别信息"""
        return {
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'label_encoder': self.label_encoder
        }
    
    def print_performance_summary(self):
        """打印性能摘要"""
        total_time = sum(self.processing_stats.values())
        
        print(f"\n📈 数据处理性能摘要:")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  数据加载: {self.processing_stats['load_time']:.2f}s ({self.processing_stats['load_time']/total_time*100:.1f}%)")
        print(f"  窗口创建: {self.processing_stats['window_time']:.2f}s ({self.processing_stats['window_time']/total_time*100:.1f}%)")
        print(f"  特征提取: {self.processing_stats['feature_time']:.2f}s ({self.processing_stats['feature_time']/total_time*100:.1f}%)")
        print(f"  数据标准化: {self.processing_stats['normalize_time']:.2f}s ({self.processing_stats['normalize_time']/total_time*100:.1f}%)")


# 别名以保持兼容性
BearingDataProcessor = OptimizedBearingDataProcessor


# 测试代码
if __name__ == "__main__":
    print("🧪 测试优化版数据处理器")
    print("=" * 50)
    
    # 初始化数据处理器
    processor = OptimizedBearingDataProcessor(data_dir='bearing_dataset', seq_length=1000)
    
    print("测试信号数据加载...")
    train_loader, val_loader, test_loader = processor.get_optimized_data_loaders(
        mode='signal', batch_size=64
    )
    
    # 测试数据形状
    for batch_data, batch_labels in train_loader:
        print(f"信号数据批次形状: {batch_data.shape}")
        print(f"标签批次形状: {batch_labels.shape}")
        print(f"数据类型: {batch_data.dtype}")
        break
    
    # 显示类别信息
    class_info = processor.get_class_info()
    print(f"\n类别数量: {class_info['num_classes']}")
    print(f"前10个类别名称: {class_info['class_names'][:10]}")
    
    # 显示性能摘要
    processor.print_performance_summary()
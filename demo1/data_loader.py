import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class BearingDataset(Dataset):
    """轴承数据集类"""
    
    def __init__(self, data, labels, seq_length=1000, mode='signal'):
        """
        初始化数据集
        
        参数:
        - data: 输入数据
        - labels: 标签
        - seq_length: 序列长度（仅对signal模式有效）
        - mode: 'signal'表示使用原始信号，'feature'表示使用统计特征
        """
        self.data = data
        self.labels = labels
        self.seq_length = seq_length
        self.mode = mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == 'signal':
            # 原始信号模式：返回序列数据
            signal = self.data[idx]
            label = self.labels[idx]
            return torch.FloatTensor(signal), torch.LongTensor([label])
        else:
            # 特征模式：返回统计特征
            features = self.data[idx]
            label = self.labels[idx]
            return torch.FloatTensor(features), torch.LongTensor([label])


class BearingDataProcessor:
    """轴承数据处理器"""
    
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
        
    def load_raw_data(self):
        """加载原始数据"""
        all_data = []
        all_labels = []
        
        print("正在加载数据...")
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        csv_files.sort()  # 确保顺序一致
        
        for file_name in csv_files:
            print(f"处理文件: {file_name}")
            file_path = os.path.join(self.data_dir, file_name)
            
            # 从文件名提取类别标签
            class_name = file_name.replace('.csv', '')
            
            # 读取CSV数据
            try:
                df = pd.read_csv(file_path, header=None)
                data = df.values.astype(np.float32)
                
                # 确保数据是一维的
                if data.ndim > 1:
                    data = data.flatten()
                
                all_data.append(data)
                all_labels.append(class_name)
                
                print(f"  - 加载了 {len(data)} 个数据点")
                
            except Exception as e:
                print(f"  - 错误: 无法加载文件 {file_name}: {e}")
                continue
        
        print(f"总共加载了 {len(all_data)} 个类别的数据")
        return all_data, all_labels
    
    def create_sliding_windows(self, data, step_size=None):
        """创建滑动窗口"""
        if step_size is None:
            step_size = int(self.seq_length * (1 - self.overlap_ratio))
        
        windows = []
        for i in range(0, len(data) - self.seq_length + 1, step_size):
            window = data[i:i + self.seq_length]
            windows.append(window)
        
        return np.array(windows)
    
    def extract_statistical_features(self, signal):
        """提取统计特征"""
        features = []
        
        # 时域特征
        features.extend([
            np.mean(signal),           # 均值
            np.std(signal),            # 标准差
            np.var(signal),            # 方差
            np.max(signal),            # 最大值
            np.min(signal),            # 最小值
            np.median(signal),         # 中位数
            np.ptp(signal),            # 峰峰值
            np.mean(np.abs(signal)),   # 平均绝对值
            np.sqrt(np.mean(signal**2)), # 均方根
            np.mean(signal**4) / (np.std(signal)**4) if np.std(signal) > 0 else 0,  # 峰度
            np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3) if np.std(signal) > 0 else 0,  # 偏度
        ])
        
        # 频域特征（简化版本）
        try:
            fft_signal = np.fft.fft(signal)
            power_spectrum = np.abs(fft_signal)**2
            
            features.extend([
                np.mean(power_spectrum),      # 功率谱均值
                np.std(power_spectrum),       # 功率谱标准差
                np.max(power_spectrum),       # 功率谱峰值
                np.argmax(power_spectrum),    # 主频率
            ])
        except:
            features.extend([0, 0, 0, 0])  # 如果FFT失败，用0填充
        
        return np.array(features, dtype=np.float32)
    
    def prepare_signal_data(self, test_size=0.2, random_state=42):
        """准备原始信号数据"""
        raw_data, raw_labels = self.load_raw_data()
        
        all_sequences = []
        all_sequence_labels = []
        
        print("\n正在创建序列数据...")
        for i, (data, label) in enumerate(zip(raw_data, raw_labels)):
            print(f"处理类别 {i+1}/{len(raw_data)}: {label}")
            
            # 创建滑动窗口
            windows = self.create_sliding_windows(data)
            print(f"  - 创建了 {len(windows)} 个窗口")
            
            # 添加到总数据中
            all_sequences.extend(windows)
            all_sequence_labels.extend([label] * len(windows))
        
        # 转换为numpy数组
        X = np.array(all_sequences)
        y = np.array(all_sequence_labels)
        
        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 数据标准化
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # 调整维度为 (batch_size, seq_length, 1) 用于CNN
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        print(f"\n数据准备完成:")
        print(f"  - 总序列数: {len(X_scaled)}")
        print(f"  - 序列形状: {X_scaled.shape}")
        print(f"  - 类别数: {len(np.unique(y_encoded))}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_feature_data(self, test_size=0.2, random_state=42):
        """准备特征数据"""
        raw_data, raw_labels = self.load_raw_data()
        
        all_features = []
        all_feature_labels = []
        
        print("\n正在提取特征...")
        for i, (data, label) in enumerate(zip(raw_data, raw_labels)):
            print(f"处理类别 {i+1}/{len(raw_data)}: {label}")
            
            # 创建滑动窗口
            windows = self.create_sliding_windows(data)
            
            # 为每个窗口提取特征
            features_list = []
            for window in windows:
                features = self.extract_statistical_features(window)
                features_list.append(features)
            
            print(f"  - 提取了 {len(features_list)} 个特征向量")
            
            # 添加到总数据中
            all_features.extend(features_list)
            all_feature_labels.extend([label] * len(features_list))
        
        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(all_feature_labels)
        
        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\n特征数据准备完成:")
        print(f"  - 总样本数: {len(X_scaled)}")
        print(f"  - 特征维度: {X_scaled.shape[1]}")
        print(f"  - 类别数: {len(np.unique(y_encoded))}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_data_loaders(self, mode='signal', batch_size=32, test_size=0.4, random_state=42):
        """获取数据加载器 - 支持6:2:2分割"""
        if mode == 'signal':
            X_train, X_temp, y_train, y_temp = self.prepare_signal_data(test_size, random_state)
        else:
            X_train, X_temp, y_train, y_temp = self.prepare_feature_data(test_size, random_state)
        
        # 进一步分割临时数据为验证集和测试集 (各占总数据的20%)
        val_size = 0.5  # temp数据的一半作为验证集，一半作为测试集
        from sklearn.model_selection import train_test_split
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"\n数据分割结果:")
        print(f"  - 训练集: {len(X_train)} 样本 ({len(X_train)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
        print(f"  - 验证集: {len(X_val)} 样本 ({len(X_val)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
        print(f"  - 测试集: {len(X_test)} 样本 ({len(X_test)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
        
        # 创建数据集
        train_dataset = BearingDataset(X_train, y_train, self.seq_length, mode)
        val_dataset = BearingDataset(X_val, y_val, self.seq_length, mode)
        test_dataset = BearingDataset(X_test, y_test, self.seq_length, mode)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_info(self):
        """获取类别信息"""
        return {
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'label_encoder': self.label_encoder
        }


# 测试代码
if __name__ == "__main__":
    # 初始化数据处理器
    processor = BearingDataProcessor(data_dir='bearing_dataset', seq_length=1000)
    
    print("测试信号数据加载...")
    train_loader, test_loader = processor.get_data_loaders(mode='signal', batch_size=16)
    
    # 测试数据形状
    for batch_data, batch_labels in train_loader:
        print(f"信号数据批次形状: {batch_data.shape}")
        print(f"标签批次形状: {batch_labels.shape}")
        break
    
    print("\n测试特征数据加载...")
    train_loader_feat, test_loader_feat = processor.get_data_loaders(mode='feature', batch_size=16)
    
    # 测试特征数据形状
    for batch_data, batch_labels in train_loader_feat:
        print(f"特征数据批次形状: {batch_data.shape}")
        print(f"标签批次形状: {batch_labels.shape}")
        break
    
    # 显示类别信息
    class_info = processor.get_class_info()
    print(f"\n类别数量: {class_info['num_classes']}")
    print(f"前10个类别名称: {class_info['class_names'][:10]}")
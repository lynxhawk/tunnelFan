import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import psutil
import warnings
warnings.filterwarnings('ignore')


class BinaryBearingDataset(Dataset):
    """二分类轴承数据集类"""
    
    def __init__(self, data, labels, seq_length=1000):
        """
        初始化数据集
        
        参数:
        - data: 输入数据
        - labels: 二分类标签 (0=正常, 1=故障)
        - seq_length: 序列长度
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = torch.FloatTensor(data)
            
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = torch.LongTensor(labels)
            
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class BinaryBearingDataProcessor:
    """二分类轴承数据处理器 - 针对HUST数据集和Normal数据集"""
    
    def __init__(self, dataset1_dir='bearing_dataset', dataset2_dir='bearing_dataset1', 
                 seq_length=1000, overlap_ratio=0.5):
        """
        初始化数据处理器
        
        参数:
        - dataset1_dir: 数据集1目录 (HUST数据集，N开头=正常)
        - dataset2_dir: 数据集2目录 (包含Normal的=正常)
        - seq_length: 序列长度
        - overlap_ratio: 窗口重叠比例
        """
        self.dataset1_dir = dataset1_dir
        self.dataset2_dir = dataset2_dir
        self.seq_length = seq_length
        self.overlap_ratio = overlap_ratio
        self.scaler = StandardScaler()
        
        # 性能监控
        self.processing_stats = {
            'load_time': 0,
            'window_time': 0,
            'normalize_time': 0
        }
        
    def get_label_from_filename(self, filename, dataset_type):
        """
        根据文件名和数据集类型获取二分类标签
        
        参数:
        - filename: 文件名
        - dataset_type: 'dataset1' 或 'dataset2'
        
        返回:
        - 0: 正常, 1: 故障
        """
        if dataset_type == 'dataset1':
            # HUST数据集：N开头的是正常，其他是故障
            return 0 if filename.startswith('N') else 1
        elif dataset_type == 'dataset2':
            # 数据集2：包含Normal的是正常，其他是故障
            return 0 if 'Normal' in filename else 1
        else:
            raise ValueError("dataset_type must be 'dataset1' or 'dataset2'")
    
    def load_dataset(self, data_dir, dataset_type, max_files=None):
        """
        加载指定数据集
        
        参数:
        - data_dir: 数据目录
        - dataset_type: 'dataset1' 或 'dataset2'
        - max_files: 最大文件数限制
        """
        import time
        start_time = time.time()
        
        all_data = []
        all_labels = []
        
        print(f"🚀 正在加载{dataset_type}数据集...")
        
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        csv_files.sort()  # 确保顺序一致
        
        if max_files:
            csv_files = csv_files[:max_files]
        
        normal_count = 0
        fault_count = 0
        
        # 使用更高效的数据读取
        for i, file_name in enumerate(csv_files):
            print(f"\r处理文件 {i+1}/{len(csv_files)}: {file_name[:30]}...", end='', flush=True)
            file_path = os.path.join(data_dir, file_name)
            
            # 获取二分类标签
            label = self.get_label_from_filename(file_name, dataset_type)
            if label == 0:
                normal_count += 1
            else:
                fault_count += 1
            
            # 读取CSV数据
            try:
                # 使用更快的读取方式
                df = pd.read_csv(file_path, header=None, dtype=np.float32, engine='c')
                data = df.values.astype(np.float32)
                
                # 确保数据是一维的
                if data.ndim > 1:
                    data = data.flatten()
                
                all_data.append(data)
                all_labels.append(label)
                
            except Exception as e:
                print(f"\n  - 错误: 无法加载文件 {file_name}: {e}")
                continue
        
        print(f"\n✅ {dataset_type}数据加载完成:")
        print(f"  - 总文件数: {len(all_data)}")
        print(f"  - 正常样本: {normal_count} 个文件")
        print(f"  - 故障样本: {fault_count} 个文件")
        
        load_time = time.time() - start_time
        self.processing_stats['load_time'] += load_time
        print(f"📊 {dataset_type}加载耗时: {load_time:.2f}秒")
        
        return all_data, all_labels
    
    def create_sliding_windows_vectorized(self, data, step_size=None):
        """向量化版本的滑动窗口创建"""
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
    
    def prepare_dataset_for_training(self, dataset_type='dataset1', max_files=None):
        """
        准备指定数据集用于训练
        
        参数:
        - dataset_type: 'dataset1' 或 'dataset2'
        - max_files: 最大文件数限制
        """
        import time
        
        # 选择数据目录
        if dataset_type == 'dataset1':
            data_dir = self.dataset1_dir
        elif dataset_type == 'dataset2':
            data_dir = self.dataset2_dir
        else:
            raise ValueError("dataset_type must be 'dataset1' or 'dataset2'")
        
        # 加载原始数据
        raw_data, raw_labels = self.load_dataset(data_dir, dataset_type, max_files)
        
        all_sequences = []
        all_sequence_labels = []
        
        print(f"\n🔄 正在为{dataset_type}创建序列数据...")
        window_start = time.time()
        
        for i, (data, label) in enumerate(zip(raw_data, raw_labels)):
            print(f"\r处理文件 {i+1}/{len(raw_data)}", end='', flush=True)
            
            # 使用向量化版本创建滑动窗口
            windows = self.create_sliding_windows_vectorized(data)
            
            if len(windows) > 0:
                all_sequences.append(windows)
                # 每个窗口都对应同一个文件的标签
                all_sequence_labels.extend([label] * len(windows))
        
        print(f"\n")
        
        # 高效合并所有序列
        if all_sequences:
            X = np.vstack(all_sequences)
        else:
            raise ValueError("没有生成任何序列数据")
            
        y = np.array(all_sequence_labels)
        
        window_time = time.time() - window_start
        self.processing_stats['window_time'] += window_time
        print(f"📊 {dataset_type}窗口创建耗时: {window_time:.2f}秒")
        
        # 调整维度为 (batch_size, seq_length, 1) 用于CNN
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"✅ {dataset_type}数据准备完成:")
        print(f"  - 总序列数: {len(X):,}")
        print(f"  - 序列形状: {X.shape}")
        print(f"  - 正常序列: {np.sum(y == 0):,} ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  - 故障序列: {np.sum(y == 1):,} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        
        return X, y
    
    def get_optimized_data_loaders(self, train_dataset='dataset1', test_dataset='dataset2',
                                 batch_size=32, max_train_files=None, max_test_files=None,
                                 val_split=0.2):
        """
        获取优化的跨数据集数据加载器
        
        参数:
        - train_dataset: 训练数据集类型 ('dataset1' 或 'dataset2')
        - test_dataset: 测试数据集类型 ('dataset1' 或 'dataset2')
        - batch_size: 批次大小
        - max_train_files: 最大训练文件数
        - max_test_files: 最大测试文件数
        - val_split: 验证集比例
        """
        
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
        
        # 准备跨数据集数据
        import time
        normalize_start = time.time()
        
        print("=" * 60)
        print("🎯 跨数据集二分类训练准备")
        print("=" * 60)
        
        # 准备训练数据集
        print(f"📚 准备训练数据集 ({train_dataset})...")
        X_train_full, y_train_full = self.prepare_dataset_for_training(train_dataset, max_train_files)
        
        # 从训练集中划分出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_split, 
            random_state=42, stratify=y_train_full
        )
        
        # 准备测试数据集
        print(f"🧪 准备测试数据集 ({test_dataset})...")
        X_test, y_test = self.prepare_dataset_for_training(test_dataset, max_test_files)
        
        # 标准化处理：基于训练集拟合，应用到所有数据集
        print("\n🔧 进行数据标准化...")
        
        # 重塑数据用于标准化
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # 应用标准化
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = self.scaler.transform(X_val_reshaped).reshape(X_val.shape)
        
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = self.scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        normalize_time = time.time() - normalize_start
        self.processing_stats['normalize_time'] += normalize_time
        print(f"📊 标准化耗时: {normalize_time:.2f}秒")
        
        print(f"\n🎉 跨数据集准备完成:")
        print(f"📊 最终数据分布:")
        total_samples = len(X_train_scaled) + len(X_val_scaled) + len(X_test_scaled)
        print(f"  - 训练集: {len(X_train_scaled):,} 样本 ({len(X_train_scaled)/total_samples*100:.1f}%)")
        print(f"    * 正常: {np.sum(y_train == 0):,}, 故障: {np.sum(y_train == 1):,}")
        print(f"  - 验证集: {len(X_val_scaled):,} 样本 ({len(X_val_scaled)/total_samples*100:.1f}%)")
        print(f"    * 正常: {np.sum(y_val == 0):,}, 故障: {np.sum(y_val == 1):,}")
        print(f"  - 测试集: {len(X_test_scaled):,} 样本 ({len(X_test_scaled)/total_samples*100:.1f}%)")
        print(f"    * 正常: {np.sum(y_test == 0):,}, 故障: {np.sum(y_test == 1):,}")
        
        # 创建优化的数据集
        train_dataset_obj = BinaryBearingDataset(X_train_scaled, y_train, self.seq_length)
        val_dataset_obj = BinaryBearingDataset(X_val_scaled, y_val, self.seq_length)
        test_dataset_obj = BinaryBearingDataset(X_test_scaled, y_test, self.seq_length)
        
        # 创建优化的数据加载器
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': False
        }
        
        # 处理PyTorch版本兼容性
        try:
            dataloader_kwargs.update({
                'persistent_workers': True if num_workers > 0 else False,
                'prefetch_factor': 4 if num_workers > 0 else 2
            })
        except:
            pass  # 旧版本PyTorch不支持这些参数
        
        train_loader = DataLoader(train_dataset_obj, shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(val_dataset_obj, shuffle=False, **dataloader_kwargs)
        test_loader = DataLoader(test_dataset_obj, shuffle=False, **dataloader_kwargs)
        
        print(f"✅ 二分类数据加载器创建完成")
        
        return train_loader, val_loader, test_loader
    
    def get_class_info(self):
        """获取二分类信息"""
        return {
            'num_classes': 2,
            'class_names': ['Normal', 'Fault'],
            'class_mapping': {0: 'Normal', 1: 'Fault'}
        }
    
    def print_performance_summary(self):
        """打印性能摘要"""
        total_time = sum(self.processing_stats.values())
        
        print(f"\n📈 数据处理性能摘要:")
        print(f"  总耗时: {total_time:.2f}秒")
        if self.processing_stats['load_time'] > 0:
            print(f"  数据加载: {self.processing_stats['load_time']:.2f}s ({self.processing_stats['load_time']/total_time*100:.1f}%)")
        if self.processing_stats['window_time'] > 0:
            print(f"  窗口创建: {self.processing_stats['window_time']:.2f}s ({self.processing_stats['window_time']/total_time*100:.1f}%)")
        if self.processing_stats['normalize_time'] > 0:
            print(f"  数据标准化: {self.processing_stats['normalize_time']:.2f}s ({self.processing_stats['normalize_time']/total_time*100:.1f}%)")


# 测试代码
if __name__ == "__main__":
    # 快速测试示例
    print("🧪 测试二分类数据处理器")
    print("=" * 50)
    
    # 初始化处理器
    processor = BinaryBearingDataProcessor(
        dataset1_dir='bearing_dataset',    # HUST数据集
        dataset2_dir='bearing_dataset1',   # 另一个数据集
        seq_length=1000,
        overlap_ratio=0.5
    )
    
    try:
        # 获取数据加载器
        train_loader, val_loader, test_loader = processor.get_optimized_data_loaders(
            train_dataset='dataset1',  # 在HUST数据集上训练
            test_dataset='dataset2',   # 在dataset2上测试
            batch_size=64,
            max_train_files=10,        # 限制文件数量以加快测试
            max_test_files=5,
            val_split=0.2
        )
        
        # 测试数据形状
        print("\n🔍 数据批次信息:")
        for batch_data, batch_labels in train_loader:
            print(f"  信号数据批次形状: {batch_data.shape}")
            print(f"  标签批次形状: {batch_labels.shape}")
            print(f"  数据类型: {batch_data.dtype}")
            print(f"  标签分布: 正常={torch.sum(batch_labels == 0).item()}, 故障={torch.sum(batch_labels == 1).item()}")
            break
        
        # 显示类别信息
        class_info = processor.get_class_info()
        print(f"\n📋 分类信息:")
        print(f"  类别数量: {class_info['num_classes']}")
        print(f"  类别名称: {class_info['class_names']}")
        print(f"  类别映射: {class_info['class_mapping']}")
        
        # 显示性能摘要
        processor.print_performance_summary()
        
        print(f"\n🎉 测试完成！")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请确保数据目录存在并包含CSV文件")
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class BearingDataProcessor:
    """
    轴承数据处理类，封装了所有数据预处理功能
    """

    def __init__(self, data_dir=None, window_size=1000, overlap=0.5, sampling_rate=10000):
        """
        初始化处理器

        参数:
        - data_dir: 数据目录
        - window_size: 窗口大小
        - overlap: 重叠比例
        - sampling_rate: 采样率
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self.label_map = None

    def load_and_process_data(self, file_path):
        """
        加载并处理原始振动数据

        参数:
        - file_path: CSV文件路径

        返回:
        - 处理后的数据
        """
        # 加载数据
        data = pd.read_csv(file_path)

        # 重命名列（根据数据描述）
        if len(data.columns) >= 4:
            data.columns = ['Time Stamp', 'X-axis',
                            'Y-axis', 'Z-axis'] + list(data.columns[4:])

        # 检查缺失值
        print(
            f"文件 {os.path.basename(file_path)} 中的缺失值: {data.isnull().sum().sum()}")

        # 如果有缺失值，填充或删除
        data = data.dropna()

        return data

    def segment_data(self, data, window_size=None, overlap=None):
        """
        将数据分割成固定大小的窗口

        参数:
        - data: 原始数据DataFrame
        - window_size: 窗口大小(可选，默认使用类初始化时设置的值)
        - overlap: 重叠比例(可选，默认使用类初始化时设置的值)

        返回:
        - 分割后的数据段列表
        """
        if window_size is None:
            window_size = self.window_size
        if overlap is None:
            overlap = self.overlap

        segments = []
        step = int(window_size * (1 - overlap))

        # 提取特征列
        features = data[['X-axis', 'Y-axis', 'Z-axis']].values

        # 分割数据
        for i in range(0, len(features) - window_size + 1, step):
            segment = features[i:i + window_size]
            segments.append(segment)

        return np.array(segments)

    def create_label_mapping(self):
        """
        创建故障类型和标签的映射

        返回:
        - 标签映射字典
        """
        # 初始化映射字典
        label_map = {}

        # 健康状态 (类别0)
        label_map["healthy_with_pulley"] = 0

        # 不健康但无皮带轮 (类别1)
        label_map["healthy_without_pulley"] = 1

        # 内圈故障
        # 内圈0.7mm故障
        label_map["inner_0.7mm_100W"] = 2
        label_map["inner_0.7mm_200W"] = 3
        label_map["inner_0.7mm_300W"] = 4

        # 内圈0.9mm故障
        label_map["inner_0.9mm_100W"] = 5
        label_map["inner_0.9mm_200W"] = 6
        label_map["inner_0.9mm_300W"] = 7

        # 内圈1.1mm故障
        label_map["inner_1.1mm_100W"] = 8
        label_map["inner_1.1mm_200W"] = 9
        label_map["inner_1.1mm_300W"] = 10

        # 内圈1.3mm故障
        label_map["inner_1.3mm_100W"] = 11
        label_map["inner_1.3mm_200W"] = 12
        label_map["inner_1.3mm_300W"] = 13

        # 内圈1.5mm故障
        label_map["inner_1.5mm_100W"] = 14
        label_map["inner_1.5mm_200W"] = 15
        label_map["inner_1.5mm_300W"] = 16

        # 内圈1.7mm故障
        label_map["inner_1.7mm_100W"] = 17
        label_map["inner_1.7mm_200W"] = 18
        label_map["inner_1.7mm_300W"] = 19

        # 外圈故障
        # 外圈0.7mm故障
        label_map["outer_0.7mm_100W"] = 20
        label_map["outer_0.7mm_200W"] = 21
        label_map["outer_0.7mm_300W"] = 22

        # 外圈0.9mm故障
        label_map["outer_0.9mm_100W"] = 23
        label_map["outer_0.9mm_200W"] = 24
        label_map["outer_0.9mm_300W"] = 25

        # 外圈1.1mm故障
        label_map["outer_1.1mm_100W"] = 26
        label_map["outer_1.1mm_200W"] = 27
        label_map["outer_1.1mm_300W"] = 28

        # 外圈1.3mm故障
        label_map["outer_1.3mm_100W"] = 29
        label_map["outer_1.3mm_200W"] = 30
        label_map["outer_1.3mm_300W"] = 31

        # 外圈1.5mm故障
        label_map["outer_1.5mm_100W"] = 32
        label_map["outer_1.5mm_200W"] = 33
        label_map["outer_1.5mm_300W"] = 34

        # 外圈1.7mm故障
        label_map["outer_1.7mm_100W"] = 35
        label_map["outer_1.7mm_200W"] = 36
        label_map["outer_1.7mm_300W"] = 37

        return label_map

    def parse_filename(self, filename):
        """
        从文件名解析故障类型、严重程度和负载信息

        参数:
        - filename: 文件名

        返回:
        - 解析后的标签
        """
        # 提取故障类型和严重程度
        if 'healthy with pulley' in filename.lower():
            return "healthy_with_pulley"
        elif 'healthy without pulley' in filename.lower():
            return "healthy_without_pulley"
        else:
            # 提取故障类型（内圈/外圈）
            if 'inner' in filename.lower():
                fault_type = 'inner'
            elif 'outer' in filename.lower():
                fault_type = 'outer'
            else:
                fault_type = 'unknown'

            # 提取严重程度（如0.7mm）
            severity_match = re.search(r'(\d+\.\d+)', filename)
            severity = severity_match.group(1) if severity_match else 'unknown'

            # 提取负载信息
            if '100watt' in filename.lower():
                load = '100W'
            elif '200watt' in filename.lower():
                load = '200W'
            elif '300watt' in filename.lower():
                load = '300W'
            else:
                load = 'unknown'

            # 组合成标签
            label = f"{fault_type}_{severity}mm_{load}"

            print(f"文件 {filename} 解析为标签: {label}")  # 添加调试信息
            return label

    def extract_time_domain_features(self, segment):
        """
        提取时域特征

        参数:
        - segment: 单个数据段

        返回:
        - 时域特征向量
        """
        features = []

        for axis in range(segment.shape[1]):  # 遍历X、Y、Z轴
            axis_data = segment[:, axis]

            # 计算统计特征
            mean = np.mean(axis_data)
            std = np.std(axis_data)
            rms = np.sqrt(np.mean(np.square(axis_data)))
            peak = np.max(np.abs(axis_data))
            peak_to_peak = np.max(axis_data) - np.min(axis_data)
            crest = peak / rms if rms != 0 else 0
            kurtosis = np.sum((axis_data - mean)**4) / \
                (len(axis_data) * std**4) if std != 0 else 0
            skewness = np.sum((axis_data - mean)**3) / \
                (len(axis_data) * std**3) if std != 0 else 0

            # 添加到特征列表
            features.extend(
                [mean, std, rms, peak, peak_to_peak, crest, kurtosis, skewness])

        return np.array(features)

    def extract_frequency_domain_features(self, segment, fs=None):
        """
        提取频域特征

        参数:
        - segment: 单个数据段
        - fs: 采样频率（可选，默认使用类初始化时设置的值）

        返回:
        - 频域特征向量
        """
        if fs is None:
            fs = self.sampling_rate

        features = []

        for axis in range(segment.shape[1]):  # 遍历X、Y、Z轴
            axis_data = segment[:, axis]

            # 计算FFT
            spectrum = np.abs(fft(axis_data))
            freq = np.fft.fftfreq(len(axis_data), d=1/fs)

            # 只保留正频率部分
            positive_freq_idx = np.where(freq > 0)[0]
            spectrum = spectrum[positive_freq_idx]
            freq = freq[positive_freq_idx]

            # 计算频域特征
            dominant_freq = freq[np.argmax(spectrum)]
            mean_freq = np.sum(freq * spectrum) / \
                np.sum(spectrum) if np.sum(spectrum) != 0 else 0
            median_freq = freq[np.argmax(
                np.cumsum(spectrum) >= np.sum(spectrum)/2)]

            # 频带能量
            freq_bands = [0, 500, 1000, 2000, 5000]
            band_energy = []

            for i in range(len(freq_bands)-1):
                lower = freq_bands[i]
                upper = freq_bands[i+1]
                band_idx = np.where((freq >= lower) & (freq < upper))[0]
                band_energy.append(np.sum(spectrum[band_idx]))

            # 添加到特征列表
            features.extend([dominant_freq, mean_freq,
                            median_freq] + band_energy)

        return np.array(features)

    def extract_wavelet_features(self, segment, wavelet='db4', level=4):
        """
        提取小波特征

        参数:
        - segment: 单个数据段
        - wavelet: 小波类型
        - level: 分解级别

        返回:
        - 小波特征向量
        """
        features = []

        for axis in range(segment.shape[1]):  # 遍历X、Y、Z轴
            axis_data = segment[:, axis]

            # 小波分解
            coeffs = pywt.wavedec(axis_data, wavelet, level=level)

            # 计算每个子带的能量
            energies = [np.sum(np.square(coeff)) for coeff in coeffs]

            # 计算每个子带的标准差
            stds = [np.std(coeff) for coeff in coeffs]

            # 添加到特征列表
            features.extend(energies + stds)

        return np.array(features)

    def data_augmentation(self, segments, noise_level=0.05, time_shift_max=50):
        """
        数据增强

        参数:
        - segments: 原始数据段
        - noise_level: 添加噪声的级别
        - time_shift_max: 最大时间偏移

        返回:
        - 增强后的数据
        """
        augmented_segments = []

        for segment in segments:
            # 原始数据
            augmented_segments.append(segment)

            # 添加噪声
            noise = np.random.normal(
                0, noise_level * np.std(segment), segment.shape)
            augmented_segments.append(segment + noise)

            # 时间偏移
            shift = np.random.randint(-time_shift_max, time_shift_max)
            shifted = np.roll(segment, shift, axis=0)
            augmented_segments.append(shifted)

            # 振幅缩放
            scale = np.random.uniform(0.9, 1.1)
            augmented_segments.append(segment * scale)

        return np.array(augmented_segments)

    def prepare_dataset(self, augment=False, use_features=False):
        """
        准备完整数据集

        参数:
        - augment: 是否应用数据增强
        - use_features: 是否提取手工特征（True）或使用原始信号（False）

        返回:
        - X: 特征数据
        - y: 标签（类别索引）
        - label_map: 标签映射
        """
        if self.data_dir is None:
            raise ValueError("数据目录未设置，请在初始化时指定data_dir或调用set_data_dir()。")

        # 创建标签映射 - 确保它总是被定义
        if self.label_map is None:
            self.label_map = self.create_label_mapping()

        X = []
        y = []

        # 获取所有CSV文件（包括子目录中的）
        csv_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        print(f"找到 {len(csv_files)} 个CSV文件")

        for file_path in csv_files:
            filename = os.path.basename(file_path)

            # 解析文件名获取标签
            label_str = self.parse_filename(filename)

            # 添加这一行：
            print(f"文件名 '{filename}' 解析得到的 label_str: '{label_str}'")

            # 如果标签在映射中存在
            if label_str in self.label_map:
                label = self.label_map[label_str]

                try:
                    # 加载数据
                    data = self.load_and_process_data(file_path)

                    # 分割数据
                    segments = self.segment_data(data)

                    # 数据增强（如果启用）
                    if augment:
                        segments = self.data_augmentation(segments)

                    # 特征提取（如果启用）
                    if use_features:
                        features = []
                        for segment in segments:
                            # 提取时域特征
                            time_features = self.extract_time_domain_features(
                                segment)

                            # 提取频域特征
                            freq_features = self.extract_frequency_domain_features(
                                segment)

                            # 提取小波特征
                            wavelet_features = self.extract_wavelet_features(
                                segment)

                            # 组合所有特征
                            combined_features = np.concatenate(
                                [time_features, freq_features, wavelet_features])
                            features.append(combined_features)

                        X.extend(features)
                    else:
                        # 使用原始信号
                        X.extend(segments)

                    # 添加标签
                    y.extend([label] * len(segments))

                    print(
                        f"处理文件: {filename}, 标签: {label_str}, 添加了 {len(segments)} 个样本")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")

        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)

        return X, y, self.label_map

    def normalize_data(self, X_train, X_val=None, X_test=None):
        """
        归一化数据

        参数:
        - X_train: 训练数据
        - X_val: 验证数据（可选）
        - X_test: 测试数据（可选）

        返回:
        - 归一化后的数据
        """
        # 对训练数据进行拟合和变换
        if X_train.ndim == 3:  # 原始信号数据 (samples, time_steps, channels)
            # 重塑为2D进行归一化
            n_samples, n_timesteps, n_channels = X_train.shape
            X_train_reshaped = X_train.reshape(
                n_samples, n_timesteps * n_channels)

            # 拟合和变换
            X_train_normalized = self.scaler.fit_transform(X_train_reshaped)

            # 重塑回3D
            X_train_normalized = X_train_normalized.reshape(
                n_samples, n_timesteps, n_channels)

            # 变换验证和测试数据（如果提供）
            results = [X_train_normalized]

            if X_val is not None:
                n_samples_val = X_val.shape[0]
                X_val_reshaped = X_val.reshape(
                    n_samples_val, n_timesteps * n_channels)
                X_val_normalized = self.scaler.transform(X_val_reshaped)
                X_val_normalized = X_val_normalized.reshape(
                    n_samples_val, n_timesteps, n_channels)
                results.append(X_val_normalized)

            if X_test is not None:
                n_samples_test = X_test.shape[0]
                X_test_reshaped = X_test.reshape(
                    n_samples_test, n_timesteps * n_channels)
                X_test_normalized = self.scaler.transform(X_test_reshaped)
                X_test_normalized = X_test_normalized.reshape(
                    n_samples_test, n_timesteps, n_channels)
                results.append(X_test_normalized)
        else:  # 特征数据 (samples, features)
            # 直接拟合和变换
            X_train_normalized = self.scaler.fit_transform(X_train)

            # 变换验证和测试数据（如果提供）
            results = [X_train_normalized]

            if X_val is not None:
                X_val_normalized = self.scaler.transform(X_val)
                results.append(X_val_normalized)

            if X_test is not None:
                X_test_normalized = self.scaler.transform(X_test)
                results.append(X_test_normalized)

        return tuple(results)

    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """分割数据为训练、验证和测试集"""
        print(f"进入split_data函数，X形状: {X.shape}, y形状: {y.shape}")
        print(f"标签分布: {np.bincount(y.astype(int))}")

        try:
            # 首先分离出测试集
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # 计算验证集占训练+验证集的比例
            val_ratio = val_size / (1 - test_size)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
            )

            print(
                f"分割后 - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            print(f"分割数据时出错: {str(e)}")
            # 返回原始数据的简单分割，不使用stratify
            total = len(X)
            train_end = int(total * (1 - test_size - val_size))
            val_end = int(total * (1 - test_size))

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            X_test = X[val_end:]
            y_test = y[val_end:]

            print(
                f"简单分割 - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

            return X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """
        创建PyTorch数据加载器

        参数:
        - X_train, X_val, X_test: 特征数据
        - y_train, y_val, y_test: 标签
        - batch_size: 批次大小

        返回:
        - PyTorch数据加载器
        """
        # 创建PyTorch数据集
        train_dataset = BearingDataset(X_train, y_train)
        val_dataset = BearingDataset(X_val, y_val)
        test_dataset = BearingDataset(X_test, y_test)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def set_data_dir(self, data_dir):
        """
        设置数据目录

        参数:
        - data_dir: 数据目录
        """
        self.data_dir = data_dir

    def visualize_data(self, file_path=None, data=None, title="振动数据可视化"):
        """
        可视化振动数据

        参数:
        - file_path: CSV文件路径（可选，如果提供则从文件加载数据）
        - data: 数据DataFrame（可选，如果提供则直接使用）
        - title: 图表标题
        """
        if file_path is not None:
            data = self.load_and_process_data(file_path)
        elif data is None:
            raise ValueError("必须提供file_path或data参数")

        plt.figure(figsize=(15, 10))

        # 绘制X轴振动
        plt.subplot(3, 1, 1)
        plt.plot(data['Time Stamp'].values[:1000],
                 data['X-axis'].values[:1000])
        plt.title(f"{title} - X轴")
        plt.xlabel("时间")
        plt.ylabel("振幅")

        # 绘制Y轴振动
        plt.subplot(3, 1, 2)
        plt.plot(data['Time Stamp'].values[:1000],
                 data['Y-axis'].values[:1000])
        plt.title(f"{title} - Y轴")
        plt.xlabel("时间")
        plt.ylabel("振幅")

        # 绘制Z轴振动
        plt.subplot(3, 1, 3)
        plt.plot(data['Time Stamp'].values[:1000],
                 data['Z-axis'].values[:1000])
        plt.title(f"{title} - Z轴")
        plt.xlabel("时间")
        plt.ylabel("振幅")

        plt.tight_layout()
        plt.show()

    def visualize_frequency_spectrum(self, file_path=None, data=None, fs=None, title="频谱分析"):
        """
        可视化频谱

        参数:
        - file_path: CSV文件路径（可选，如果提供则从文件加载数据）
        - data: 数据DataFrame（可选，如果提供则直接使用）
        - fs: 采样频率（可选，默认使用类初始化时设置的值）
        - title: 图表标题
        """
        if fs is None:
            fs = self.sampling_rate

        if file_path is not None:
            data = self.load_and_process_data(file_path)
        elif data is None:
            raise ValueError("必须提供file_path或data参数")

        plt.figure(figsize=(15, 10))

        # 计算X轴频谱
        x_spectrum = np.abs(fft(data['X-axis'].values[:1000]))
        freqs = np.fft.fftfreq(1000, d=1/fs)

        # 只保留正频率部分
        positive_freq_idx = np.where(freqs > 0)[0]
        x_spectrum = x_spectrum[positive_freq_idx]
        freqs = freqs[positive_freq_idx]

        # 绘制X轴频谱
        plt.subplot(3, 1, 1)
        plt.plot(freqs, x_spectrum)
        plt.title(f"{title} - X轴")
        plt.xlabel("频率 (Hz)")
        plt.ylabel("幅度")

        # 计算Y轴频谱
        y_spectrum = np.abs(fft(data['Y-axis'].values[:1000]))
        y_spectrum = y_spectrum[positive_freq_idx]

        # 绘制Y轴频谱
        plt.subplot(3, 1, 2)
        plt.plot(freqs, y_spectrum)
        plt.title(f"{title} - Y轴")
        plt.xlabel("频率 (Hz)")
        plt.ylabel("幅度")

        # 计算Z轴频谱
        z_spectrum = np.abs(fft(data['Z-axis'].values[:1000]))
        z_spectrum = z_spectrum[positive_freq_idx]

        # 绘制Z轴频谱
        plt.subplot(3, 1, 3)
        plt.plot(freqs, z_spectrum)
        plt.title(f"{title} - Z轴")
        plt.xlabel("频率 (Hz)")
        plt.ylabel("幅度")

        plt.tight_layout()
        plt.show()

    def visualize_class_distribution(self, y, label_map=None):
        """
        可视化类别分布（支持浮点型标签）

        参数:
        - y: 标签数组（可以是浮点型）
        - label_map: 标签映射（可选）
        """
        if label_map is None:
            label_map = self.label_map

        # 使用pandas的value_counts而不是bincount
        import pandas as pd
        unique_labels = pd.Series(y).value_counts().sort_index()

        class_names = []
        if label_map is None:
            # 如果没有标签映射，只显示类别索引
            class_names = [f"Class {i}" for i in unique_labels.index]
        else:
            # 如果有标签映射，显示实际类别名称
            reverse_map = {v: k for k, v in label_map.items()}
            class_names = [reverse_map.get(
                float(i), f"Class {i}") for i in unique_labels.index]

        plt.figure(figsize=(15, 8))
        plt.bar(range(len(unique_labels)), unique_labels.values)
        plt.xticks(range(len(unique_labels)), class_names, rotation=90)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Sample Count')
        plt.tight_layout()
        plt.show()
        print("可视化结束")


class BearingDataset(Dataset):
    """
    轴承数据集类，用于PyTorch数据加载器
    """

    def __init__(self, X, y):
        """
        初始化数据集

        参数:
        - X: 特征数据
        - y: 标签
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项

        参数:
        - idx: 索引

        返回:
        - 数据项和标签
        """
        return self.X[idx], self.y[idx]


# 使用示例
if __name__ == "__main__":
    # 创建数据处理器
    processor = BearingDataProcessor(
        data_dir="bearing_data",  # 数据目录
        window_size=1000,         # 窗口大小
        overlap=0.5,              # 重叠比例
        sampling_rate=10000       # 采样率
    )

    # 示例：加载并可视化单个文件
    """
    file_path = "0.7inner100watt67V2Iv.csv"
    data = processor.load_and_process_data(file_path)
    processor.visualize_data(data=data, title="轴承振动数据")
    processor.visualize_frequency_spectrum(data=data)
    """

    # 示例：准备完整数据集
    """
    # 使用原始信号
    X, y, label_map = processor.prepare_dataset(augment=False, use_features=False)
    
    # 或使用提取的特征
    # X, y, label_map = processor.prepare_dataset(augment=False, use_features=True)
    
    # 查看类别分布
    processor.visualize_class_distribution(y, label_map)
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # 归一化数据
    X_train_norm, X_val_norm, X_test_norm = processor.normalize_data(X_train, X_val, X_test)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_dataloaders(
        X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, batch_size=32
    )
    """

    print("PyTorch数据处理模块示例完成")

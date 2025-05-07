import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import os
import time


class SVMClassifier:
    """
    SVM分类器，用于轴承故障诊断

    支持原始信号或特征提取后的数据
    """

    def __init__(self, input_shape, num_classes, kernel='rbf', C=1.0, gamma='scale',
                 probability=True, class_weight='balanced', random_state=42, svm_model_path=None):
        """
        初始化SVM分类器

        参数:
        - input_shape: 输入特征维度 (batch_size, features) 或 (batch_size, seq_length, channels)
        - num_classes: 分类类别数
        - kernel: 核函数，默认为'rbf'
        - C: 正则化参数，默认为1.0
        - gamma: 'rbf'核函数的参数，默认为'scale'
        - probability: 是否启用概率估计，默认为True
        - class_weight: 类别权重，默认为'balanced'
        - random_state: 随机种子，默认为42
        - svm_model_path: 预训练模型路径，如果提供则加载模型
        """
        self.num_classes = num_classes

        # 处理输入形状
        if len(input_shape) == 3:  # 原始信号数据
            self.input_dim = input_shape[1] * \
                input_shape[2]  # seq_length * channels
            self.seq_length = input_shape[1]
            self.channels = input_shape[2]
            self.is_raw_signal = True
        else:  # 特征数据
            self.input_dim = input_shape[1]
            self.is_raw_signal = False

        # 创建SVM模型
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            class_weight=class_weight,
            random_state=random_state,
            verbose=True
        )

        self.is_trained = False

        # 如果提供了模型路径，加载模型
        if svm_model_path and os.path.exists(svm_model_path):
            self.model = load(svm_model_path)
            self.is_trained = True

    def to(self, device):
        """
        模拟PyTorch的to()方法，用于兼容PyTorch模型接口

        参数:
        - device: 计算设备

        返回:
        - self: 返回自身以支持链式调用
        """
        # SVM是CPU模型，不需要移动到GPU
        # 这个方法只是为了兼容PyTorch模型接口
        return self

    def __preprocess_data(self, x):
        """
        预处理输入数据

        参数:
        - x: 输入数据

        返回:
        - 预处理后的数据
        """
        # 如果输入是PyTorch张量，转换为NumPy数组
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # 如果是原始信号数据，展平
        if self.is_raw_signal and len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)

        return x

    def fit(self, train_loader, val_loader=None):
        """
        训练SVM模型

        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器(可选)

        返回:
        - 训练历史
        """
        # 从数据加载器收集所有训练数据
        X_train = []
        y_train = []

        for data, target in train_loader:
            X_batch = self.__preprocess_data(data)
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

            X_train.append(X_batch)
            y_train.append(target)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        # 记录训练开始时间
        start_time = time.time()

        # 训练模型
        print(f"开始训练SVM模型，训练样本数量: {X_train.shape[0]}")
        self.model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"SVM模型训练完成，用时: {training_time:.2f} 秒")

        # 评估训练集性能
        train_accuracy = self.model.score(X_train, y_train)
        print(f"训练集准确率: {train_accuracy:.4f}")

        # 如果提供了验证集，评估验证集性能
        val_accuracy = None
        if val_loader:
            X_val = []
            y_val = []

            for data, target in val_loader:
                X_batch = self.__preprocess_data(data)
                if isinstance(target, torch.Tensor):
                    target = target.cpu().numpy()

                X_val.append(X_batch)
                y_val.append(target)

            X_val = np.vstack(X_val)
            y_val = np.concatenate(y_val)

            val_accuracy = self.model.score(X_val, y_val)
            print(f"验证集准确率: {val_accuracy:.4f}")

        self.is_trained = True

        # 创建类似于神经网络模型的历史记录
        history = {
            'train_acc': [train_accuracy],
            'val_acc': [val_accuracy] if val_accuracy is not None else None
        }

        return history

    def grid_search(self, train_loader, param_grid=None):
        """
        使用网格搜索找到最佳参数

        参数:
        - train_loader: 训练数据加载器
        - param_grid: 参数网格，默认为None

        返回:
        - 最佳参数
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'kernel': ['rbf', 'linear', 'poly']
            }

        # 从数据加载器收集所有训练数据
        X_train = []
        y_train = []

        for data, target in train_loader:
            X_batch = self.__preprocess_data(data)
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

            X_train.append(X_batch)
            y_train.append(target)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        # 创建网格搜索对象
        grid_search = GridSearchCV(
            SVC(probability=True, class_weight='balanced'),
            param_grid,
            cv=5,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1  # 使用所有可用CPU核心
        )

        # 执行网格搜索
        print("开始执行网格搜索...")
        grid_search.fit(X_train, y_train)

        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

        # 更新模型为最佳模型
        self.model = grid_search.best_estimator_
        self.is_trained = True

        return grid_search.best_params_

    def forward(self, x):
        """
        前向传播（预测）

        参数:
        - x: 输入数据

        返回:
        - logits: 类别预测的概率（模拟logits）
        - dummy_attention: 虚拟注意力权重（为了兼容其他模型）
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit()方法")

        # 预处理数据
        X = self.__preprocess_data(x)

        # 获取决策函数值（非概率）
        decision_values = self.model.decision_function(X)

        # 对于二分类问题，将其转换为与多类问题相同的形状
        if self.num_classes == 2:
            decision_values = np.column_stack(
                [-decision_values, decision_values])

        # 创建虚拟注意力权重
        if self.is_raw_signal:
            dummy_attention = np.ones(
                (X.shape[0], self.seq_length)) / self.seq_length
        else:
            dummy_attention = np.ones((X.shape[0], 10)) / 10  # 任意长度10

        # 转换为PyTorch张量
        logits = torch.FloatTensor(decision_values)
        dummy_attention = torch.FloatTensor(dummy_attention)

        # 如果输入是CUDA张量，将输出也移到CUDA
        if isinstance(x, torch.Tensor) and x.is_cuda:
            logits = logits.to(x.device)
            dummy_attention = dummy_attention.to(x.device)

        return logits, dummy_attention

    def eval(self):
        """
        设置模型为评估模式（兼容PyTorch模型接口）

        返回:
        - self: 返回自身以支持链式调用
        """
        # SVM没有训练/评估模式的区别
        # 这个方法只是为了兼容PyTorch模型接口
        return self

    def train(self):
        """
        设置模型为训练模式（兼容PyTorch模型接口）

        返回:
        - self: 返回自身以支持链式调用
        """
        # SVM没有训练/评估模式的区别
        # 这个方法只是为了兼容PyTorch模型接口
        return self

    def parameters(self):
        """
        返回模型参数（兼容PyTorch模型接口）

        返回:
        - 空列表: SVM没有可训练的参数
        """
        # SVM没有可训练的参数
        # 这个方法只是为了兼容PyTorch模型接口
        return []

    def get_latent(self, x):
        """
        获取特征嵌入表示（兼容PyTorch模型接口）

        参数:
        - x: 输入数据

        返回:
        - 支持向量与样本的距离
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit()方法")

        # 预处理数据
        X = self.__preprocess_data(x)

        # 计算样本到支持向量的距离
        n_samples = X.shape[0]
        sv_count = self.model.support_vectors_.shape[0]

        # 初始化距离矩阵
        distances = np.zeros((n_samples, sv_count))

        # 对于每个样本和每个支持向量，计算距离
        for i in range(n_samples):
            for j in range(sv_count):
                if self.model.kernel == 'linear':
                    # 对于线性核，使用点积作为特征
                    distances[i, j] = np.dot(
                        X[i], self.model.support_vectors_[j])
                else:
                    # 对于其他核，使用欧几里得距离
                    distances[i, j] = np.linalg.norm(
                        X[i] - self.model.support_vectors_[j])

        # 转换为PyTorch张量
        distances_tensor = torch.FloatTensor(distances)

        # 如果输入是CUDA张量，将输出也移到CUDA
        if isinstance(x, torch.Tensor) and x.is_cuda:
            distances_tensor = distances_tensor.to(x.device)

        return distances_tensor

    def save(self, filepath):
        """
        保存模型

        参数:
        - filepath: 保存路径
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法保存")

        dump(self.model, filepath)
        print(f"SVM模型已保存至 {filepath}")

    def load(self, filepath):
        """
        加载模型

        参数:
        - filepath: 模型路径

        返回:
        - self: 返回自身以支持链式调用
        """
        self.model = load(filepath)
        self.is_trained = True
        print(f"SVM模型已从 {filepath} 加载")
        return self


# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 16
    seq_length = 1000
    input_channels = 3
    num_classes = 38

    # 创建随机数据
    X_raw = np.random.randn(batch_size, seq_length, input_channels)
    X_features = np.random.randn(batch_size, 100)  # 假设特征维度为100
    y = np.random.randint(0, num_classes, size=batch_size)

    # 测试原始信号输入
    print("测试原始信号输入:")
    svm_raw = SVMClassifier(
        (batch_size, seq_length, input_channels), num_classes)

    # 模拟数据加载器
    class DummyDataLoader:
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size

        def __iter__(self):
            for i in range(0, len(self.X), self.batch_size):
                yield self.X[i:i+self.batch_size], self.y[i:i+self.batch_size]

    # 创建数据加载器
    train_loader_raw = DummyDataLoader(X_raw, y, batch_size)

    # 测试fit方法（只训练一个小批次用于演示）
    print("测试SVM拟合（原始信号）:")
    history = svm_raw.fit(train_loader_raw)
    print(f"训练历史: {history}")

    # 测试forward方法
    print("测试SVM前向传播（原始信号）:")
    logits, attention = svm_raw.forward(X_raw)
    print(f"Logits形状: {logits.shape}, 注意力权重形状: {attention.shape}")

    # 测试获取潜在表示
    print("测试SVM获取潜在表示（原始信号）:")
    latent = svm_raw.get_latent(X_raw)
    print(f"潜在表示形状: {latent.shape}")

    # 测试特征输入
    print("\n测试特征输入:")
    svm_features = SVMClassifier(
        (batch_size, X_features.shape[1]), num_classes)

    # 创建数据加载器
    train_loader_features = DummyDataLoader(X_features, y, batch_size)

    # 测试fit方法（只训练一个小批次用于演示）
    print("测试SVM拟合（特征）:")
    history = svm_features.fit(train_loader_features)
    print(f"训练历史: {history}")

    # 测试forward方法
    print("测试SVM前向传播（特征）:")
    logits, attention = svm_features.forward(X_features)
    print(f"Logits形状: {logits.shape}, 注意力权重形状: {attention.shape}")

    # 测试获取潜在表示
    print("测试SVM获取潜在表示（特征）:")
    latent = svm_features.get_latent(X_features)
    print(f"潜在表示形状: {latent.shape}")

    # 测试保存和加载
    print("\n测试保存和加载:")
    svm_features.save("test_svm_model.joblib")
    svm_loaded = SVMClassifier((batch_size, X_features.shape[1]), num_classes)
    svm_loaded.load("test_svm_model.joblib")

    # 测试加载后的模型
    print("测试加载后的模型:")
    logits, attention = svm_loaded.forward(X_features)
    print(f"Logits形状: {logits.shape}, 注意力权重形状: {attention.shape}")

    # 删除测试文件
    import os
    os.remove("test_svm_model.joblib")

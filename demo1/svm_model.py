import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import time
from tqdm import tqdm


class SVMClassifier(nn.Module):
    """
    SVM分类器，用于轴承故障诊断
    继承自nn.Module以便与主训练脚本兼容

    支持原始信号或特征提取后的数据
    """

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38, 
                 kernel='rbf', C=1.0, gamma='scale', probability=True, 
                 class_weight='balanced', random_state=42, use_scaler=True,
                 dropout_rate=0.0, **kwargs):
        """
        初始化SVM分类器

        参数:
        - input_channels: 输入通道数
        - seq_length: 序列长度
        - num_classes: 分类类别数
        - kernel: 核函数，默认为'rbf'
        - C: 正则化参数，默认为1.0
        - gamma: 'rbf'核函数的参数，默认为'scale'
        - probability: 是否启用概率估计，默认为True
        - class_weight: 类别权重，默认为'balanced'
        - random_state: 随机种子，默认为42
        - use_scaler: 是否使用数据标准化，默认为True
        - dropout_rate: 为兼容性保留，SVM不使用dropout
        """
        super(SVMClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.use_scaler = use_scaler
        
        # 计算输入维度 (展平后的维度)
        self.input_dim = seq_length * input_channels

        # 创建SVM模型
        self.svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            class_weight=class_weight,
            random_state=random_state,
            verbose=False  # 减少输出
        )

        # 数据标准化器
        if self.use_scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.is_trained = False
        self._device = torch.device('cpu')  # SVM默认使用CPU

    def to(self, device):
        """
        模拟PyTorch的to()方法，用于兼容PyTorch模型接口
        """
        # SVM是CPU模型，但保存设备信息以便处理张量
        self._device = device
        return super().to('cpu')  # SVM始终在CPU上运行

    def _preprocess_data(self, x):
        """
        预处理输入数据

        参数:
        - x: 输入数据 [batch_size, seq_length, input_channels]

        返回:
        - 预处理后的数据 [batch_size, input_dim]
        """
        # 如果输入是PyTorch张量，转换为NumPy数组
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # 展平信号数据: [batch_size, seq_length, channels] -> [batch_size, seq_length * channels]
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        
        return x

    def _collect_data_from_loader(self, data_loader):
        """
        从数据加载器收集所有数据

        参数:
        - data_loader: PyTorch数据加载器

        返回:
        - X: 特征数据
        - y: 标签数据
        """
        X_list = []
        y_list = []

        print("正在收集数据...")
        for data, target in tqdm(data_loader, desc="收集数据"):
            X_batch = self._preprocess_data(data)
            
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy().squeeze()
            
            X_list.append(X_batch)
            y_list.append(target)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        return X, y

    def fit_data_loaders(self, train_loader, val_loader=None):
        """
        使用数据加载器训练SVM模型
        兼容主训练脚本的接口

        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器(可选)

        返回:
        - 训练历史 (兼容主脚本格式)
        """
        # 收集训练数据
        X_train, y_train = self._collect_data_from_loader(train_loader)
        
        # 数据标准化
        if self.use_scaler:
            print("正在标准化数据...")
            X_train = self.scaler.fit_transform(X_train)

        # 记录训练开始时间
        start_time = time.time()

        # 训练模型
        print(f"开始训练SVM模型，训练样本数量: {X_train.shape[0]}")
        print(f"特征维度: {X_train.shape[1]}")
        
        self.svm_model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"SVM模型训练完成，用时: {training_time:.2f} 秒")

        # 评估训练集性能
        train_accuracy = self.svm_model.score(X_train, y_train)
        print(f"训练集准确率: {train_accuracy:.4f}")

        # 如果提供了验证集，评估验证集性能
        val_accuracy = None
        if val_loader:
            X_val, y_val = self._collect_data_from_loader(val_loader)
            
            if self.use_scaler:
                X_val = self.scaler.transform(X_val)

            val_accuracy = self.svm_model.score(X_val, y_val)
            print(f"验证集准确率: {val_accuracy:.4f}")

        self.is_trained = True

        # 创建类似于神经网络模型的历史记录
        history = {
            'train_loss': [0.0],  # SVM没有损失概念，设为0
            'train_acc': [train_accuracy],
            'val_loss': [0.0] if val_accuracy is not None else [],
            'val_acc': [val_accuracy] if val_accuracy is not None else []
        }

        return history

    def forward(self, x):
        """
        前向传播（预测）
        兼容PyTorch模型接口

        参数:
        - x: 输入数据 [batch_size, seq_length, input_channels]

        返回:
        - logits: 类别预测的决策函数值
        - dummy_attention: 虚拟注意力权重（为了兼容其他模型）
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit_data_loaders()方法")

        # 预处理数据
        X = self._preprocess_data(x)
        
        # 数据标准化
        if self.use_scaler and self.scaler is not None:
            X = self.scaler.transform(X)

        # 获取决策函数值（类似于logits）
        if hasattr(self.svm_model, 'decision_function'):
            decision_values = self.svm_model.decision_function(X)
            
            # 对于二分类问题，将其转换为与多类问题相同的形状
            if self.num_classes == 2 and len(decision_values.shape) == 1:
                decision_values = np.column_stack([-decision_values, decision_values])
        else:
            # 如果没有decision_function，使用predict_proba
            if hasattr(self.svm_model, 'predict_proba'):
                decision_values = self.svm_model.predict_proba(X)
            else:
                # 最后备选方案：使用预测结果创建one-hot
                predictions = self.svm_model.predict(X)
                decision_values = np.eye(self.num_classes)[predictions]

        # 创建虚拟注意力权重
        batch_size = X.shape[0]
        dummy_attention = np.ones((batch_size, self.seq_length // 4)) / (self.seq_length // 4)

        # 转换为PyTorch张量
        logits = torch.FloatTensor(decision_values)
        dummy_attention = torch.FloatTensor(dummy_attention)

        # 如果指定了设备，将张量移到该设备
        logits = logits.to(self._device)
        dummy_attention = dummy_attention.to(self._device)

        return logits, dummy_attention

    def eval(self):
        """
        设置模型为评估模式（兼容PyTorch模型接口）
        """
        # SVM没有训练/评估模式的区别
        return self

    def train(self, mode=True):
        """
        设置模型为训练模式（兼容PyTorch模型接口）
        """
        # SVM没有训练/评估模式的区别
        return self

    def get_latent(self, x):
        """
        获取特征嵌入表示（兼容主脚本的t-SNE可视化）

        参数:
        - x: 输入数据

        返回:
        - 预处理后的特征表示
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit_data_loaders()方法")

        # 预处理数据
        X = self._preprocess_data(x)
        
        # 数据标准化
        if self.use_scaler and self.scaler is not None:
            X = self.scaler.transform(X)

        # 对于SVM，我们返回标准化后的原始特征作为"潜在表示"
        # 或者可以计算到支持向量的距离
        if hasattr(self.svm_model, 'support_vectors_'):
            # 计算到前64个支持向量的距离作为特征表示
            sv_count = min(64, self.svm_model.support_vectors_.shape[0])
            distances = np.zeros((X.shape[0], sv_count))
            
            for i in range(sv_count):
                distances[:, i] = np.linalg.norm(
                    X - self.svm_model.support_vectors_[i], axis=1)
            
            features = distances
        else:
            # 如果支持向量不可用，使用降维后的原始特征
            # 取前128维特征
            features = X[:, :min(128, X.shape[1])]

        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(features)
        features_tensor = features_tensor.to(self._device)

        return features_tensor

    def grid_search(self, train_loader, param_grid=None, cv=3):
        """
        使用网格搜索找到最佳参数

        参数:
        - train_loader: 训练数据加载器
        - param_grid: 参数网格，默认为None
        - cv: 交叉验证折数

        返回:
        - 最佳参数
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            }

        # 收集数据
        X_train, y_train = self._collect_data_from_loader(train_loader)
        
        # 数据标准化
        if self.use_scaler:
            X_train = self.scaler.fit_transform(X_train)

        # 创建网格搜索对象
        grid_search = GridSearchCV(
            SVC(probability=True, class_weight='balanced'),
            param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1  # 使用所有可用CPU核心
        )

        # 执行网格搜索
        print("开始执行网格搜索...")
        grid_search.fit(X_train, y_train)

        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

        # 更新模型为最佳模型
        self.svm_model = grid_search.best_estimator_
        self.is_trained = True

        return grid_search.best_params_

    def save_model(self, filepath):
        """
        保存模型 (兼容主脚本接口)

        参数:
        - filepath: 保存路径
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法保存")

        save_dict = {
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'input_channels': self.input_channels,
            'seq_length': self.seq_length,
            'num_classes': self.num_classes,
            'use_scaler': self.use_scaler
        }
        
        dump(save_dict, filepath)
        print(f"SVM模型已保存至 {filepath}")

    def load_model(self, filepath):
        """
        加载模型 (兼容主脚本接口)

        参数:
        - filepath: 模型路径
        """
        save_dict = load(filepath)
        
        self.svm_model = save_dict['svm_model']
        self.scaler = save_dict.get('scaler', None)
        self.input_channels = save_dict['input_channels']
        self.seq_length = save_dict['seq_length']
        self.num_classes = save_dict['num_classes']
        self.use_scaler = save_dict.get('use_scaler', True)
        
        self.is_trained = True
        print(f"SVM模型已从 {filepath} 加载")
        return self


# Factory function for compatibility with main script
def create_svm_model(input_channels, seq_length, num_classes, **kwargs):
    """
    Factory function to create SVM model
    Compatible with main script's create_model function
    """
    return SVMClassifier(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        **kwargs
    )


# Wrapper class for enhanced compatibility
class SVMTrainer:
    """
    SVM训练器包装类，提供与主脚本BearingClassificationTrainer类似的接口
    """
    
    def __init__(self, model, device, save_dir='checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader, **kwargs):
        """
        训练SVM模型
        
        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - **kwargs: 其他参数（为兼容性保留）
        
        返回:
        - 训练历史
        """
        print("开始SVM训练...")
        
        # 使用模型的fit_data_loaders方法训练
        history = self.model.fit_data_loaders(train_loader, val_loader)
        
        # 更新训练历史
        self.training_history['train_loss'].extend(history['train_loss'])
        self.training_history['train_acc'].extend(history['train_acc'])
        self.training_history['val_loss'].extend(history['val_loss'])
        self.training_history['val_acc'].extend(history['val_acc'])
        
        # 更新最佳准确率
        if history['val_acc'] and len(history['val_acc']) > 0:
            self.best_accuracy = history['val_acc'][0]
        elif history['train_acc'] and len(history['train_acc']) > 0:
            self.best_accuracy = history['train_acc'][0]
            
        # 保存模型
        model_path = os.path.join(self.save_dir, f'svm_best_model_acc_{self.best_accuracy:.4f}.joblib')
        self.model.save_model(model_path)
        
        print(f"SVM训练完成，最佳准确率: {self.best_accuracy:.4f}")
        
        return self.training_history


if __name__ == "__main__":
    # 测试代码
    print("测试SVM模型...")
    
    # 测试参数
    batch_size = 32
    seq_length = 1000
    input_channels = 3
    num_classes = 10  # 使用较小的类别数进行快速测试
    
    # 创建随机测试数据
    X = torch.randn(batch_size * 3, seq_length, input_channels)
    y = torch.randint(0, num_classes, (batch_size * 3,))
    
    # 创建简单的数据加载器
    class SimpleDataLoader:
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            
        def __iter__(self):
            for i in range(0, len(self.X), self.batch_size):
                end_idx = min(i + self.batch_size, len(self.X))
                yield self.X[i:end_idx], self.y[i:end_idx]
                
        def __len__(self):
            return (len(self.X) + self.batch_size - 1) // self.batch_size
    
    # 分割数据
    train_size = batch_size * 2
    train_X, val_X = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    train_loader = SimpleDataLoader(train_X, train_y, batch_size)
    val_loader = SimpleDataLoader(val_X, val_y, batch_size)
    
    # 测试工厂函数
    print("测试工厂函数...")
    model = create_svm_model(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        C=1.0,
        kernel='rbf',
        use_scaler=True
    )
    
    print(f"模型创建成功: {type(model)}")
    print(f"输入维度: {model.input_dim}")
    print(f"类别数: {model.num_classes}")
    
    # 测试训练
    print("\n测试模型训练...")
    history = model.fit_data_loaders(train_loader, val_loader)
    print(f"训练历史: {history}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    test_x = torch.randn(8, seq_length, input_channels)
    model.eval()
    
    with torch.no_grad():
        logits, attention = model(test_x)
        print(f"输入形状: {test_x.shape}")
        print(f"输出形状: {logits.shape}")
        print(f"注意力权重形状: {attention.shape}")
    
    # 测试潜在表示
    print("\n测试潜在表示...")
    latent = model.get_latent(test_x)
    print(f"潜在表示形状: {latent.shape}")
    
    # 测试保存和加载
    print("\n测试保存和加载...")
    save_path = "test_svm_model.joblib"
    model.save_model(save_path)
    
    # 创建新模型并加载
    model2 = create_svm_model(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes
    )
    model2.load_model(save_path)
    
    # 测试加载后的模型
    with torch.no_grad():
        logits2, attention2 = model2(test_x)
        print(f"加载后模型输出形状: {logits2.shape}")
    
    # 清理测试文件
    if os.path.exists(save_path):
        os.remove(save_path)
    
    # 测试训练器包装类
    print("\n测试SVM训练器...")
    trainer_model = create_svm_model(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        C=0.1  # 使用较小的C值以加快训练
    )
    
    trainer = SVMTrainer(trainer_model, 'cpu', 'test_checkpoints')
    training_history = trainer.train(train_loader, val_loader)
    print(f"训练器训练历史: {training_history}")
    
    # 清理测试目录
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
    
    print("\nSVM模型测试完成！")
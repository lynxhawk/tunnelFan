import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class SelfAttention(nn.Module):
    """自定义自注意力层"""

    def __init__(self, input_dim, attention_dim=64):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim

        # 注意力层的权重矩阵
        self.W = nn.Linear(input_dim, attention_dim)
        self.u = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # 计算注意力得分
        u = torch.tanh(self.W(x))  # (batch_size, seq_len, attention_dim)
        scores = self.u(u)  # (batch_size, seq_len, 1)

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(
            scores, dim=1)  # (batch_size, seq_len, 1)

        # 计算加权和
        # (batch_size, input_dim)
        context = torch.sum(x * attention_weights, dim=1)

        return context, attention_weights.squeeze(-1)  # 返回上下文向量和注意力权重


class CNNBiGRU_Attention(nn.Module):
    """CNN + BiGRU + 自注意力机制模型"""

    def __init__(self, input_channels=3, seq_length=1000, num_classes=38,
                 filters=64, kernel_size=3, gru_hidden=100, dropout_rate=0.3):
        super(CNNBiGRU_Attention, self).__init__()

        # 保存参数
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.gru_hidden = gru_hidden

        # CNN层 - 第一层
        self.conv1 = nn.Conv1d(input_channels, filters,
                               kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.pool1 = nn.MaxPool1d(2)

        # CNN层 - 第二层
        self.conv2 = nn.Conv1d(
            filters, filters*2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters*2)
        self.pool2 = nn.MaxPool1d(2)

        # CNN层 - 第三层
        self.conv3 = nn.Conv1d(filters*2, filters*4,
                               kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(filters*4)

        # 计算GRU输入序列长度（经过池化后）
        self.gru_seq_len = seq_length // 4

        # 双向GRU层 - 替换原来的LSTM
        self.gru = nn.GRU(filters*4, gru_hidden,
                          batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 自注意力层
        # 因为是双向GRU，所以维度是gru_hidden*2
        self.attention = SelfAttention(gru_hidden*2)

        # 全连接层
        self.fc1 = nn.Linear(gru_hidden*2, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_channels)

        # 调整输入维度以适应Conv1d
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)

        # CNN层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        # 调整维度以适应GRU
        x = x.permute(0, 2, 1)  # (batch_size, seq_length/4, filters*4)

        # GRU层 - 替换原来的LSTM
        x, _ = self.gru(x)
        x = self.dropout1(x)

        # 自注意力层
        context, attention_weights = self.attention(x)

        # 全连接层
        x = F.relu(self.fc1(context))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, attention_weights


class BearingDataset(Dataset):
    """轴承数据集"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=50, patience=10, scheduler=None):
    """
    训练模型

    参数:
    - model: 创建的模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - device: 设备(CPU/GPU)
    - num_epochs: 训练轮数
    - patience: 早停耐心值
    - scheduler: 学习率调度器(可选)

    返回:
    - 训练好的模型和训练历史
    """
    # 将模型移至指定设备
    model = model.to(device)

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # 早停参数
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    # 计算总批次数，用于显示进度
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)

    # 记录开始时间
    import time
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 打印epoch开始信息
        print(f"\nEpoch {epoch+1}/{num_epochs} 开始...")

        # 使用tqdm创建进度条（如果已安装）
        try:
            from tqdm import tqdm
            train_loader = tqdm(train_loader, desc=f"训练中 (Epoch {epoch+1})",
                                leave=False, unit="batch")
        except ImportError:
            print("提示: 安装tqdm包可以获得更好的进度显示效果: pip install tqdm")

        # 训练循环
        batch_count = 0
        for inputs, labels in train_loader:
            batch_count += 1
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 如果没有使用tqdm，手动打印进度
            if 'tqdm' not in globals():
                if batch_count % 10 == 0 or batch_count == total_train_batches:
                    progress = 100. * batch_count / total_train_batches
                    print(f"\r训练进度: [{batch_count}/{total_train_batches}] {progress:.1f}% " +
                          f"- 当前损失: {loss.item():.4f}", end="")

        # 计算训练指标
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print("\n开始验证...")

        # 使用tqdm创建进度条（如果已安装）
        try:
            from tqdm import tqdm
            val_loader = tqdm(val_loader, desc=f"验证中 (Epoch {epoch+1})",
                              leave=False, unit="batch")
        except ImportError:
            pass

        batch_count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                batch_count += 1
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # 如果没有使用tqdm，手动打印进度
                if 'tqdm' not in globals():
                    if batch_count % 5 == 0 or batch_count == total_val_batches:
                        progress = 100. * batch_count / total_val_batches
                        print(
                            f"\r验证进度: [{batch_count}/{total_val_batches}] {progress:.1f}%", end="")

        # 计算验证指标
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # 更新学习率（如果提供了调度器）
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # 记录历史
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # 计算耗时
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        # 打印进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs} | " +
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | " +
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | " +
              f"LR: {current_lr:.6f} | " +
              f"时间: {epoch_time:.1f}秒 | 总耗时: {total_time/60:.1f}分钟")

        # 检查早停
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"模型性能提升！保存最佳模型 (验证损失: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"模型性能未提升 ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"触发早停条件，在第 {epoch+1} 轮停止训练")
                break

    # 打印总结
    print(f"\n训练完成！总耗时: {(time.time() - start_time)/60:.1f}分钟")
    print(f"最佳验证损失: {best_val_loss:.4f}")

    # 加载最佳模型权重
    model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(model, test_loader, criterion, device, class_names=None):
    """
    评估模型

    参数:
    - model: 训练好的模型
    - test_loader: 测试数据加载器
    - criterion: 损失函数
    - device: 设备(CPU/GPU)
    - class_names: 类别名称列表(可选)

    返回:
    - 测试损失、准确率和混淆矩阵
    """
    # 将模型移至指定设备并设为评估模式
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # 统计
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # 保存预测结果和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算测试指标
    test_loss /= test_total
    test_acc = test_correct / test_total

    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 找出数据中实际存在的类别
    unique_labels = np.unique(np.concatenate([all_labels, all_predictions]))

    # 打印分类报告
    if class_names:
        # 只使用数据中实际存在的类别对应的名称
        used_class_names = [class_names[i] if i < len(class_names) else f"Class {i}"
                            for i in unique_labels]

        print("\nClassification Report:")
        try:
            # 尝试使用指定的标签名称
            print(classification_report(all_labels, all_predictions,
                                        labels=unique_labels,
                                        target_names=used_class_names))
        except ValueError as e:
            print(f"Error using provided class names: {e}")
            print("Falling back to numeric labels...")
            print(classification_report(all_labels, all_predictions))
    else:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))

    return test_loss, test_acc, cm, np.array(all_predictions), np.array(all_labels)


def visualize_learning_curves(history):
    """
    可视化学习曲线

    参数:
    - history: 训练历史
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.ylabel('loss rate')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves_pytorch.png', dpi=300)
    plt.show()


def visualize_confusion_matrix(cm, class_names=None):
    """
    可视化混淆矩阵

    参数:
    - cm: 混淆矩阵
    - class_names: 类别名称(可选)
    """
    import seaborn as sns

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.xlabel('prediction')
    plt.ylabel('ground truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_pytorch.png', dpi=300)
    plt.show()


def visualize_attention_weights(model, test_loader, device, num_samples=3):
    """
    可视化注意力权重

    参数:
    - model: 训练好的模型
    - test_loader: 测试数据加载器
    - device: 设备(CPU/GPU)
    - num_samples: 要可视化的样本数量
    """
    # 将模型移至指定设备并设为评估模式
    model = model.to(device)
    model.eval()

    # 获取一些样本
    inputs_list = []
    attention_weights_list = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            if len(inputs_list) >= num_samples:
                break

            inputs = inputs.to(device)

            # 获取注意力权重
            _, attention_weights = model(inputs)

            # 保存样本和注意力权重
            inputs_list.append(inputs.cpu().numpy())
            attention_weights_list.append(attention_weights.cpu().numpy())

    # 可视化
    plt.figure(figsize=(15, 10))

    for i in range(num_samples):
        # 获取一个样本和其注意力权重
        sample = inputs_list[i][0]  # 第一个批次中的第一个样本
        attention = attention_weights_list[i][0]  # 对应的注意力权重

        # 计算原始信号的峰值，用于绘图标准化
        signal_peak = np.max(np.abs(sample))

        # 绘制子图
        plt.subplot(num_samples, 1, i+1)

        # 绘制X轴信号
        plt.plot(sample[:, 0], 'b-', label='X axis signal', alpha=0.7)

        # 绘制Y轴信号
        plt.plot(sample[:, 1], 'g-', label='Y axis signal', alpha=0.7)

        # 绘制Z轴信号
        plt.plot(sample[:, 2], 'r-', label='Z axis signal', alpha=0.7)

        # 绘制注意力权重（根据原始信号的峰值进行缩放）
        # 注意：需要上采样注意力权重以匹配原始信号长度
        # 假设attention是GRU之后的注意力，长度可能是原始信号的1/4
        upsampled_attention = np.repeat(attention, 4)[:len(sample)]
        plt.plot(upsampled_attention * signal_peak, 'k-',
                 label='Attention Weight', linewidth=2)

        plt.title(f'sample {i+1}')
        plt.xlabel('step')
        plt.ylabel('amplitude')
        plt.legend()

    plt.tight_layout()
    plt.savefig('attention_visualization_pytorch.png', dpi=300)
    plt.show()


def save_model(model, filepath):
    """
    保存模型

    参数:
    - model: 训练好的模型
    - filepath: 保存路径
    """
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存至 {filepath}")


def load_model(model, filepath, device):
    """
    加载模型

    参数:
    - model: 未训练的模型实例
    - filepath: 模型权重路径
    - device: 设备(CPU/GPU)

    返回:
    - 加载了权重的模型
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# 使用示例
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    model = CNNBiGRU_Attention(
        input_channels=3,      # X, Y, Z轴
        seq_length=1000,       # 时间步长
        num_classes=38,        # 类别数量
        filters=64,            # CNN滤波器数量
        kernel_size=3,         # 卷积核大小
        gru_hidden=100,        # GRU隐藏单元数量 (替换原来的lstm_hidden)
        dropout_rate=0.3       # Dropout比率
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 假设我们已经有了处理好的数据
    """
    # 创建数据加载器
    train_dataset = BearingDataset(X_train, y_train)
    val_dataset = BearingDataset(X_val, y_val)
    test_dataset = BearingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 训练模型
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=50, patience=10, scheduler=scheduler
    )
    
    # 可视化学习曲线
    visualize_learning_curves(history)
    
    # 评估模型
    test_loss, test_acc, cm, all_predictions, all_labels = evaluate_model(
        model, test_loader, criterion, device, class_names=None
    )
    
    # 可视化混淆矩阵
    visualize_confusion_matrix(cm)
    
    # 可视化注意力权重
    visualize_attention_weights(model, test_loader, device)
    
    # 保存模型
    save_model(model, 'bearing_fault_model_pytorch.pth')
    """

    print("PyTorch模型结构示例已完成")

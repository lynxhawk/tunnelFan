import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import json
import time

# 导入自定义模块
from pytorch_cnn_bilstm_attention import (
    CNNLSTM_Attention, train_model, evaluate_model,
    visualize_learning_curves, visualize_confusion_matrix,
    visualize_attention_weights, save_model, load_model
)
from pytorch_data_processing import BearingDataProcessor

# 导入新添加的模型
from mlp_model import MLPClassifier
from cnn_attention import CNNClassifier  # 带注意力机制的CNN
from cnn_model import SimpleCNNClassifier  # 不带注意力机制的简单CNN
from svm_model import SVMClassifier  # 支持向量机分类器
from pytorch_cnn_bigru_attention import CNNBiGRU_Attention
from informer_models import (
    DirectInformerClassifier, LightCNNInformerClassifier,
    FeatureInformerClassifier, CNNInformerAttention
)


def parse_arguments():
    """
    解析命令行参数

    返回:
    - 解析后的参数
    """
    parser = argparse.ArgumentParser(description='轴承故障诊断系统')

    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='bearing_data',
                        help='数据目录路径')
    parser.add_argument('--window_size', type=int, default=1000,
                        help='信号窗口大小')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='窗口重叠比例')
    parser.add_argument('--use_features', action='store_true',
                        help='是否使用手工提取的特征（默认使用原始信号）')
    parser.add_argument('--augment', action='store_true',
                        help='是否使用数据增强')

    # 模型相关参数
    parser.add_argument('--model_type', type=str,
                        choices=['cnn_lstm_attention', 'cnn_bigru_attention',
                                 'cnn_attention', 'cnn_model', 'mlp', 'svm',
                                 'direct_informer', 'light_cnn_informer', 'feature_informer',  # 添加新的模型类型
                                 'cnn_informer_attention'],
                        default='cnn_lstm_attention',
                        help='模型类型：CNN-LSTM-Attention/CNN-BiGRU-Attention/CNN(带注意力)/CNN简单版/MLP/SVM/直接Informer/轻量CNN-Informer/特征-Informer/CNN-Informer-Attention')
    parser.add_argument('--filters', type=int, default=64,
                        help='CNN滤波器数量')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='CNN卷积核大小')
    parser.add_argument('--lstm_hidden', type=int, default=100,
                        help='LSTM隐藏单元数量')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比率')

    # 添加Informer特定参数
    parser.add_argument('--informer_d_model', type=int, default=256,
                        help='Informer模型维度')
    parser.add_argument('--informer_n_heads', type=int, default=8,
                        help='Informer中的注意力头数量')
    parser.add_argument('--informer_d_ff', type=int, default=512,
                        help='Informer中前馈网络的维度')
    parser.add_argument('--informer_depth', type=int, default=2,
                        help='Informer编码器的层数')
    parser.add_argument('--informer_factor', type=int, default=5,
                        help='Informer中ProbSparse注意力的因子')

    # SVM特定参数
    parser.add_argument('--svm_kernel', type=str, default='rbf',
                        choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help='SVM核函数类型')
    parser.add_argument('--svm_C', type=float, default=1.0,
                        help='SVM正则化参数C')
    parser.add_argument('--svm_gamma', type=str, default='scale',
                        help='SVM RBF核参数gamma')
    parser.add_argument('--svm_grid_search', action='store_true',
                        help='是否对SVM执行网格搜索以寻找最佳参数')

    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')

    # 系统功能
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'],
                        default='train', help='系统模式：训练/测试/预测')
    parser.add_argument('--model_path', type=str, default='bearing_model.pth',
                        help='模型保存/加载路径')
    parser.add_argument('--predict_file', type=str,
                        help='要预测的文件路径（仅预测模式）')

    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()

    # 检查CUDA可用性
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建数据处理器
    processor = BearingDataProcessor(
        data_dir=args.data_dir,
        window_size=args.window_size,
        overlap=args.overlap,
        sampling_rate=10000  # 假设采样率为10kHz
    )

    # 根据模式执行不同功能
    if args.mode == 'train':
        train_workflow(processor, args, device)
    elif args.mode == 'test':
        test_workflow(processor, args, device)
    elif args.mode == 'predict':
        predict_workflow(processor, args, device)


def create_model(model_type, input_shape, num_classes, args, device):
    """
    根据指定的类型创建模型

    参数:
    - model_type: 模型类型
    - input_shape: 输入数据形状（特征数据或原始信号）
    - num_classes: 类别数量
    - args: 命令行参数
    - device: 计算设备

    返回:
    - 创建的模型
    """
    if args.use_features:
        # 对于特征数据
        if model_type not in ['mlp', 'svm', 'feature_informer']:  # 添加feature_informer
            print(f"警告: 对于特征数据，'{model_type}'模型不适用，自动切换为'mlp'。")
            model_type = 'mlp'

        input_dim = input_shape[1]  # 特征维度

        if model_type == 'feature_informer':  # 新增的特征+Informer模型
            model = FeatureInformerClassifier(
                feature_dim=input_dim,
                num_classes=num_classes,
                d_model=args.informer_d_model,
                n_heads=args.informer_n_heads,
                d_ff=args.informer_d_ff,
                depth=args.informer_depth,
                factor=args.informer_factor,
                dropout_rate=args.dropout
            )    
        elif model_type == 'mlp':
            model = MLPClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_layers=[256, 128, 64],
                dropout_rate=args.dropout
            )
        elif model_type == 'svm':
            model = SVMClassifier(
                input_shape=input_shape,
                num_classes=num_classes,
                kernel=args.svm_kernel,
                C=args.svm_C,
                gamma=args.svm_gamma
            )
    else:
        # 对于原始信号数据
        input_channels = input_shape[2]  # 通常是3（X, Y, Z轴）
        seq_length = input_shape[1]      # 窗口大小

        if model_type == 'direct_informer':  # 新增的直接Informer模型
            model = DirectInformerClassifier(
                input_channels=input_channels,
                seq_length=seq_length,
                num_classes=num_classes,
                d_model=args.informer_d_model,
                n_heads=args.informer_n_heads,
                d_ff=args.informer_d_ff,
                depth=args.informer_depth,
                factor=args.informer_factor,
                dropout_rate=args.dropout
            )
        elif model_type == 'light_cnn_informer':  # 新增的轻量级CNN+Informer模型
            model = LightCNNInformerClassifier(
                input_channels=input_channels,
                seq_length=seq_length,
                num_classes=num_classes,
                filters=args.filters // 2,  # 使用较少的滤波器
                kernel_size=args.kernel_size,
                d_model=args.informer_d_model,
                n_heads=args.informer_n_heads,
                d_ff=args.informer_d_ff,
                depth=args.informer_depth,
                factor=args.informer_factor,
                dropout_rate=args.dropout
            )
        elif model_type == 'cnn_informer_attention':  # 原始的CNN+Informer模型
            model = CNNInformerAttention(
                input_channels=input_channels,
                seq_length=seq_length,
                num_classes=num_classes,
                filters=args.filters,
                kernel_size=args.kernel_size,
                informer_d_model=args.informer_d_model,
                informer_n_heads=args.informer_n_heads,
                informer_d_ff=args.informer_d_ff,
                informer_depth=args.informer_depth,
                informer_factor=args.informer_factor,
                dropout_rate=args.dropout
            )
        elif model_type == 'cnn_lstm_attention':
            model = CNNLSTM_Attention(
                input_channels=input_channels,
                seq_length=seq_length,
                num_classes=num_classes,
                filters=args.filters,
                kernel_size=args.kernel_size,
                lstm_hidden=args.lstm_hidden,
                dropout_rate=args.dropout
            )
        elif model_type == 'cnn_bigru_attention':
            model = CNNBiGRU_Attention(
                input_channels=input_channels,
                seq_length=seq_length,
                num_classes=num_classes,
                filters=args.filters,
                kernel_size=args.kernel_size,
                gru_hidden=args.lstm_hidden,  # 复用lstm_hidden参数
                dropout_rate=args.dropout
            )
        elif model_type == 'cnn_attention':
            model = CNNClassifier(
                input_channels=input_channels,
                seq_length=seq_length,
                num_classes=num_classes,
                base_filters=args.filters,
                kernel_sizes=[args.kernel_size,
                              args.kernel_size+2, args.kernel_size+4],
                dropout_rate=args.dropout
            )
        elif model_type == 'cnn_model':
            model = SimpleCNNClassifier(
                input_channels=input_channels,
                seq_length=seq_length,
                num_classes=num_classes,
                base_filters=args.filters,
                kernel_sizes=[args.kernel_size,
                              args.kernel_size+2, args.kernel_size+4],
                dropout_rate=args.dropout
            )
        elif model_type == 'mlp':
            input_dim = seq_length * input_channels  # 展平原始信号
            model = MLPClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_layers=[512, 256, 128],  # 对于原始信号，使用更大的隐藏层
                dropout_rate=args.dropout
            )
        elif model_type == 'svm':
            model = SVMClassifier(
                input_shape=input_shape,
                num_classes=num_classes,
                kernel=args.svm_kernel,
                C=args.svm_C,
                gamma=args.svm_gamma
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    return model.to(device)


def train_workflow(processor, args, device):
    """
    训练工作流

    参数:
    - processor: 数据处理器
    - args: 命令行参数
    - device: 计算设备
    """
    print("开始训练工作流...")

    # 准备数据集
    print(f"从 {args.data_dir} 加载数据...")
    X, y, label_map = processor.prepare_dataset(
        augment=args.augment,
        use_features=args.use_features
    )

    # 保存标签映射
    with open('label_map.json', 'w') as f:
        json.dump({k: int(v) for k, v in label_map.items()}, f)

    print(f"数据集大小: {X.shape}")
    print(f"标签数量: {len(label_map)}")

    # 找出实际存在于数据中的类别
    unique_classes = np.unique(y)
    print(f"数据中存在的类别: {unique_classes}")
    print(f"存在的类别数量: {len(unique_classes)}")

    # 获取类别名称列表，确保只包含存在于数据中的类别
    class_names = []
    for i in unique_classes:
        # 找到标签映射中与此索引对应的名称
        for name, idx in label_map.items():
            if idx == i:
                class_names.append(name)
                break
        else:
            class_names.append(f"Class {i}")

    print(f"类别名称: {class_names}")

    # 可视化类别分布
    processor.visualize_class_distribution(y, label_map)

    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)

    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 归一化数据
    X_train_norm, X_val_norm, X_test_norm = processor.normalize_data(
        X_train, X_val, X_test)

    # 保存归一化参数
    np.save('normalization_mean.npy', processor.scaler.mean_)
    np.save('normalization_std.npy', processor.scaler.scale_)

    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_dataloaders(
        X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, batch_size=args.batch_size
    )

    # 创建指定类型的模型
    model = create_model(args.model_type, X_train_norm.shape,
                         len(unique_classes), args, device)

    # 打印模型结构
    print(model)

    # 记录训练开始时间
    start_time = time.time()

    # 根据模型类型选择不同的训练方法
    if args.model_type == 'svm':
        # 对于SVM模型
        if args.svm_grid_search:
            # 如果启用网格搜索
            print("执行SVM网格搜索...")
            best_params = model.grid_search(train_loader)
            print(f"SVM最佳参数: {best_params}")

            # 更新配置文件中的SVM参数
            args.svm_kernel = best_params['kernel']
            args.svm_C = best_params['C']
            args.svm_gamma = best_params['gamma']
        else:
            # 直接训练SVM
            print("开始训练SVM模型...")
            history = model.fit(train_loader, val_loader)

        # 保存SVM模型
        # 确保SVM模型路径有正确的扩展名
        svm_model_path = args.model_path
        if not svm_model_path.endswith('.joblib'):
            svm_model_path = svm_model_path.rsplit('.', 1)[0] + '.joblib'
        model.save(svm_model_path)
        args.model_path = svm_model_path  # 更新模型路径
    else:
        # 对于神经网络模型
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # 训练模型
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=args.epochs, patience=args.patience, scheduler=scheduler
        )

        # 保存模型
        save_model(model, args.model_path)

    # 记录训练结束时间
    training_time = time.time() - start_time
    print(f"训练完成，用时: {training_time:.2f} 秒")

    # 可视化学习曲线 (如果有历史记录)
    if args.model_type != 'svm':
        # 对于神经网络模型，使用标准的可视化函数
        visualize_learning_curves(history)
    elif 'history' in locals() and history is not None:
        # 对于SVM模型，只显示准确率
        plt.figure(figsize=(12, 5))

        # 绘制准确率曲线
        plt.plot(history['train_acc'], label='train_acc')
        if history['val_acc'] is not None:
            plt.plot(history['val_acc'], label='val_acc')
        plt.title('SVM Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy rate')
        plt.legend()

        plt.tight_layout()
        plt.savefig('learning_curves_svm.png', dpi=300)
        plt.show()

    # 评估模型
    print("在测试集上评估模型...")
    if args.model_type == 'svm':
        # 对于SVM模型，直接计算准确率
        X_test_flat = X_test_norm.reshape(
            X_test_norm.shape[0], -1) if len(X_test_norm.shape) == 3 else X_test_norm
        predictions = model.model.predict(X_test_flat)
        test_acc = np.mean(predictions == y_test)
        print(f"测试准确率: {test_acc:.4f}")

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, predictions)

        # 可视化混淆矩阵
        visualize_confusion_matrix(cm, class_names)

        # 创建临时变量以便兼容后续代码
        all_predictions = predictions
        all_labels = y_test
    else:
        # 对于神经网络模型，使用evaluate_model函数
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, cm, all_predictions, all_labels = evaluate_model(
            model, test_loader, criterion, device, class_names
        )

        # 可视化混淆矩阵
        visualize_confusion_matrix(cm, class_names)

    # 可视化注意力权重 (对于有意义的注意力机制的模型)
    if not args.use_features and args.model_type in ['cnn_lstm_attention', 'cnn_bigru_attention', 'cnn_attention']:
        visualize_attention_weights(model, test_loader, device)

    # 可视化潜在空间（使用t-SNE）- 对所有模型类型
    visualize_latent_space(model, test_loader, device,
                           len(unique_classes), all_labels)

    # 保存配置信息
    config = {
        'model_type': args.model_type,
        'window_size': args.window_size,
        'use_features': args.use_features,
        'input_channels': X_train_norm.shape[2] if not args.use_features else None,
        'seq_length': X_train_norm.shape[1] if not args.use_features else None,
        'feature_dim': X_train_norm.shape[1] if args.use_features else None,
        'num_classes': len(unique_classes),
        # SVM特定参数
        'svm_kernel': args.svm_kernel if args.model_type == 'svm' else None,
        'svm_C': args.svm_C if args.model_type == 'svm' else None,
        'svm_gamma': args.svm_gamma if args.model_type == 'svm' else None
    }

    with open('model_config.json', 'w') as f:
        json.dump(config, f)

    print(f"模型已保存至 {args.model_path}")
    print(f"最终测试准确率: {test_acc:.4f}")


def test_workflow(processor, args, device):
    """
    测试工作流

    参数:
    - processor: 数据处理器
    - args: 命令行参数
    - device: 计算设备
    """
    print("开始测试工作流...")

    # 加载配置
    try:
        with open('model_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("错误：找不到模型配置文件。请先训练模型或提供有效的配置文件。")
        return

    # 加载标签映射
    try:
        with open('label_map.json', 'r') as f:
            label_map = json.load(f)
    except FileNotFoundError:
        print("错误：找不到标签映射文件。请先训练模型或提供有效的标签映射。")
        return

    # 获取模型类型
    model_type = config.get('model_type', 'cnn_lstm_attention')

    # 加载数据
    print(f"从 {args.data_dir} 加载数据...")
    X, y, _ = processor.prepare_dataset(
        augment=False,  # 测试不需要数据增强
        use_features=config['use_features']
    )

    # 找出实际存在于数据中的类别
    unique_classes = np.unique(y)
    print(f"数据中存在的类别: {unique_classes}")
    print(f"存在的类别数量: {len(unique_classes)}")

    # 获取类别名称列表，确保只包含存在于数据中的类别
    class_names = []
    for i in unique_classes:
        # 找到标签映射中与此索引对应的名称
        for name, idx in label_map.items():
            if int(idx) == i:
                class_names.append(name)
                break
        else:
            class_names.append(f"Class {i}")

    print(f"类别名称: {class_names}")

    # 分割数据（只取测试集）
    _, _, X_test, _, _, y_test = processor.split_data(X, y)

    # 加载归一化参数
    try:
        mean = np.load('normalization_mean.npy')
        std = np.load('normalization_std.npy')
        processor.scaler.mean_ = mean
        processor.scaler.scale_ = std
    except FileNotFoundError:
        print("警告：找不到归一化参数文件。将使用测试数据进行拟合，可能导致性能下降。")

    # 归一化数据
    X_test_norm, = processor.normalize_data(X_test)

    # 创建测试数据加载器
    test_dataset = processor.BearingDataset(X_test_norm, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size)

    # 设置SVM特定参数（如果适用）
    if model_type == 'svm':
        args.svm_kernel = config.get('svm_kernel', 'rbf')
        args.svm_C = config.get('svm_C', 1.0)
        args.svm_gamma = config.get('svm_gamma', 'scale')

    # 创建模型
    if config['use_features']:
        input_shape = (X_test_norm.shape[0], config['feature_dim'])
    else:
        input_shape = (
            X_test_norm.shape[0], config['seq_length'], config['input_channels'])

    # 创建模型
    model = create_model(model_type, input_shape,
                         config['num_classes'], args, device)

    # 加载模型权重
    if model_type == 'svm':
        # 确保SVM模型路径有正确的扩展名
        svm_model_path = args.model_path
        if not svm_model_path.endswith('.joblib'):
            svm_model_path = svm_model_path.rsplit('.', 1)[0] + '.joblib'
        model.load(svm_model_path)
    else:
        model = load_model(model, args.model_path, device)

    # 评估模型
    print("评估模型性能...")
    if model_type == 'svm':
        # 对于SVM模型，直接计算准确率
        X_test_flat = X_test_norm.reshape(
            X_test_norm.shape[0], -1) if len(X_test_norm.shape) == 3 else X_test_norm
        predictions = model.model.predict(X_test_flat)
        test_acc = np.mean(predictions == y_test)
        print(f"测试准确率: {test_acc:.4f}")

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, predictions)

        # 可视化混淆矩阵
        visualize_confusion_matrix(cm, class_names)

        # 创建临时变量以便兼容后续代码
        all_predictions = predictions
        all_labels = y_test
    else:
        # 对于神经网络模型，使用evaluate_model函数
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, cm, all_predictions, all_labels = evaluate_model(
            model, test_loader, criterion, device, class_names
        )

        # 可视化混淆矩阵
        visualize_confusion_matrix(cm, class_names)

    # 可视化注意力权重（对于有意义的注意力机制的模型）
    if not config['use_features'] and model_type in ['cnn_lstm_attention', 'cnn_bigru_attention', 'cnn_attention']:
        visualize_attention_weights(model, test_loader, device)
    # 可视化潜在空间（使用t-SNE）- 对所有模型类型
    visualize_latent_space(model, test_loader, device,
                           config['num_classes'], all_labels)

    print(f"测试准确率: {test_acc:.4f}")


def predict_workflow(processor, args, device):
    """
    预测工作流

    参数:
    - processor: 数据处理器
    - args: 命令行参数
    - device: 计算设备
    """
    if args.predict_file is None:
        print("错误：预测模式需要指定 --predict_file 参数。")
        return

    print(f"开始预测工作流，预测文件: {args.predict_file}...")

    # 加载配置
    try:
        with open('model_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("错误：找不到模型配置文件。请先训练模型或提供有效的配置文件。")
        return

    # 加载标签映射
    try:
        with open('label_map.json', 'r') as f:
            label_map = json.load(f)
            # 反转映射，用于获取类别名称
            reverse_label_map = {int(v): k for k, v in label_map.items()}
    except FileNotFoundError:
        print("错误：找不到标签映射文件。请先训练模型或提供有效的标签映射。")
        return

    # 获取模型类型
    model_type = config.get('model_type', 'cnn_lstm_attention')

    # 加载并处理预测文件
    try:
        # 加载数据
        data = processor.load_and_process_data(args.predict_file)
        segments = processor.segment_data(data)

        # 提取特征（如果需要）
        if config['use_features']:
            features = []
            for segment in segments:
                # 提取时域特征
                time_features = processor.extract_time_domain_features(segment)

                # 提取频域特征
                freq_features = processor.extract_frequency_domain_features(
                    segment)

                # 提取小波特征
                wavelet_features = processor.extract_wavelet_features(segment)

                # 组合所有特征
                combined_features = np.concatenate(
                    [time_features, freq_features, wavelet_features])
                features.append(combined_features)

            X_pred = np.array(features)
            input_shape = (X_pred.shape[0], config['feature_dim'])
        else:
            # 使用原始信号
            X_pred = segments
            input_shape = (
                X_pred.shape[0], config['seq_length'], config['input_channels'])

        # 加载归一化参数
        try:
            mean = np.load('normalization_mean.npy')
            std = np.load('normalization_std.npy')
            processor.scaler.mean_ = mean
            processor.scaler.scale_ = std
        except FileNotFoundError:
            print("警告：找不到归一化参数文件。将使用预测数据进行拟合，可能导致性能下降。")

        # 归一化数据
        X_pred_norm, = processor.normalize_data(X_pred)

        # 设置SVM特定参数（如果适用）
        if model_type == 'svm':
            args.svm_kernel = config.get('svm_kernel', 'rbf')
            args.svm_C = config.get('svm_C', 1.0)
            args.svm_gamma = config.get('svm_gamma', 'scale')

        # 创建模型
        model = create_model(model_type, input_shape,
                             config['num_classes'], args, device)

        # 加载模型权重
        if model_type == 'svm':
            # 确保SVM模型路径有正确的扩展名
            svm_model_path = args.model_path
            if not svm_model_path.endswith('.joblib'):
                svm_model_path = svm_model_path.rsplit('.', 1)[0] + '.joblib'
            model.load(svm_model_path)

            # 预测
            X_pred_flat = X_pred_norm.reshape(
                X_pred_norm.shape[0], -1) if len(X_pred_norm.shape) == 3 else X_pred_norm
            predictions = model.model.predict(X_pred_flat)

            # SVM没有注意力权重，创建一个空列表
            attention_weights_list = []
        else:
            # 对于神经网络模型，使用forward()方法
            model = load_model(model, args.model_path, device)
            model.eval()
            predictions = []
            attention_weights_list = []

            # 转换为PyTorch张量
            X_pred_tensor = torch.FloatTensor(X_pred_norm).to(device)

            with torch.no_grad():
                for i in range(0, len(X_pred_tensor), args.batch_size):
                    batch = X_pred_tensor[i:i+args.batch_size]
                    outputs, attention_weights = model(batch)
                    _, predicted = torch.max(outputs, 1)
                    predictions.extend(predicted.cpu().numpy())
                    attention_weights_list.extend(
                        attention_weights.cpu().numpy())

        # 统计预测结果
        predictions = np.array(predictions)
        class_counts = np.bincount(
            predictions, minlength=config['num_classes'])
        most_common_class = np.argmax(class_counts)

        # 计算置信度
        confidence = class_counts[most_common_class] / len(predictions)

        # 获取预测类别名称
        predicted_class_name = reverse_label_map.get(
            int(most_common_class), f"未知类别 {most_common_class}")

        print(f"\n预测结果:")
        print(f"预测类别: {predicted_class_name}")
        print(f"置信度: {confidence:.2f}")

        # 可视化预测结果
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(class_counts)), class_counts)
        plt.xticks(range(len(class_counts)),
                   [reverse_label_map.get(i, f"类别 {i}")
                    for i in range(len(class_counts))],
                   rotation=90)
        plt.title('number of samples per class')
        plt.xlabel('class')
        plt.ylabel('number')
        plt.tight_layout()
        plt.savefig('prediction_distribution.png', dpi=300)
        plt.show()

        # 可视化最有信心的样本的振动信号和注意力权重（仅对于带注意力机制的模型）
        if not config['use_features'] and model_type in ['cnn_lstm_attention', 'cnn_bigru_attention', 'cnn_attention'] and attention_weights_list:
            # 找出预测为最常见类别的样本
            indices = np.where(predictions == most_common_class)[0]

            if len(indices) > 0:
                # 选择最前面的三个样本
                sample_indices = indices[:min(3, len(indices))]

                plt.figure(figsize=(15, 10))

                for i, idx in enumerate(sample_indices):
                    sample = X_pred[idx]
                    attention = attention_weights_list[idx]

                    # 计算原始信号的峰值，用于绘图标准化
                    signal_peak = np.max(np.abs(sample))

                    # 绘制子图
                    plt.subplot(len(sample_indices), 1, i+1)

                    # 绘制X轴信号
                    plt.plot(sample[:, 0], 'b-',
                             label='X-aixs signal', alpha=0.7)

                    # 绘制Y轴信号
                    plt.plot(sample[:, 1], 'g-',
                             label='Y-aixs signal', alpha=0.7)

                    # 绘制Z轴信号
                    plt.plot(sample[:, 2], 'r-',
                             label='Z-aixs signal', alpha=0.7)

                    # 绘制注意力权重（根据原始信号的峰值进行缩放）
                    # 上采样注意力权重以匹配原始信号长度
                    upsampled_attention = np.repeat(attention, 4)[:len(sample)]
                    plt.plot(upsampled_attention * signal_peak, 'k-',
                             label='attention weight', linewidth=2)

                    plt.title(f'sample {i+1} - predict {predicted_class_name}')
                    plt.xlabel('step')
                    plt.ylabel('amplitude')
                    plt.legend()

                plt.tight_layout()
                plt.savefig('prediction_attention.png', dpi=300)
                plt.show()

        return predicted_class_name, confidence

    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None, 0.0


def visualize_latent_space(model, data_loader, device, num_classes, labels=None):
    """
    使用t-SNE可视化潜在空间

    参数:
    - model: 训练好的模型
    - data_loader: 数据加载器
    - device: 计算设备
    - num_classes: 类别数量
    - labels: 如果提供，使用这些标签而不是从数据加载器提取
    """
    # 将模型设为评估模式
    model.eval()

    # 收集所有样本的潜在表示
    latent_representations = []
    all_labels = []

    # 检查模型类型
    is_svm = isinstance(model, SVMClassifier)

    if is_svm:
        # 对于SVM模型，直接获取所有样本
        for data, target in data_loader:
            # 预处理数据
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
                target = target.cpu().numpy()

            # 展平数据
            data_flat = data.reshape(
                data.shape[0], -1) if len(data.shape) == 3 else data

            # 获取到支持向量的距离
            latent = model.get_latent(data_flat)

            latent_representations.append(latent)
            all_labels.append(target)

        # 合并所有批次的数据
        latent_representations = np.vstack(latent_representations)
        all_labels = np.concatenate(all_labels)
    else:
        # 对于神经网络模型
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 50:  # 限制样本数量，t-SNE计算成本高
                    break

                data, target = data.to(device), target.to(device)

                # 获取潜在表示
                if hasattr(model, 'get_latent'):
                    # 如果模型有get_latent方法，使用它
                    latent = model.get_latent(data)
                else:
                    # 否则使用注意力权重
                    _, attention_weights = model(data)
                    latent = attention_weights

                latent_representations.append(latent.cpu().numpy())
                all_labels.append(target.cpu().numpy())

        # 合并所有批次的数据
        latent_representations = np.vstack(latent_representations)
        if labels is not None:
            all_labels = labels[:len(latent_representations)]
        else:
            all_labels = np.concatenate(all_labels)

    # 使用t-SNE降维
    print("计算t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42,
                init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(latent_representations)

    # 绘制t-SNE结果
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.get_cmap('tab20', num_classes)

    for i in range(num_classes):
        indices = all_labels == i
        if np.any(indices):  # 只绘制有样本的类别
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        c=[cmap(i)], label=f'class {i}', alpha=0.7)

    plt.title('t-SNE Visualization of Latent Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300)
    plt.show()


class FeatureBasedModel(nn.Module):
    """
    基于手工特征的全连接神经网络模型
    """

    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64], dropout=0.3):
        super(FeatureBasedModel, self).__init__()

        layers = []
        prev_dim = input_dim

        # 添加隐藏层
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

        # 用于获取潜在表示的辅助函数
        self.hidden_layers = nn.Sequential(*layers[:-1])  # 除了最后一层

    def forward(self, x):
        outputs = self.model(x)
        # 为了与CNN-LSTM-Attention模型兼容，我们返回一个假的注意力权重
        # 实际上这个模型没有注意力机制
        batch_size = x.size(0)
        dummy_attention = torch.ones(
            batch_size, 10, device=x.device) / 10  # 10是任意选择的
        return outputs, dummy_attention

    def get_latent(self, x):
        """
        获取倒数第二层的输出作为潜在表示
        """
        return self.hidden_layers(x)


if __name__ == "__main__":
    main()

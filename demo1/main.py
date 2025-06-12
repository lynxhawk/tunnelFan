import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from data_loader import BearingDataProcessor

# 导入模型 - 使用直接导入方式
from models import MLPClassifier, CNNClassifier, create_model
from ltsf_linear_model import LTSFLinearClassifier, create_ltsf_linear_model
from svm_model import SVMClassifier, create_svm_model
from rnn_models import LSTMClassifier, GRUClassifier, create_lstm_model, create_gru_model
# 在现有导入后添加：
from transformer_models import TransformerClassifier, CNNTransformerClassifier, create_transformer_model, create_cnn_transformer_model
# 在现有导入后添加：
from se_cnn_patchtst_models import (
    SECNNPatchTSTStandard, SECNNPatchTSTOneLayer, SECNNPatchTSTThreeLayer, StandardCNNPatchTST,
    create_se_cnn_patchtst_standard, create_se_cnn_patchtst_one_layer, 
    create_se_cnn_patchtst_three_layer, create_standard_cnn_patchtst
)


def is_svm_model(model):
    """检查是否为SVM模型"""
    return isinstance(model, SVMClassifier)


def get_supported_models():
    """获取支持的模型列表"""
    return ['MLP', 'CNN', 'LTSF', 'SVM', 'LSTM', 'GRU', 'TRANSFORMER', 'CNN_TRANSFORMER',
            'SE_CNN_PATCHTST', 'SE_CNN_PATCHTST_ONE', 'SE_CNN_PATCHTST_THREE', 'CNN_PATCHTST']


def create_unified_model(model_type, input_shape, num_classes, **kwargs):
    """统一的模型创建函数"""
    model_type = model_type.upper()
    
    # 处理输入形状
    if len(input_shape) == 3:  # [batch, seq_length, channels]
        batch_size, seq_length, input_channels = input_shape
        input_dim = seq_length * input_channels  # 用于MLP
    elif len(input_shape) == 2:  # [seq_length, channels] 去掉batch维度
        seq_length, input_channels = input_shape
        input_dim = seq_length * input_channels
    else:  # 特征数据 [features]
        input_dim = input_shape[0] if len(input_shape) == 1 else input_shape[1]
        seq_length = 1000  # 默认值
        input_channels = 1
    
    # 提取通用参数
    dropout_rate = kwargs.get('dropout_rate', 0.3)
    
    if model_type == 'MLP':
        hidden_layers = kwargs.get('hidden_layers', [512, 256, 128])
        model = MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )
    
    elif model_type == 'CNN':
        base_filters = kwargs.get('base_filters', 64)
        model = CNNClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            base_filters=base_filters,
            dropout_rate=dropout_rate
        )
    
    elif model_type == 'LTSF':
        hidden_dim = kwargs.get('hidden_dim', 64)
        kernel_size = kwargs.get('kernel_size', 25)
        individual = kwargs.get('individual', False)
        
        model = create_ltsf_linear_model(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            individual=individual,
            dropout_rate=dropout_rate
        )
    
    elif model_type == 'SVM':
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 1.0)
        gamma = kwargs.get('gamma', 'scale')
        use_scaler = kwargs.get('use_scaler', True)
        
        model = create_svm_model(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            kernel=kernel,
            C=C,
            gamma=gamma,
            use_scaler=use_scaler,
            dropout_rate=dropout_rate  # SVM会忽略这个参数
        )
    
    elif model_type == 'LSTM':
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 2)
        bidirectional = kwargs.get('bidirectional', False)
        attention = kwargs.get('attention', False)
        model_variant = kwargs.get('model_variant', 'standard')
        
        model = create_lstm_model(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            attention=attention,
            model_variant=model_variant
        )

    elif model_type == 'GRU':
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 2)
        bidirectional = kwargs.get('bidirectional', False)
        attention = kwargs.get('attention', False)
        
        model = create_gru_model(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            attention=attention
        )
    elif model_type == 'TRANSFORMER':
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('transformer_layers', 4)
        dim_feedforward = kwargs.get('dim_feedforward', 256)
        
        model = create_transformer_model(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate
        )

    elif model_type == 'CNN_TRANSFORMER':
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_transformer_layers = kwargs.get('transformer_layers', 3)
        dim_feedforward = kwargs.get('dim_feedforward', 256)
        cnn_filters = kwargs.get('cnn_filters', 32)
        cnn_kernel_size = kwargs.get('cnn_kernel_size', 7)
        cnn_layers = kwargs.get('cnn_layers', 2)
        pool_size = kwargs.get('pool_size', 4)
        
        model = create_cnn_transformer_model(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            cnn_layers=cnn_layers,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            pool_size=pool_size
        )
    
    elif model_type == 'SE_CNN_PATCHTST':
        patch_size = kwargs.get('patch_size', 16)
        stride = kwargs.get('stride', 8)
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('transformer_layers', 3)
        base_filters = kwargs.get('base_filters', 32)
        se_reduction = kwargs.get('se_reduction', 8)
        use_se = kwargs.get('use_se', True)
        
        model = create_se_cnn_patchtst_standard(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            base_filters=base_filters,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )

    elif model_type == 'SE_CNN_PATCHTST_ONE':
        patch_size = kwargs.get('patch_size', 16)
        stride = kwargs.get('stride', 8)
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('transformer_layers', 3)
        base_filters = kwargs.get('base_filters', 32)
        se_reduction = kwargs.get('se_reduction', 8)
        use_se = kwargs.get('use_se', True)
        
        model = create_se_cnn_patchtst_one_layer(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            base_filters=base_filters,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )

    elif model_type == 'SE_CNN_PATCHTST_THREE':
        patch_size = kwargs.get('patch_size', 16)
        stride = kwargs.get('stride', 8)
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('transformer_layers', 3)
        base_filters = kwargs.get('base_filters', 32)
        se_reduction = kwargs.get('se_reduction', 8)
        use_se = kwargs.get('use_se', True)
        
        model = create_se_cnn_patchtst_three_layer(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            base_filters=base_filters,
            dropout_rate=dropout_rate,
            use_se=use_se,
            se_reduction=se_reduction
        )

    elif model_type == 'CNN_PATCHTST':
        patch_size = kwargs.get('patch_size', 16)
        stride = kwargs.get('stride', 8)
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('transformer_layers', 3)
        base_filters = kwargs.get('base_filters', 32)
        
        model = create_standard_cnn_patchtst(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            base_filters=base_filters,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


class BearingClassificationTrainer:
    """轴承故障分类训练器 - 支持多种模型类型"""
    
    def __init__(self, model, device, save_dir='checkpoints', model_type='CNN'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.model_type = model_type.upper()
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 检查是否为SVM模型
        self.is_svm = is_svm_model(model)
        
        if self.is_svm:
            print("检测到SVM模型，将使用SVM专用训练流程")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch - 仅适用于神经网络模型"""
        if self.is_svm:
            raise RuntimeError("SVM模型不支持epoch训练，请使用train方法")
            
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_data, batch_labels in pbar:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.squeeze().to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs, _ = self.model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*total_correct/total_samples:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch - 仅适用于神经网络模型"""
        if self.is_svm:
            raise RuntimeError("SVM模型不支持epoch验证，请使用evaluate方法")
            
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_data, batch_labels in pbar:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                # 前向传播
                outputs, _ = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
                
                # 保存预测结果用于详细分析
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*total_correct/total_samples:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc, all_predictions, all_labels
    
    def train_svm(self, train_loader, val_loader, **kwargs):
        """SVM专用训练方法"""
        if not self.is_svm:
            raise RuntimeError("此方法仅适用于SVM模型")
        
        print("开始SVM训练...")
        
        # 使用SVM的fit_data_loaders方法
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
        
        # 保存SVM模型
        model_path = os.path.join(self.save_dir, f'svm_best_model_acc_{self.best_accuracy:.4f}.joblib')
        self.model.save_model(model_path)
        print(f"SVM模型已保存到: {model_path}")
        
        return self.training_history
    
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001,
              weight_decay=1e-5, patience=15, save_best=True):
        """完整训练流程 - 自动适配模型类型"""
        
        # 如果是SVM模型，使用SVM专用训练
        if self.is_svm:
            return self.train_svm(train_loader, val_loader)
        
        # 神经网络模型的标准训练流程
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        print(f"开始训练 {self.model_type} 模型，共 {num_epochs} 个epoch...")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        early_stop_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc, val_predictions, val_labels = self.validate_epoch(val_loader, criterion)
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # 打印结果
            print(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                early_stop_counter = 0
                
                if save_best:
                    # 删除之前的最佳模型文件
                    self._cleanup_old_models()
                    # 保存新的最佳模型
                    self.save_model(f'best_model_acc_{val_acc:.4f}.pth')
                    print(f"保存最佳模型 (Accuracy: {val_acc:.4f})")
            else:
                early_stop_counter += 1
            
            # 早停检查
            if early_stop_counter >= patience:
                print(f"验证准确率连续 {patience} 个epoch未提升，执行早停")
                break
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time/60:.2f} 分钟")
        print(f"最佳验证准确率: {self.best_accuracy:.4f}")
        
        return self.training_history
    
    def _cleanup_old_models(self):
        """清理旧的最佳模型文件"""
        if self.is_svm:
            return  # SVM模型有自己的保存机制
            
        try:
            # 查找所有以 'best_model' 开头的文件
            best_model_files = [f for f in os.listdir(self.save_dir) if f.startswith('best_model_acc_')]
            
            # 删除所有旧的最佳模型文件
            for old_file in best_model_files:
                old_path = os.path.join(self.save_dir, old_file)
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"删除旧模型: {old_file}")
        except Exception as e:
            print(f"清理旧模型时出错: {e}")
    
    def save_model(self, filename):
        """保存模型"""
        if self.is_svm:
            # SVM模型使用自己的保存方法
            save_path = os.path.join(self.save_dir, filename.replace('.pth', '.joblib'))
            self.model.save_model(save_path)
        else:
            # 神经网络模型使用PyTorch保存方法
            save_path = os.path.join(self.save_dir, filename)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'best_accuracy': self.best_accuracy,
                'training_history': self.training_history,
                'model_type': self.model_type
            }, save_path)
    
    def load_model(self, filename):
        """加载模型"""
        if self.is_svm:
            # SVM模型使用自己的加载方法
            load_path = os.path.join(self.save_dir, filename.replace('.pth', '.joblib'))
            self.model.load_model(load_path)
            # SVM模型需要手动设置best_accuracy
            self.best_accuracy = 0.0  # 或者从文件名解析
        else:
            # 神经网络模型使用PyTorch加载方法
            load_path = os.path.join(self.save_dir, filename)
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
            self.training_history = checkpoint.get('training_history', {})
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if self.is_svm:
            print("SVM模型训练历史较简单，跳过绘图")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.training_history['train_loss'], label='Training Loss', color='blue')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title(f'{self.model_type} Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.training_history['train_acc'], label='Training Accuracy', color='blue')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_title(f'{self.model_type} Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_model(model, test_loader, device, class_names=None, save_results=True, save_dir='checkpoints'):
    """评估模型性能 - 兼容所有模型类型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_latent_features = []
    
    is_svm = is_svm_model(model)
    
    print(f"正在评估{'SVM' if is_svm else 'Neural Network'}模型...")
    
    if not is_svm:
        # 神经网络模型评估
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(test_loader):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.squeeze().to(device)
                
                outputs, _ = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
                # 获取潜在特征用于t-SNE可视化
                if hasattr(model, 'get_latent'):
                    latent = model.get_latent(batch_data)
                    all_latent_features.extend(latent.cpu().numpy())
    else:
        # SVM模型评估
        for batch_data, batch_labels in tqdm(test_loader):
            batch_labels = batch_labels.squeeze()
            
            outputs, _ = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            # 获取潜在特征用于t-SNE可视化
            if hasattr(model, 'get_latent'):
                latent = model.get_latent(batch_data)
                all_latent_features.extend(latent.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"测试准确率: {accuracy:.4f}")
    
    # 分类报告
    if class_names and len(class_names) <= 30:  # 扩展显示限制
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=class_names, digits=4))
    elif class_names and len(class_names) > 30:
        # 对于类别过多的情况，只显示汇总统计
        print(f"\n分类汇总 (共{len(class_names)}个类别):")
        print(classification_report(all_labels, all_predictions, digits=4))
    else:
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, digits=4))
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 可视化混淆矩阵
    visualize_confusion_matrix(cm, class_names, save_results, save_dir)
    
    # 创建详细分析报告
    if len(cm) > 10:
        create_detailed_confusion_analysis(cm, class_names, save_dir)
    
    # t-SNE可视化
    if all_latent_features:
        visualize_tsne(np.array(all_latent_features), np.array(all_labels), 
                      class_names, save_results, save_dir)
    
    return accuracy, all_predictions, all_labels, cm


def visualize_confusion_matrix(cm, class_names=None, save_results=True, save_dir='checkpoints'):
    """可视化混淆矩阵 - 支持类别聚合"""
    
    # 如果类别数量太多(>10)，进行聚合
    if len(cm) > 10:
        print(f"📊 检测到{len(cm)}个类别，聚合为10个组以便可视化...")
        
        # 计算聚合的混淆矩阵
        aggregated_cm = aggregate_confusion_matrix(cm, num_groups=10)
        
        # 创建聚合后的类别名称
        aggregated_class_names = [f'Group {i}' for i in range(10)]
        
        # 绘制聚合后的混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(aggregated_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=aggregated_class_names, yticklabels=aggregated_class_names,
                    square=True, cbar_kws={"shrink": .8})
        plt.title(f'Aggregated Confusion Matrix (10 Groups from {len(cm)} Classes)')
        plt.xlabel('Predicted Group')
        plt.ylabel('True Group')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_results:
            save_path = os.path.join(save_dir, 'confusion_matrix_aggregated.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"聚合混淆矩阵已保存到: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        # 保存详细的分组信息
        save_aggregation_info(len(cm), save_dir)
        
    else:
        # 原有的详细混淆矩阵逻辑
        plt.figure(figsize=(12, 10))
        
        if class_names and len(class_names) <= 30:
            # 对于类别数较少的情况，显示详细的混淆矩阵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        square=True, cbar_kws={"shrink": .8})
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        else:
            # 对于类别数较多的情况，不显示标签
            sns.heatmap(cm, cmap='Blues', square=True, cbar_kws={"shrink": .8})
            plt.xlabel(f'Predicted Label ({len(cm)} classes)')
            plt.ylabel(f'True Label ({len(cm)} classes)')
        
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_results:
            save_path = os.path.join(save_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        plt.show()


def aggregate_confusion_matrix(cm, num_groups=10):
    """将混淆矩阵聚合为指定数量的组"""
    num_classes = len(cm)
    
    # 计算每组的大小
    group_size = num_classes // num_groups
    remainder = num_classes % num_groups
    
    # 创建聚合后的混淆矩阵
    aggregated_cm = np.zeros((num_groups, num_groups), dtype=int)
    
    # 聚合逻辑：将连续的类别分组
    for i in range(num_classes):
        for j in range(num_classes):
            # 确定源类别属于哪个组
            true_group = min(i // group_size, num_groups - 1)
            pred_group = min(j // group_size, num_groups - 1)
            
            # 累加到对应的组
            aggregated_cm[true_group, pred_group] += cm[i, j]
    
    return aggregated_cm


def save_aggregation_info(num_classes, save_dir):
    """保存聚合信息"""
    group_size = num_classes // 10
    remainder = num_classes % 10
    
    aggregation_info = {
        "total_classes": num_classes,
        "num_groups": 10,
        "group_mapping": {}
    }
    
    # 生成分组映射
    for group_id in range(10):
        start_class = group_id * group_size
        if group_id == 9:  # 最后一组包含余数
            end_class = num_classes - 1
        else:
            end_class = (group_id + 1) * group_size - 1
        
        aggregation_info["group_mapping"][f"Group {group_id}"] = {
            "class_range": f"{start_class}-{end_class}",
            "num_classes": end_class - start_class + 1
        }
    
    # 保存到JSON文件
    info_path = os.path.join(save_dir, 'confusion_matrix_aggregation_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(aggregation_info, f, indent=2, ensure_ascii=False)
    
    print(f"📋 分组信息已保存到: {info_path}")
    
    # 打印分组信息
    print(f"\n📊 混淆矩阵分组详情:")
    for group_name, info in aggregation_info["group_mapping"].items():
        print(f"  {group_name}: 类别 {info['class_range']} ({info['num_classes']}个类别)")


def create_detailed_confusion_analysis(cm, class_names, save_dir):
    """创建详细的混淆矩阵分析报告"""
    num_classes = len(cm)
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(num_classes):
        if np.sum(cm[i, :]) > 0:
            accuracy = cm[i, i] / np.sum(cm[i, :])
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0.0)
    
    # 找出表现最好和最差的类别
    best_classes = np.argsort(class_accuracies)[-5:][::-1]  # 前5个最好的
    worst_classes = np.argsort(class_accuracies)[:5]        # 前5个最差的
    
    # 生成分析报告
    analysis = {
        "overall_accuracy": np.trace(cm) / np.sum(cm),
        "per_class_accuracy": {
            str(i): float(acc) for i, acc in enumerate(class_accuracies)
        },
        "best_performing_classes": {
            str(i): float(class_accuracies[i]) for i in best_classes
        },
        "worst_performing_classes": {
            str(i): float(class_accuracies[i]) for i in worst_classes
        },
        "total_samples": int(np.sum(cm)),
        "num_classes": num_classes
    }
    
    # 保存分析报告
    analysis_path = os.path.join(save_dir, 'confusion_matrix_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"📈 详细分析报告已保存到: {analysis_path}")
    
    # 打印简要分析
    print(f"\n📈 混淆矩阵分析摘要:")
    print(f"  总体准确率: {analysis['overall_accuracy']:.4f}")
    print(f"  表现最好的5个类别:")
    for class_id in best_classes:
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
        print(f"    类别 {class_id} ({class_name[:20]}...): {class_accuracies[class_id]:.4f}")


def visualize_tsne(latent_features, labels, class_names=None, save_results=True, 
                   save_dir='checkpoints', max_samples=5000):
    """使用t-SNE可视化潜在特征"""
    print("正在计算t-SNE可视化...")
    
    # 限制样本数量以提高计算效率
    if len(latent_features) > max_samples:
        indices = np.random.choice(len(latent_features), max_samples, replace=False)
        latent_features = latent_features[indices]
        labels = labels[indices]
    
    # 计算t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                n_iter=1000, learning_rate='auto')
    tsne_results = tsne.fit_transform(latent_features)
    
    # 绘制t-SNE结果
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        indices = labels == label
        if np.any(indices):
            class_name = class_names[label] if class_names and label < len(class_names) else f'Class {label}'
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                       c=[colors[i]], label=class_name, alpha=0.7, s=10)
    
    plt.title('t-SNE Visualization of Latent Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # 如果类别太多，不显示图例
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    
    plt.tight_layout()
    
    if save_results:
        save_path = os.path.join(save_dir, 'tsne_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE可视化已保存到: {save_path}")
    plt.show()


def save_model(model, filepath, additional_info=None):
    """保存模型"""
    # 检查是否为SVM模型
    if is_svm_model(model):
        # SVM使用自己的保存方法
        model.save_model(filepath)
        return
    
    # 神经网络模型使用PyTorch保存方法
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__
    }
    
    if additional_info:
        save_dict.update(additional_info)
    
    torch.save(save_dict, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(model, filepath, device):
    """加载模型"""
    # 检查是否为SVM模型
    if is_svm_model(model):
        # SVM使用自己的加载方法
        model.load_model(filepath)
        return model
    
    # 神经网络模型使用PyTorch加载方法
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='轴承故障诊断训练')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='bearing_dataset',
                       help='数据集目录')
    parser.add_argument('--seq_length', type=int, default=1000,
                       help='序列长度')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                       help='窗口重叠比例')
    
    # 模型相关参数 - 增加了新的模型选择
    parser.add_argument('--model', type=str, default='CNN', 
                   choices=['MLP', 'CNN', 'LTSF', 'SVM', 'LSTM', 'GRU', 'TRANSFORMER', 'CNN_TRANSFORMER',
                           'SE_CNN_PATCHTST', 'SE_CNN_PATCHTST_ONE', 'SE_CNN_PATCHTST_THREE', 'CNN_PATCHTST'],
                   help='选择模型类型')
    parser.add_argument('--filters', type=int, default=64,
                       help='CNN滤波器数量')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比率')
    
    # LTSF模型特定参数
    parser.add_argument('--ltsf_hidden_dim', type=int, default=64,
                       help='LTSF模型隐藏层维度')
    parser.add_argument('--ltsf_kernel_size', type=int, default=25,
                       help='LTSF模型分解核大小')
    parser.add_argument('--ltsf_individual', action='store_true',
                       help='LTSF模型是否使用独立通道处理')
    
    # SVM模型特定参数
    parser.add_argument('--svm_kernel', type=str, default='rbf',
                       choices=['rbf', 'linear', 'poly', 'sigmoid'],
                       help='SVM核函数')
    parser.add_argument('--svm_C', type=float, default=1.0,
                       help='SVM正则化参数')
    parser.add_argument('--svm_gamma', type=str, default='scale',
                       help='SVM gamma参数')
    parser.add_argument('--svm_grid_search', action='store_true',
                       help='是否对SVM进行网格搜索')
    
    # RNN模型通用参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='RNN隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                    help='RNN层数')
    parser.add_argument('--bidirectional', action='store_true',
                    help='是否使用双向RNN')
    parser.add_argument('--attention', action='store_true',
                    help='是否使用注意力机制')

    # LSTM特定参数
    parser.add_argument('--lstm_variant', type=str, default='standard',
                    choices=['standard', 'stacked'],
                    help='LSTM模型变体')

    # Transformer模型参数
    parser.add_argument('--d_model', type=int, default=128,
                    help='Transformer模型维度')
    parser.add_argument('--nhead', type=int, default=8,
                    help='Transformer注意力头数')
    parser.add_argument('--transformer_layers', type=int, default=4,
                    help='Transformer层数')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                    help='Transformer前馈网络维度')

    # CNN-Transformer特定参数
    parser.add_argument('--cnn_filters', type=int, default=32,
                    help='CNN-Transformer中CNN滤波器数量')
    parser.add_argument('--cnn_kernel_size', type=int, default=7,
                    help='CNN-Transformer中CNN核大小')
    parser.add_argument('--cnn_layers', type=int, default=2,
                    help='CNN-Transformer中CNN层数')
    parser.add_argument('--pool_size', type=int, default=4,
                    help='CNN-Transformer中池化大小')
    
    # PatchTST模型参数
    parser.add_argument('--patch_size', type=int, default=16,
                    help='PatchTST模型patch大小')
    parser.add_argument('--stride', type=int, default=8,
                    help='PatchTST模型patch步长')
    parser.add_argument('--se_reduction', type=int, default=8,
                    help='SE Block的降维比例')
    parser.add_argument('--use_se', action='store_true', default=True,
                    help='是否使用SE Block')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    
    # 数据分割比例
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='测试集比例')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='auto',
                       help='设备选择: auto, cpu, cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'test'], 
                       default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='模型路径')
    
    return parser.parse_args()


def train_workflow(processor, args, device):
    """训练工作流"""
    print("=" * 60)
    print("开始训练工作流")
    print("=" * 60)
    
    # 准备数据 - 使用6:2:2比例
    print("正在加载和处理数据...")
    
    # 根据模型类型选择数据模式
    if args.model.upper() == 'MLP':
        mode = 'feature'
    elif args.model.upper() == 'SVM':
        # SVM可以处理两种数据，默认使用原始信号
        mode = 'signal'
    elif args.model.upper() in ['LSTM', 'GRU', 'TRANSFORMER', 'CNN_TRANSFORMER', 
                           'SE_CNN_PATCHTST', 'SE_CNN_PATCHTST_ONE', 'SE_CNN_PATCHTST_THREE', 'CNN_PATCHTST']:  # 添加这行
        mode = 'signal'
    else:
        mode = 'signal'
    
    # 修复：使用正确的参数名称
    train_loader, val_loader, test_loader = processor.get_data_loaders(
        mode=mode,
        batch_size=args.batch_size,
        test_size=0.4,  # 40%用于验证+测试，会自动分成各20%
        random_state=args.random_seed
    )
    
    # 获取类别信息
    class_info = processor.get_class_info()
    num_classes = class_info['num_classes']
    class_names = class_info['class_names']
    
    print(f"数据加载完成:")
    print(f"  - 类别数量: {num_classes}")
    print(f"  - 训练批次数: {len(train_loader)}")
    print(f"  - 验证批次数: {len(val_loader)}")
    print(f"  - 测试批次数: {len(test_loader)}")
    
    # 获取输入形状
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape
    print(f"  - 输入形状: {input_shape}")
    
    # 创建模型 - 使用统一的create_unified_model函数
    print(f"\n创建 {args.model} 模型...")
    
    # 准备模型参数
    model_kwargs = {
        'dropout_rate': args.dropout,
        'base_filters': args.filters,
    }
    
    # 添加LTSF特定参数
    if args.model.upper() == 'LTSF':
        model_kwargs.update({
            'hidden_dim': args.ltsf_hidden_dim,
            'kernel_size': args.ltsf_kernel_size,
            'individual': args.ltsf_individual,
        })
    
    # 添加SVM特定参数
    elif args.model.upper() == 'SVM':
        model_kwargs.update({
            'kernel': args.svm_kernel,
            'C': args.svm_C,
            'gamma': args.svm_gamma,
            'use_scaler': True,
        })
    elif args.model.upper() in ['LSTM', 'GRU']:
        model_kwargs.update({
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'bidirectional': args.bidirectional,
            'attention': args.attention,
        })
    
        # LSTM特定参数
        if args.model.upper() == 'LSTM':
            model_kwargs['model_variant'] = args.lstm_variant

    elif args.model.upper() in ['TRANSFORMER', 'CNN_TRANSFORMER']:
        model_kwargs.update({
            'd_model': args.d_model,
            'nhead': args.nhead,
            'transformer_layers': args.transformer_layers,
            'dim_feedforward': args.dim_feedforward,
        })
    
        # CNN-Transformer特定参数
        if args.model.upper() == 'CNN_TRANSFORMER':
            model_kwargs.update({
                'cnn_filters': args.cnn_filters,
                'cnn_kernel_size': args.cnn_kernel_size,
                'cnn_layers': args.cnn_layers,
                'pool_size': args.pool_size,
            })

    elif args.model.upper() in ['SE_CNN_PATCHTST', 'SE_CNN_PATCHTST_ONE', 'SE_CNN_PATCHTST_THREE', 'CNN_PATCHTST']:
        model_kwargs.update({
            'patch_size': args.patch_size,
            'stride': args.stride,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'transformer_layers': args.transformer_layers,
            'base_filters': args.filters,
            'se_reduction': args.se_reduction,
            'use_se': args.use_se,
        })
    # 创建模型
    model = create_unified_model(
        model_type=args.model,
        input_shape=input_shape[1:],  # 去掉batch维度
        num_classes=num_classes,
        **model_kwargs
    )
    
    model = model.to(device)
    
    # 打印模型信息
    if is_svm_model(model):
        print(f"SVM模型创建完成")
        print(f"  - 核函数: {args.svm_kernel}")
        print(f"  - C参数: {args.svm_C}")
        print(f"  - Gamma参数: {args.svm_gamma}")
    else:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"神经网络模型创建完成，参数数量: {param_count:,}")
    
    # 如果是SVM且需要网格搜索
    if is_svm_model(model) and args.svm_grid_search:
        print("\n执行SVM网格搜索...")
        best_params = model.grid_search(train_loader)
        print(f"网格搜索完成，最佳参数: {best_params}")
    
    else:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{args.model}模型创建完成，参数数量: {param_count:,}")
        
        # 打印RNN特定信息
        if args.model.upper() in ['LSTM', 'GRU']:
            print(f"  - 隐藏层维度: {args.hidden_dim}")
            print(f"  - 层数: {args.num_layers}")
            print(f"  - 双向: {args.bidirectional}")
            print(f"  - 注意力机制: {args.attention}")
            if args.model.upper() == 'LSTM':
                print(f"  - 模型变体: {args.lstm_variant}")
        elif args.model.upper() in ['TRANSFORMER', 'CNN_TRANSFORMER']:
            print(f"  - 模型维度: {args.d_model}")
            print(f"  - 注意力头数: {args.nhead}")
            print(f"  - Transformer层数: {args.transformer_layers}")
            print(f"  - 前馈网络维度: {args.dim_feedforward}")
            if args.model.upper() == 'CNN_TRANSFORMER':
                print(f"  - CNN滤波器数: {args.cnn_filters}")
                print(f"  - CNN层数: {args.cnn_layers}")
                print(f"  - 池化大小: {args.pool_size}")
        elif args.model.upper() in ['SE_CNN_PATCHTST', 'SE_CNN_PATCHTST_ONE', 'SE_CNN_PATCHTST_THREE', 'CNN_PATCHTST']:
            print(f"  - Patch大小: {args.patch_size}")
            print(f"  - Patch步长: {args.stride}")
            print(f"  - 模型维度: {args.d_model}")
            print(f"  - 注意力头数: {args.nhead}")
            print(f"  - Transformer层数: {args.transformer_layers}")
            print(f"  - CNN滤波器数: {args.filters}")
            if 'SE_CNN' in args.model.upper():
                print(f"  - 使用SE Block: {args.use_se}")
                print(f"  - SE降维比例: {args.se_reduction}")
    # 创建训练器
    trainer = BearingClassificationTrainer(model, device, args.save_dir, args.model)
    
    # 开始训练
    print(f"\n开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    # 绘制训练历史
    print("\n绘制训练曲线...")
    trainer.plot_training_history(
        save_path=os.path.join(args.save_dir, f'{args.model}_training_curves.png')
    )
    
    # 在测试集上评估最佳模型
    print("\n" + "="*60)
    print("在测试集上评估最佳模型")
    print("="*60)
    
    # 加载最佳模型（如果不是SVM）
    if not is_svm_model(model):
        best_model_files = [f for f in os.listdir(args.save_dir) if f.startswith('best_model')]
        if best_model_files:
            best_model_file = sorted(best_model_files)[-1]
            best_model_path = os.path.join(args.save_dir, best_model_file)
            model = load_model(model, best_model_path, device)
            print(f"已加载最佳模型: {best_model_file}")
    else:
        print("SVM模型已完成训练，直接评估")
    
    # 评估模型
    test_accuracy, predictions, labels, cm = evaluate_model(
        model, test_loader, device, class_names, save_results=True, save_dir=args.save_dir
    )
    
    # 保存最终模型
    if is_svm_model(model):
        final_model_path = os.path.join(args.save_dir, f'{args.model}_final_model.joblib')
    else:
        final_model_path = os.path.join(args.save_dir, f'{args.model}_final_model.pth')
    
    save_model(model, final_model_path, {
        'test_accuracy': test_accuracy,
        'num_classes': num_classes,
        'class_names': class_names,
        'training_history': history,
        'model_config': {
            'model_type': args.model,
            'seq_length': args.seq_length,
            'num_classes': num_classes,
            'input_shape': list(input_shape),
            'model_kwargs': model_kwargs
        }
    })
    
    # 保存配置和结果
    results = {
        'model_type': args.model,
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(trainer.best_accuracy),
        'num_classes': num_classes,
        'class_names': class_names,
        'training_config': vars(args)
    }
    
    results_path = os.path.join(args.save_dir, f'{args.model}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练完成！")
    print(f"最佳验证准确率: {trainer.best_accuracy:.4f}")
    print(f"最终测试准确率: {test_accuracy:.4f}")
    print(f"结果已保存到: {args.save_dir}")


def test_workflow(processor, args, device):
    """测试工作流"""
    print("=" * 60)
    print("开始测试工作流")
    print("=" * 60)
    
    # 确定模型文件路径
    if args.model.upper() == 'SVM':
        model_config_path = os.path.join(args.save_dir, f'{args.model}_final_model.joblib')
    else:
        model_config_path = os.path.join(args.save_dir, f'{args.model}_final_model.pth')
    
    if not os.path.exists(model_config_path):
        print(f"错误: 找不到模型文件 {model_config_path}")
        print("请先运行训练模式或指定正确的模型路径")
        return
    
    # 加载模型配置
    if args.model.upper() == 'SVM':
        # SVM模型配置从结果文件加载
        results_path = os.path.join(args.save_dir, f'{args.model}_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                model_config = results.get('training_config', {})
        else:
            print("警告: 找不到SVM配置文件，使用默认配置")
            model_config = {'model_type': 'SVM'}
    else:
        # 神经网络模型配置从checkpoint加载
        checkpoint = torch.load(model_config_path, map_location=device)
        model_config = checkpoint['model_config']
    
    print(f"加载模型配置: {model_config}")
    
    # 准备数据
    if model_config.get('model_type', args.model).upper() == 'MLP':
        mode = 'feature'
    else:
        mode = 'signal'
    
    # 加载测试数据
    train_loader, val_loader, test_loader = processor.get_data_loaders(
        mode=mode,
        batch_size=args.batch_size,
        test_size=0.4,
        random_state=args.random_seed
    )
    
    # 重建模型
    sample_batch = next(iter(test_loader))
    input_shape = sample_batch[0].shape
    
    # 准备模型参数
    model_kwargs = model_config.get('model_kwargs', {
        'dropout_rate': args.dropout,
        'base_filters': args.filters,
    })
    
    model = create_unified_model(
        model_type=model_config.get('model_type', args.model),
        input_shape=input_shape[1:],
        num_classes=model_config.get('num_classes', 38),
        **model_kwargs
    )
    
    # 加载权重
    model = load_model(model, model_config_path, device)
    
    # 评估模型
    if args.model.upper() == 'SVM':
        class_names = None  # SVM结果文件中可能包含类别名称
        if 'results' in locals() and 'class_names' in results:
            class_names = results['class_names']
    else:
        checkpoint = torch.load(model_config_path, map_location=device)
        class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(model_config['num_classes'])])
    
    test_accuracy, predictions, labels, cm = evaluate_model(
        model, test_loader, device, class_names, save_results=True, save_dir=args.save_dir
    )
    
    print(f"\n测试完成！")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 显示训练时的测试准确率（如果可用）
    if args.model.upper() != 'SVM':
        checkpoint = torch.load(model_config_path, map_location=device)
        if 'test_accuracy' in checkpoint:
            print(f"训练时测试准确率: {checkpoint['test_accuracy']:.4f}")


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 显示支持的模型
        supported_models = get_supported_models()
        print(f"支持的模型类型: {supported_models}")
        print(f"当前选择的模型: {args.model}")
        
        # 验证模型选择
        if args.model.upper() not in supported_models:
            print(f"错误: 不支持的模型类型 '{args.model}'")
            print(f"支持的模型: {supported_models}")
            return
        
        # 设置随机种子
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        
        # 设备选择
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        print(f"使用设备: {device}")
        print(f"随机种子: {args.random_seed}")
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 验证数据目录是否存在
        if not os.path.exists(args.data_dir):
            print(f"错误: 数据目录不存在: {args.data_dir}")
            print("请检查数据路径或使用 --data_dir 参数指定正确的路径")
            return
        
        # 创建数据处理器
        try:
            processor = BearingDataProcessor(
                data_dir=args.data_dir,
                seq_length=args.seq_length,
                overlap_ratio=args.overlap_ratio
            )
            print("数据处理器创建成功")
        except Exception as e:
            print(f"错误: 创建数据处理器失败: {e}")
            print("请检查数据目录结构和BearingDataProcessor类的实现")
            return
        
        # 根据模式执行不同工作流
        if args.mode == 'train':
            print(f"\n开始训练模式 - 模型: {args.model}")
            train_workflow(processor, args, device)
        elif args.mode == 'test':
            print(f"\n开始测试模式 - 模型: {args.model}")
            test_workflow(processor, args, device)
        else:
            print(f"错误: 未知的运行模式: {args.mode}")
            print("支持的模式: train, test")
            return
            
        print("\n程序执行完成！")
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except ImportError as e:
        print(f"\n导入错误: {e}")
        print("请确保以下文件存在:")
        print("  - data_loader.py (包含 BearingDataProcessor)")
        print("  - models.py (包含 MLPClassifier, CNNClassifier)")
        print("  - ltsf_linear_model.py (包含 LTSFLinearClassifier)")
        print("  - svm_model.py (包含 SVMClassifier)")
        print("  - rnn_models.py (包含 LSTMClassifier, GRUClassifier)")
        print("  - transformer_models.py (包含 TransformerClassifier, CNNTransformerClassifier)")
        print("  - se_cnn_patchtst_models.py (包含 SE-CNN-PatchTST系列模型)")

    except Exception as e:
        print(f"\n运行时错误: {e}")
        print("请检查配置和数据文件")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
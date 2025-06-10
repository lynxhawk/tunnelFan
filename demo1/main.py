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
from models import MLPClassifier, CNNClassifier, create_model


class BearingClassificationTrainer:
    """轴承故障分类训练器"""
    
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
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
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
        """验证一个epoch"""
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
    
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001,
              weight_decay=1e-5, patience=15, save_best=True):
        """完整训练流程"""
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        print(f"开始训练，共 {num_epochs} 个epoch...")
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
    
    def save_model(self, filename):
        """保存模型"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }, save_path)
    
    def load_model(self, filename):
        """加载模型"""
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', {})
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.training_history['train_loss'], label='Training Loss', color='blue')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.training_history['train_acc'], label='Training Accuracy', color='blue')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_model(model, test_loader, device, class_names=None, save_results=True):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_latent_features = []
    
    print("正在评估模型...")
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
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"测试准确率: {accuracy:.4f}")
    
    # 分类报告
    if class_names and len(class_names) <= 30:  # 扩展显示限制
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=class_names, digits=4))
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 可视化混淆矩阵
    visualize_confusion_matrix(cm, class_names, save_results)
    
    # t-SNE可视化
    if all_latent_features:
        visualize_tsne(np.array(all_latent_features), np.array(all_labels), 
                      class_names, save_results)
    
    return accuracy, all_predictions, all_labels, cm


def visualize_confusion_matrix(cm, class_names=None, save_results=True):
    """可视化混淆矩阵"""
    plt.figure(figsize=(15, 12))
    
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
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_tsne(latent_features, labels, class_names=None, save_results=True, 
                   max_samples=5000):
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
        plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_model(model, filepath, additional_info=None):
    """保存模型"""
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
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='CNN', choices=['MLP', 'CNN'],
                       help='选择模型类型')
    parser.add_argument('--filters', type=int, default=64,
                       help='CNN滤波器数量')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比率')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=15,
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
    
    # 创建模型
    print(f"\n创建 {args.model} 模型...")
    if args.model.upper() == 'MLP':
        feature_dim = input_shape[1]
        model = MLPClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_layers=[512, 256, 128, 64],
            dropout_rate=args.dropout
        )
    elif args.model.upper() == 'CNN':
        model = CNNClassifier(
            input_channels=input_shape[2],
            seq_length=input_shape[1],
            num_classes=num_classes,
            base_filters=args.filters,
            dropout_rate=args.dropout
        )
    
    model = model.to(device)
    print(f"模型创建完成，参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建训练器
    trainer = BearingClassificationTrainer(model, device, args.save_dir)
    
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
    
    # 加载最佳模型
    best_model_files = [f for f in os.listdir(args.save_dir) if f.startswith('best_model')]
    if best_model_files:
        best_model_file = sorted(best_model_files)[-1]
        best_model_path = os.path.join(args.save_dir, best_model_file)
        model = load_model(model, best_model_path, device)
        print(f"已加载最佳模型: {best_model_file}")
    
    # 评估模型
    test_accuracy, predictions, labels, cm = evaluate_model(
        model, test_loader, device, class_names[:20] if len(class_names) > 20 else class_names
    )
    
    # 保存最终模型
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
            'input_shape': list(input_shape)
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
    
    # 加载模型配置
    model_config_path = os.path.join(args.save_dir, f'{args.model}_final_model.pth')
    if not os.path.exists(model_config_path):
        print(f"错误: 找不到模型文件 {model_config_path}")
        print("请先运行训练模式或指定正确的模型路径")
        return
    
    # 加载模型
    checkpoint = torch.load(model_config_path, map_location=device)
    model_config = checkpoint['model_config']
    
    print(f"加载模型配置: {model_config}")
    
    # 准备数据
    if model_config['model_type'].upper() == 'MLP':
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
    if model_config['model_type'].upper() == 'MLP':
        sample_batch = next(iter(test_loader))
        feature_dim = sample_batch[0].shape[1]
        model = MLPClassifier(
            input_dim=feature_dim,
            num_classes=model_config['num_classes'],
            hidden_layers=[512, 256, 128, 64],
            dropout_rate=args.dropout
        )
    elif model_config['model_type'].upper() == 'CNN':
        model = CNNClassifier(
            input_channels=model_config['input_shape'][2],
            seq_length=model_config['input_shape'][1],
            num_classes=model_config['num_classes'],
            base_filters=args.filters,
            dropout_rate=args.dropout
        )
    
    # 加载权重
    model = load_model(model, model_config_path, device)
    
    # 评估模型
    class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(model_config['num_classes'])])
    test_accuracy, predictions, labels, cm = evaluate_model(
        model, test_loader, device, class_names[:20] if len(class_names) > 20 else class_names
    )
    
    print(f"\n测试完成！")
    print(f"测试准确率: {test_accuracy:.4f}")
    if 'test_accuracy' in checkpoint:
        print(f"训练时测试准确率: {checkpoint['test_accuracy']:.4f}")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
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
    
    # 创建数据处理器
    processor = BearingDataProcessor(
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        overlap_ratio=args.overlap_ratio
    )
    
    # 根据模式执行不同工作流
    if args.mode == 'train':
        train_workflow(processor, args, device)
    elif args.mode == 'test':
        test_workflow(processor, args, device)


if __name__ == "__main__":
    main()
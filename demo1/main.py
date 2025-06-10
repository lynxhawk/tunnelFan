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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
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


def evaluate_model(model, test_loader, device, class_names=None):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("正在评估模型...")
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            outputs, _ = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"测试准确率: {accuracy:.4f}")
    
    # 分类报告
    if class_names and len(class_names) <= 20:  # 只有类别不太多时才打印详细报告
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=class_names, digits=4))
    
    # 混淆矩阵（仅在类别数较少时可视化）
    if class_names and len(class_names) <= 20:
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    return accuracy, all_predictions, all_labels


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='轴承故障诊断训练')
    parser.add_argument('--model', type=str, default='CNN', choices=['MLP', 'CNN'],
                       help='选择模型类型')
    parser.add_argument('--data_dir', type=str, default='bearing_dataset',
                       help='数据集目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--seq_length', type=int, default=1000,
                       help='序列长度')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备选择: auto, cpu, cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 数据准备
    print("准备数据...")
    processor = BearingDataProcessor(
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        overlap_ratio=0.5
    )
    
    # 根据模型类型选择数据模式
    if args.model.upper() == 'MLP':
        mode = 'feature'
    else:
        mode = 'signal'
    
    train_loader, test_loader = processor.get_data_loaders(
        mode=mode,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=42
    )
    
    # 获取类别信息
    class_info = processor.get_class_info()
    num_classes = class_info['num_classes']
    class_names = class_info['class_names']
    
    print(f"数据加载完成，共 {num_classes} 个类别")
    
    # 模型创建
    print(f"创建 {args.model} 模型...")
    if args.model.upper() == 'MLP':
        # 获取特征维度
        sample_batch = next(iter(train_loader))
        feature_dim = sample_batch[0].shape[1]
        model = MLPClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_layers=[512, 256, 128, 64],
            dropout_rate=0.3
        )
    elif args.model.upper() == 'CNN':
        model = CNNClassifier(
            input_channels=1,
            seq_length=args.seq_length,
            num_classes=num_classes,
            base_filters=64,
            dropout_rate=0.3
        )
    
    model = model.to(device)
    
    # 训练器初始化
    trainer = BearingClassificationTrainer(model, device, args.save_dir)
    
    # 开始训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # 这里用test_loader作为验证集
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-5,
        patience=15
    )
    
    # 绘制训练历史
    trainer.plot_training_history(
        save_path=os.path.join(args.save_dir, f'{args.model}_training_history.png')
    )
    
    # 最终评估
    print("\n" + "="*50)
    print("最终模型评估")
    print("="*50)
    
    # 加载最佳模型进行评估
    best_model_files = [f for f in os.listdir(args.save_dir) if f.startswith('best_model')]
    if best_model_files:
        best_model_file = sorted(best_model_files)[-1]  # 选择最新的最佳模型
        trainer.load_model(best_model_file)
        print(f"加载最佳模型: {best_model_file}")
    
    # 评估模型
    accuracy, predictions, labels = evaluate_model(
        model, test_loader, device, 
        class_names if len(class_names) <= 20 else None
    )
    
    # 保存结果
    results = {
        'model_type': args.model,
        'test_accuracy': accuracy,
        'num_classes': num_classes,
        'best_val_accuracy': trainer.best_accuracy,
        'class_names': class_names
    }
    
    import json
    with open(os.path.join(args.save_dir, f'{args.model}_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练完成！结果已保存到 {args.save_dir}")
    print(f"最佳验证准确率: {trainer.best_accuracy:.4f}")
    print(f"最终测试准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()
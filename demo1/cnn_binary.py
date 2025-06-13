import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import time
import os


class ThreeLayerCNN(nn.Module):
    """三层CNN二分类模型 - 专门用于轴承故障检测"""
    
    def __init__(self, input_length=1000, input_channels=1, num_classes=2, dropout_rate=0.3):
        """
        初始化三层CNN模型
        
        参数:
        - input_length: 输入序列长度
        - input_channels: 输入通道数
        - num_classes: 分类数量（二分类=2）
        - dropout_rate: Dropout比例
        """
        super(ThreeLayerCNN, self).__init__()
        
        self.input_length = input_length
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, 
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, 
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 第三层卷积
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, 
                              kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 计算全连接层的输入维度
        self.feature_dim = self._get_conv_output_size()
        
        # 全连接层
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_conv_output_size(self):
        """计算卷积层输出的特征维度"""
        # 模拟数据通过卷积层
        dummy_input = torch.randn(1, self.input_channels, self.input_length)
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 调整输入维度：从 (batch, seq_len, channels) 到 (batch, channels, seq_len)
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1).unsqueeze(1)  # (batch, 1, seq_len)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)
        
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'feature_dim': self.feature_dim
        }


class BinaryBearingTrainer:
    """二分类轴承故障检测训练器"""
    
    def __init__(self, model, device='auto', learning_rate=0.001, weight_decay=1e-4):
        """
        初始化训练器
        
        参数:
        - model: CNN模型
        - device: 计算设备
        - learning_rate: 学习率
        - weight_decay: 权重衰减
        """
        # 自动选择设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 损失函数（处理类别不平衡）
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def calculate_class_weights(self, train_loader):
        """计算类别权重以处理数据不平衡"""
        class_counts = torch.zeros(2)
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        
        total_samples = class_counts.sum()
        class_weights = total_samples / (2 * class_counts)
        
        print(f"📊 类别分布:")
        print(f"  正常样本: {int(class_counts[0]):,} ({class_counts[0]/total_samples*100:.1f}%)")
        print(f"  故障样本: {int(class_counts[1]):,} ({class_counts[1]/total_samples*100:.1f}%)")
        print(f"  类别权重: {class_weights.numpy()}")
        
        return class_weights.to(self.device)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_data, batch_labels in progress_bar:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_labels.size(0)
            correct_predictions += (predicted == batch_labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples*100:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            
            for batch_data, batch_labels in progress_bar:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_labels.size(0)
                correct_predictions += (predicted == batch_labels).sum().item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct_predictions/total_samples*100:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15,
              save_best_model=True, model_save_path='best_binary_model.pth'):
        """
        训练模型
        
        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - epochs: 训练轮数
        - early_stopping_patience: 早停耐心值
        - save_best_model: 是否保存最佳模型
        - model_save_path: 模型保存路径
        """
        
        print("🚀 开始训练二分类CNN模型")
        print("=" * 60)
        
        # 计算类别权重
        class_weights = self.calculate_class_weights(train_loader)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 显示模型信息
        model_info = self.model.get_model_info()
        print(f"📋 模型信息:")
        print(f"  总参数数量: {model_info['total_parameters']:,}")
        print(f"  可训练参数: {model_info['trainable_parameters']:,}")
        print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
        print(f"  特征维度: {model_info['feature_dim']}")
        
        start_time = time.time()
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0
                
                if save_best_model:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch,
                        'model_info': model_info
                    }, model_save_path)
            else:
                early_stopping_counter += 1
            
            # 显示进度
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # 早停检查
            if early_stopping_counter >= early_stopping_patience:
                print(f"\n⏹️  早停触发！验证精度在 {early_stopping_patience} 个epoch内未改善")
                break
        
        # 训练结束
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成！")
        print(f"📊 训练统计:")
        print(f"  总训练时间: {total_time/60:.1f} 分钟")
        print(f"  最佳验证精度: {self.best_val_acc*100:.2f}%")
        print(f"  最终学习率: {current_lr:.2e}")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ 已加载最佳模型权重")
    
    def evaluate(self, test_loader, class_names=['Normal', 'Fault']):
        """
        评估模型性能
        
        参数:
        - test_loader: 测试数据加载器
        - class_names: 类别名称
        """
        print("\n🧪 开始模型评估...")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        test_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Testing")
            
            for batch_data, batch_labels in progress_bar:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # 计算指标
        test_loss /= len(test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\n📊 测试结果:")
        print(f"  测试损失: {test_loss:.4f}")
        print(f"  准确率: {accuracy*100:.2f}%")
        print(f"  精确率: {precision*100:.2f}%")
        print(f"  召回率: {recall*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
        
        # 详细分类报告
        print(f"\n📋 详细分类报告:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm, class_names)
        
        return {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'true_labels': all_labels,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(self.train_history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.train_history['val_loss'], label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot([acc*100 for acc in self.train_history['train_acc']], 
                label='Train Acc', color='blue')
        ax2.plot([acc*100 for acc in self.train_history['val_acc']], 
                label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(self.train_history['learning_rates'], color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 验证精度改善
        best_val_acc = [max(self.train_history['val_acc'][:i+1]) 
                       for i in range(len(self.train_history['val_acc']))]
        ax4.plot([acc*100 for acc in best_val_acc], color='purple')
        ax4.set_title('Best Validation Accuracy Progress')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Best Val Accuracy (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_sample(self, data):
        """预测单个样本"""
        self.model.eval()
        
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        elif data.dim() == 2:
            data = data.unsqueeze(-1)  # (batch, seq_len, 1)
        
        data = data.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return {
            'prediction': predicted.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'confidence': torch.max(probabilities, 1)[0].cpu().numpy()
        }


# 完整的训练和评估流程
def complete_binary_classification_pipeline():
    """完整的二分类流程"""
    
    print("🎯 轴承故障二分类 - 完整流程")
    print("=" * 60)
    
    # 1. 数据准备
    from data_binary import BinaryBearingDataProcessor
    
    processor = BinaryBearingDataProcessor(
        dataset1_dir='bearing_dataset',    # HUST数据集
        dataset2_dir='bearing_dataset1',   # 另一个数据集
        seq_length=1000,
        overlap_ratio=0.5
    )
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = processor.get_optimized_data_loaders(
        train_dataset='dataset1',  # 在HUST数据集上训练
        test_dataset='dataset2',   # 在dataset2上测试
        batch_size=64,
        max_train_files=200,       # 可以根据需要调整
        max_test_files=100,
        val_split=0.2
    )
    
    # 2. 模型初始化
    model = ThreeLayerCNN(
        input_length=1000,
        input_channels=1,
        num_classes=2,
        dropout_rate=0.3
    )
    
    # 3. 训练器初始化
    trainer = BinaryBearingTrainer(
        model=model,
        device='auto',
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    # 4. 训练模型
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_best_model=True,
        model_save_path='best_binary_cnn_model.pth'
    )
    
    # 5. 绘制训练历史
    trainer.plot_training_history()
    
    # 6. 测试评估
    test_results = trainer.evaluate(test_loader, class_names=['Normal', 'Fault'])
    
    # 7. 显示性能摘要
    processor.print_performance_summary()
    
    return trainer, test_results, processor


if __name__ == "__main__":
    # 运行完整流程
    trainer, results, processor = complete_binary_classification_pipeline()
    
    print(f"\n🎉 二分类模型训练和评估完成！")
    print(f"📊 最终测试精度: {results['accuracy']*100:.2f}%")
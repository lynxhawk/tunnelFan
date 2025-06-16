import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import time
import os


class FeatureExtractor(nn.Module):
    """特征提取模块 - 基于CNN"""
    
    def __init__(self, input_length=1000, input_channels=1, feature_dim=256, dropout_rate=0.3):
        super(FeatureExtractor, self).__init__()
        
        self.input_length = input_length
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # CNN特征提取层
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 计算卷积输出维度
        self.conv_output_size = self._get_conv_output_size()
        
        # 特征映射层
        self.feature_mapping = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self):
        """计算卷积层输出维度"""
        dummy_input = torch.randn(1, self.input_channels, self.input_length)
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """初始化权重"""
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
        # 调整输入维度
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN特征提取
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        
        # 展平并映射到特征空间
        x = x.view(x.size(0), -1)
        features = self.feature_mapping(x)
        
        return features


class Classifier(nn.Module):
    """分类器模块"""
    
    def __init__(self, feature_dim=256, num_classes=2, dropout_rate=0.3):
        super(Classifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """前向传播"""
        return self.classifier(features)


class DomainDiscriminator(nn.Module):
    """域判别器 - 用于区分两个特征提取器的输出"""
    
    def __init__(self, feature_dim=256, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """前向传播"""
        return self.discriminator(features)


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层 - 用于对抗训练"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def gradient_reversal(x, alpha=1.0):
    """梯度反转函数"""
    return GradientReversalLayer.apply(x, alpha)


class DomainAdaptationTrainer:
    """域适应训练器"""
    
    def __init__(self, source_feature_extractor, target_feature_extractor, 
                 classifier, discriminator, device='auto'):
        """
        初始化域适应训练器
        
        参数:
        - source_feature_extractor: 源域（HUST）特征提取器
        - target_feature_extractor: 目标域（CWRU）特征提取器  
        - classifier: 分类器
        - discriminator: 域判别器
        - device: 计算设备
        """
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        # 模型组件
        self.source_fe = source_feature_extractor.to(self.device)
        self.target_fe = target_feature_extractor.to(self.device)
        self.classifier = classifier.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        # 损失函数
        self.classification_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.BCELoss()
        
        # 优化器
        self.optimizer_target_fe = optim.Adam(self.target_fe.parameters(), lr=0.001)
        self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=0.001)
        
        # 训练历史
        self.train_history = {
            'classification_loss': [],
            'domain_loss': [],
            'target_fe_loss': [],
            'discriminator_accuracy': []
        }
        
        self.best_domain_confusion = 0.0  # 域混淆度（越接近0.5越好）
    
    def pretrain_source_model(self, source_loader, epochs=50):
        """
        预训练源域模型（HUST数据集）
        
        参数:
        - source_loader: 源域数据加载器
        - epochs: 预训练轮数
        """
        print("🚀 开始预训练源域模型...")
        
        # 冻结源域特征提取器的预训练
        optimizer_source = optim.Adam(
            list(self.source_fe.parameters()) + list(self.classifier.parameters()), 
            lr=0.001
        )
        
        self.source_fe.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(source_loader, desc=f"预训练 Epoch {epoch+1}/{epochs}")
            
            for batch_data, batch_labels in progress_bar:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer_source.zero_grad()
                
                # 特征提取和分类
                features = self.source_fe(batch_data)
                outputs = self.classifier(features)
                loss = self.classification_loss(outputs, batch_labels)
                
                loss.backward()
                optimizer_source.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct/total*100:.2f}%'
                })
            
            avg_loss = epoch_loss / len(source_loader)
            accuracy = correct / total
            
            print(f"预训练 Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Acc: {accuracy*100:.2f}%")
        
        # 冻结源域特征提取器
        for param in self.source_fe.parameters():
            param.requires_grad = False
        
        print("✅ 源域模型预训练完成，特征提取器已冻结")
    
    def train_domain_adaptation(self, source_loader, target_loader, epochs=100, 
                              lambda_domain=1.0, alpha_schedule=None):
        """
        训练域适应模型
        
        参数:
        - source_loader: 源域数据加载器（有标签）
        - target_loader: 目标域数据加载器（无标签）
        - epochs: 训练轮数
        - lambda_domain: 域损失权重
        - alpha_schedule: 梯度反转强度调度
        """
        print("🔄 开始域适应训练...")
        
        self.target_fe.train()
        self.classifier.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            # 计算梯度反转强度
            if alpha_schedule is None:
                alpha = min(1.0, 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0)
            else:
                alpha = alpha_schedule(epoch, epochs)
            
            epoch_cls_loss = 0.0
            epoch_domain_loss = 0.0
            epoch_target_loss = 0.0
            
            domain_correct = 0
            domain_total = 0
            cls_correct = 0
            cls_total = 0
            
            # 创建数据迭代器
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            
            max_batches = min(len(source_loader), len(target_loader))
            progress_bar = tqdm(range(max_batches), desc=f"DA Epoch {epoch+1}/{epochs}")
            
            for batch_idx in progress_bar:
                try:
                    source_data, source_labels = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_data, source_labels = next(source_iter)
                
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)
                
                batch_size = min(source_data.size(0), target_data.size(0))
                source_data = source_data[:batch_size]
                source_labels = source_labels[:batch_size]
                target_data = target_data[:batch_size]
                
                # ============= 训练判别器 =============
                self.optimizer_discriminator.zero_grad()
                
                # 源域特征（标签=1）
                with torch.no_grad():
                    source_features = self.source_fe(source_data)
                target_features = self.target_fe(target_data).detach()
                
                # 域标签
                source_domain_labels = torch.ones(batch_size, 1).to(self.device)
                target_domain_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # 判别器预测
                source_domain_pred = self.discriminator(source_features)
                target_domain_pred = self.discriminator(target_features)
                
                # 判别器损失
                domain_loss_d = (self.domain_loss(source_domain_pred, source_domain_labels) + 
                               self.domain_loss(target_domain_pred, target_domain_labels)) / 2
                
                domain_loss_d.backward()
                self.optimizer_discriminator.step()
                
                # ============= 训练目标特征提取器和分类器 =============
                self.optimizer_target_fe.zero_grad()
                self.optimizer_classifier.zero_grad()
                
                # 分类损失（源域）
                source_features = self.source_fe(source_data)
                source_pred = self.classifier(source_features)
                cls_loss = self.classification_loss(source_pred, source_labels)
                
                # 域对抗损失（目标域）
                target_features = self.target_fe(target_data)
                reversed_target_features = gradient_reversal(target_features, alpha)
                target_domain_pred = self.discriminator(reversed_target_features)
                
                # 目标是让判别器无法区分（标签=1，即希望被误认为源域）
                target_domain_labels_adv = torch.ones(batch_size, 1).to(self.device)
                domain_loss_g = self.domain_loss(target_domain_pred, target_domain_labels_adv)
                
                # 总损失
                total_loss = cls_loss + lambda_domain * domain_loss_g
                
                total_loss.backward()
                self.optimizer_target_fe.step()
                self.optimizer_classifier.step()
                
                # 统计
                epoch_cls_loss += cls_loss.item()
                epoch_domain_loss += domain_loss_d.item()
                epoch_target_loss += domain_loss_g.item()
                
                # 分类准确率
                _, cls_predicted = torch.max(source_pred.data, 1)
                cls_total += source_labels.size(0)
                cls_correct += (cls_predicted == source_labels).sum().item()
                
                # 域判别准确率
                domain_pred_binary = torch.cat([
                    (source_domain_pred > 0.5).float(),
                    (target_domain_pred > 0.5).float()
                ])
                domain_true_binary = torch.cat([
                    torch.ones(batch_size, 1),
                    torch.zeros(batch_size, 1)
                ]).to(self.device)
                
                domain_total += domain_true_binary.size(0)
                domain_correct += (domain_pred_binary == domain_true_binary).sum().item()
                
                progress_bar.set_postfix({
                    'Cls_Loss': f'{cls_loss.item():.4f}',
                    'Dom_Loss': f'{domain_loss_d.item():.4f}',
                    'Cls_Acc': f'{cls_correct/cls_total*100:.2f}%',
                    'Dom_Acc': f'{domain_correct/domain_total*100:.2f}%',
                    'Alpha': f'{alpha:.3f}'
                })
            
            # 记录历史
            self.train_history['classification_loss'].append(epoch_cls_loss / max_batches)
            self.train_history['domain_loss'].append(epoch_domain_loss / max_batches)
            self.train_history['target_fe_loss'].append(epoch_target_loss / max_batches)
            self.train_history['discriminator_accuracy'].append(domain_correct / domain_total)
            
            # 检查域混淆程度
            domain_confusion = abs(domain_correct / domain_total - 0.5)  # 越接近0越好
            if domain_confusion < self.best_domain_confusion or self.best_domain_confusion == 0:
                self.best_domain_confusion = domain_confusion
                self.save_best_model(epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  分类损失: {epoch_cls_loss/max_batches:.4f}")
            print(f"  域损失: {epoch_domain_loss/max_batches:.4f}")
            print(f"  分类准确率: {cls_correct/cls_total*100:.2f}%")
            print(f"  域判别准确率: {domain_correct/domain_total*100:.2f}% (目标: 50%)")
            print(f"  域混淆度: {domain_confusion:.4f} (越小越好)")
            print(f"  梯度反转强度: {alpha:.3f}")
    
    def save_best_model(self, epoch):
        """保存最佳模型"""
        torch.save({
            'epoch': epoch,
            'target_fe_state_dict': self.target_fe.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'best_domain_confusion': self.best_domain_confusion,
            'train_history': self.train_history
        }, 'best_domain_adaptation_model.pth')
        
        print(f"💾 保存最佳模型 (Epoch {epoch+1}, 域混淆度: {self.best_domain_confusion:.4f})")
    
    def evaluate_target_domain(self, target_loader_with_labels, class_names=['Normal', 'Fault']):
        """
        评估目标域性能（如果有标签的话）
        
        参数:
        - target_loader_with_labels: 带标签的目标域数据加载器
        - class_names: 类别名称
        """
        print("🧪 评估目标域性能...")
        
        self.target_fe.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(target_loader_with_labels, desc="评估"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 使用目标域特征提取器
                features = self.target_fe(batch_data)
                outputs = self.classifier(features)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\n📊 目标域测试结果:")
        print(f"  准确率: {accuracy*100:.2f}%")
        print(f"  精确率: {precision*100:.2f}%")
        print(f"  召回率: {recall*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\n📋 混淆矩阵:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'true_labels': all_labels,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """绘制训练历史"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(self.train_history['classification_loss']) + 1)
            
            # 分类损失
            ax1.plot(epochs, self.train_history['classification_loss'], 'b-', label='Classification Loss')
            ax1.set_title('Classification Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # 域损失
            ax2.plot(epochs, self.train_history['domain_loss'], 'r-', label='Domain Loss (Discriminator)')
            ax2.plot(epochs, self.train_history['target_fe_loss'], 'g-', label='Domain Loss (Generator)')
            ax2.set_title('Domain Adaptation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            # 域判别准确率
            ax3.plot(epochs, [acc*100 for acc in self.train_history['discriminator_accuracy']], 'purple')
            ax3.axhline(y=50, color='red', linestyle='--', label='Target (50%)')
            ax3.set_title('Domain Discriminator Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy (%)')
            ax3.legend()
            ax3.grid(True)
            
            # 域混淆度
            domain_confusion = [abs(acc - 0.5) for acc in self.train_history['discriminator_accuracy']]
            ax4.plot(epochs, domain_confusion, 'orange')
            ax4.set_title('Domain Confusion (Lower is Better)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Confusion (|Acc - 0.5|)')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig('domain_adaptation_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"⚠️  无法生成训练历史图: {e}")


# 使用示例和完整流程
def complete_domain_adaptation_pipeline():
    """完整的域适应流程"""
    
    print("🎯 基于GAN的轴承故障域适应系统")
    print("=" * 60)
    print("📋 系统架构:")
    print("  - 源域: HUST数据集 (有标签)")
    print("  - 目标域: CWRU数据集 (无标签)")
    print("  - 方法: 对抗域适应 (Adversarial Domain Adaptation)")
    print("=" * 60)
    
    # 1. 数据准备
    print(f"\n{'='*20} 步骤1: 数据准备 {'='*20}")
    
    # 这里需要你的数据加载器
    # source_loader = 你的HUST数据加载器
    # target_loader = 你的CWRU数据加载器
    # target_loader_with_labels = 你的CWRU测试数据加载器（如果有标签）
    
    # 2. 模型初始化
    print(f"\n{'='*20} 步骤2: 模型初始化 {'='*20}")
    
    feature_dim = 256
    
    source_feature_extractor = FeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim
    )
    
    target_feature_extractor = FeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim
    )
    
    classifier = Classifier(
        feature_dim=feature_dim,
        num_classes=2
    )
    
    discriminator = DomainDiscriminator(
        feature_dim=feature_dim
    )
    
    # 3. 初始化训练器
    print(f"\n{'='*20} 步骤3: 训练器初始化 {'='*20}")
    
    trainer = DomainAdaptationTrainer(
        source_feature_extractor=source_feature_extractor,
        target_feature_extractor=target_feature_extractor,
        classifier=classifier,
        discriminator=discriminator,
        device='auto'
    )
    
    print("✅ 域适应系统初始化完成")
    print("🔧 模型组件:")
    print(f"  - 源域特征提取器: {sum(p.numel() for p in source_feature_extractor.parameters()):,} 参数")
    print(f"  - 目标域特征提取器: {sum(p.numel() for p in target_feature_extractor.parameters()):,} 参数")
    print(f"  - 分类器: {sum(p.numel() for p in classifier.parameters()):,} 参数")
    print(f"  - 域判别器: {sum(p.numel() for p in discriminator.parameters()):,} 参数")
    
    return trainer


if __name__ == "__main__":
    # 运行完整流程
    trainer = complete_domain_adaptation_pipeline()
    
    print(f"\n🎉 域适应系统创建完成！")
    print(f"📝 使用步骤:")
    print(f"  1. 准备数据加载器")
    print(f"  2. 调用 trainer.pretrain_source_model(source_loader)")
    print(f"  3. 调用 trainer.train_domain_adaptation(source_loader, target_loader)")
    print(f"  4. 调用 trainer.evaluate_target_domain(target_test_loader)")
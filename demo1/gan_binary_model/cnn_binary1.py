import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class ImprovedFeatureExtractor(nn.Module):
    """改进的特征提取器 - 增加稳定性"""
    
    def __init__(self, input_length=1000, input_channels=1, feature_dim=256, dropout_rate=0.3):
        super(ImprovedFeatureExtractor, self).__init__()
        
        self.input_length = input_length
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # 改进的CNN架构 - 使用更小的卷积核和更多层
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.AdaptiveAvgPool1d(8)  # 自适应池化
        
        # 全局特征映射
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征映射层 - 添加残差连接思想
        self.feature_mapping = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN特征提取
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # 全局池化
        x = self.global_pool(x).squeeze(-1)
        
        # 特征映射
        features = self.feature_mapping(x)
        
        return features


class ImprovedDomainDiscriminator(nn.Module):
    """改进的域判别器 - 增加正则化"""
    
    def __init__(self, feature_dim=256, hidden_dim=128):
        super(ImprovedDomainDiscriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim//2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """前向传播"""
        return torch.sigmoid(self.discriminator(features))


class StableGradientReversalLayer(torch.autograd.Function):
    """稳定的梯度反转层"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 添加梯度裁剪
        grad_output = torch.clamp(grad_output, -1.0, 1.0)
        output = grad_output.neg() * ctx.alpha
        return output, None


def stable_gradient_reversal(x, alpha=1.0):
    """稳定的梯度反转函数"""
    return StableGradientReversalLayer.apply(x, alpha)


class ImprovedDomainAdaptationTrainer:
    """改进的域适应训练器 - 增加训练稳定性"""
    
    def __init__(self, source_feature_extractor, target_feature_extractor, 
                 classifier, discriminator, device='auto'):
        
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
        
        # 改进的优化器配置
        self.optimizer_target_fe = optim.Adam(
            self.target_fe.parameters(), 
            lr=0.0001,  # 降低学习率
            betas=(0.5, 0.999),  # 调整momentum
            weight_decay=1e-5
        )
        self.optimizer_classifier = optim.Adam(
            self.classifier.parameters(), 
            lr=0.0001,
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(), 
            lr=0.0002,  # 判别器学习率稍高
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler_target_fe = StepLR(self.optimizer_target_fe, step_size=30, gamma=0.5)
        self.scheduler_classifier = StepLR(self.optimizer_classifier, step_size=30, gamma=0.5)
        self.scheduler_discriminator = StepLR(self.optimizer_discriminator, step_size=30, gamma=0.7)
        
        # 训练历史
        self.train_history = {
            'classification_loss': [],
            'domain_loss_d': [],
            'domain_loss_g': [],
            'discriminator_accuracy': [],
            'classification_accuracy': []
        }
        
        self.best_domain_confusion = float('inf')
        
        # 训练参数
        self.warmup_epochs = 10  # 预热轮数
        self.lambda_schedule = 'progressive'  # 渐进式权重调度
    
    def pretrain_source_model(self, source_loader, epochs=50):
        """预训练源域模型"""
        print("🚀 开始预训练源域模型...")
        
        # 源域优化器
        optimizer_source = optim.Adam(
            list(self.source_fe.parameters()) + list(self.classifier.parameters()), 
            lr=0.001, weight_decay=1e-4
        )
        scheduler_source = StepLR(optimizer_source, step_size=20, gamma=0.5)
        
        self.source_fe.train()
        self.classifier.train()
        
        best_acc = 0.0
        
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
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.source_fe.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                
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
            
            scheduler_source.step()
            
            avg_loss = epoch_loss / len(source_loader)
            accuracy = correct / total
            
            if accuracy > best_acc:
                best_acc = accuracy
                # 保存最佳源域模型
                torch.save({
                    'source_fe_state_dict': self.source_fe.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                    'accuracy': accuracy
                }, 'best_source_model.pth')
            
            print(f"预训练 Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Acc: {accuracy*100:.2f}%, Best: {best_acc*100:.2f}%")
        
        # 冻结源域特征提取器
        for param in self.source_fe.parameters():
            param.requires_grad = False
        
        print(f"✅ 源域模型预训练完成，最佳准确率: {best_acc*100:.2f}%")
    
    def get_lambda_schedule(self, epoch, total_epochs):
        """获取域损失权重调度"""
        if self.lambda_schedule == 'progressive':
            # 渐进式增加
            p = float(epoch) / total_epochs
            return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        elif self.lambda_schedule == 'constant':
            return 1.0
        elif self.lambda_schedule == 'decay':
            return np.exp(-epoch / total_epochs)
        else:
            return 1.0
    
    def get_alpha_schedule(self, epoch, total_epochs):
        """获取梯度反转强度调度"""
        p = float(epoch) / total_epochs
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
    
    def train_domain_adaptation_improved(self, source_loader, target_loader, epochs=100):
        """改进的域适应训练"""
        print("🔄 开始改进的域适应训练...")
        
        self.target_fe.train()
        self.classifier.train()
        self.discriminator.train()
        
        # 训练统计
        d_losses = []
        g_losses = []
        cls_losses = []
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 获取调度参数
            lambda_domain = self.get_lambda_schedule(epoch, epochs)
            alpha = self.get_alpha_schedule(epoch, epochs)
            
            # 是否在预热期
            is_warmup = epoch < self.warmup_epochs
            
            epoch_cls_loss = 0.0
            epoch_domain_loss_d = 0.0
            epoch_domain_loss_g = 0.0
            
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
                
                # ============= 阶段1: 训练判别器 =============
                # 多步训练判别器以增强稳定性
                d_steps = 2 if not is_warmup else 1
                
                for _ in range(d_steps):
                    self.optimizer_discriminator.zero_grad()
                    
                    # 源域特征（真实）
                    with torch.no_grad():
                        source_features = self.source_fe(source_data)
                        target_features = self.target_fe(target_data)
                    
                    # 添加噪声增强稳定性
                    noise_std = 0.1 if not is_warmup else 0.05
                    source_features_noisy = source_features + torch.randn_like(source_features) * noise_std
                    target_features_noisy = target_features + torch.randn_like(target_features) * noise_std
                    
                    # 域标签 - 使用标签平滑
                    source_domain_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
                    target_domain_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1
                    
                    # 判别器预测
                    source_domain_pred = self.discriminator(source_features_noisy)
                    target_domain_pred = self.discriminator(target_features_noisy)
                    
                    # 判别器损失
                    domain_loss_d = (
                        self.domain_loss(source_domain_pred, source_domain_labels) + 
                        self.domain_loss(target_domain_pred, target_domain_labels)
                    ) / 2
                    
                    domain_loss_d.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    
                    self.optimizer_discriminator.step()
                
                # ============= 阶段2: 训练生成器（特征提取器+分类器） =============
                self.optimizer_target_fe.zero_grad()
                self.optimizer_classifier.zero_grad()
                
                # 分类损失（源域）
                source_features = self.source_fe(source_data)
                source_pred = self.classifier(source_features)
                cls_loss = self.classification_loss(source_pred, source_labels)
                
                # 域对抗损失（目标域）
                if not is_warmup:  # 预热期不进行域对抗
                    target_features = self.target_fe(target_data)
                    
                    # 稳定的梯度反转
                    reversed_target_features = stable_gradient_reversal(target_features, alpha)
                    target_domain_pred = self.discriminator(reversed_target_features)
                    
                    # 使用软标签
                    target_domain_labels_adv = torch.ones(batch_size, 1).to(self.device) * 0.9
                    domain_loss_g = self.domain_loss(target_domain_pred, target_domain_labels_adv)
                    
                    # 总损失
                    total_loss = cls_loss + lambda_domain * domain_loss_g
                else:
                    # 预热期只训练分类
                    domain_loss_g = torch.tensor(0.0)
                    total_loss = cls_loss
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.target_fe.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                
                self.optimizer_target_fe.step()
                self.optimizer_classifier.step()
                
                # 统计
                epoch_cls_loss += cls_loss.item()
                epoch_domain_loss_d += domain_loss_d.item()
                epoch_domain_loss_g += domain_loss_g.item() if isinstance(domain_loss_g, torch.Tensor) else 0
                
                # 分类准确率
                _, cls_predicted = torch.max(source_pred.data, 1)
                cls_total += source_labels.size(0)
                cls_correct += (cls_predicted == source_labels).sum().item()
                
                # 域判别准确率
                with torch.no_grad():
                    source_features_eval = self.source_fe(source_data)
                    target_features_eval = self.target_fe(target_data)
                    
                    source_domain_pred_eval = self.discriminator(source_features_eval)
                    target_domain_pred_eval = self.discriminator(target_features_eval)
                    
                    domain_pred_binary = torch.cat([
                        (source_domain_pred_eval > 0.5).float(),
                        (target_domain_pred_eval > 0.5).float()
                    ])
                    domain_true_binary = torch.cat([
                        torch.ones(batch_size, 1),
                        torch.zeros(batch_size, 1)
                    ]).to(self.device)
                    
                    domain_total += domain_true_binary.size(0)
                    domain_correct += (domain_pred_binary == domain_true_binary).sum().item()
                
                progress_bar.set_postfix({
                    'Cls_Loss': f'{cls_loss.item():.4f}',
                    'Dom_Loss_D': f'{domain_loss_d.item():.4f}',
                    'Dom_Loss_G': f'{domain_loss_g.item() if isinstance(domain_loss_g, torch.Tensor) else 0:.4f}',
                    'Cls_Acc': f'{cls_correct/cls_total*100:.2f}%',
                    'Dom_Acc': f'{domain_correct/domain_total*100:.2f}%',
                    'Lambda': f'{lambda_domain:.3f}',
                    'Alpha': f'{alpha:.3f}'
                })
            
            # 更新学习率
            self.scheduler_target_fe.step()
            self.scheduler_classifier.step()
            self.scheduler_discriminator.step()
            
            # 记录历史
            avg_cls_loss = epoch_cls_loss / max_batches
            avg_domain_loss_d = epoch_domain_loss_d / max_batches
            avg_domain_loss_g = epoch_domain_loss_g / max_batches
            cls_accuracy = cls_correct / cls_total
            domain_accuracy = domain_correct / domain_total
            
            self.train_history['classification_loss'].append(avg_cls_loss)
            self.train_history['domain_loss_d'].append(avg_domain_loss_d)
            self.train_history['domain_loss_g'].append(avg_domain_loss_g)
            self.train_history['discriminator_accuracy'].append(domain_accuracy)
            self.train_history['classification_accuracy'].append(cls_accuracy)
            
            # 保存统计
            d_losses.append(avg_domain_loss_d)
            g_losses.append(avg_domain_loss_g)
            cls_losses.append(avg_cls_loss)
            
            # 检查收敛
            domain_confusion = abs(domain_accuracy - 0.5)
            if domain_confusion < self.best_domain_confusion:
                self.best_domain_confusion = domain_confusion
                self.save_best_model(epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s):")
            print(f"  分类损失: {avg_cls_loss:.4f} | 分类准确率: {cls_accuracy*100:.2f}%")
            print(f"  域损失(D): {avg_domain_loss_d:.4f} | 域损失(G): {avg_domain_loss_g:.4f}")
            print(f"  域判别准确率: {domain_accuracy*100:.2f}% | 域混淆度: {domain_confusion:.4f}")
            print(f"  Lambda: {lambda_domain:.3f} | Alpha: {alpha:.3f} | 预热期: {is_warmup}")
            
            # 早停检查
            if len(d_losses) > 20:
                recent_d_loss = np.mean(d_losses[-10:])
                old_d_loss = np.mean(d_losses[-20:-10])
                
                if abs(recent_d_loss - old_d_loss) < 0.001 and domain_accuracy > 0.45 and domain_accuracy < 0.55:
                    print(f"✅ 域适应收敛检测：域判别准确率稳定在 {domain_accuracy*100:.2f}%")
                    break
        
        print(f"✅ 域适应训练完成！最佳域混淆度: {self.best_domain_confusion:.4f}")
    
    def save_best_model(self, epoch):
        """保存最佳模型"""
        torch.save({
            'epoch': epoch,
            'target_fe_state_dict': self.target_fe.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'best_domain_confusion': self.best_domain_confusion,
            'train_history': self.train_history
        }, 'best_improved_domain_adaptation_model.pth')
        
        print(f"💾 保存最佳模型 (Epoch {epoch+1}, 域混淆度: {self.best_domain_confusion:.4f})")


# 使用示例函数
def create_improved_domain_adaptation_system():
    """创建改进的域适应系统"""
    
    print("🔧 创建改进版域适应系统...")
    
    feature_dim = 256
    
    # 使用改进的模型
    source_feature_extractor = ImprovedFeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim,
        dropout_rate=0.3
    )
    
    target_feature_extractor = ImprovedFeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim,
        dropout_rate=0.3
    )
    
    # 使用原来的分类器（如果工作良好）
    from domain_adaptation_system import Classifier
    classifier = Classifier(
        feature_dim=feature_dim,
        num_classes=2,
        dropout_rate=0.3
    )
    
    # 使用改进的判别器
    discriminator = ImprovedDomainDiscriminator(
        feature_dim=feature_dim,
        hidden_dim=128
    )
    
    # 使用改进的训练器
    trainer = ImprovedDomainAdaptationTrainer(
        source_feature_extractor=source_feature_extractor,
        target_feature_extractor=target_feature_extractor,
        classifier=classifier,
        discriminator=discriminator,
        device='auto'
    )
    
    print("✅ 改进版域适应系统创建完成")
    print("🔧 主要改进:")
    print("  - 更稳定的网络架构")
    print("  - 梯度裁剪和正则化")
    print("  - 改进的学习率调度")
    print("  - 预热训练机制")
    print("  - 标签平滑和噪声注入")
    print("  - 自适应权重调度")
    
    return trainer


if __name__ == "__main__":
    # 创建改进的系统
    trainer = create_improved_domain_adaptation_system()
    
    print(f"\n🎉 改进版域适应系统准备就绪！")
    print(f"📝 使用步骤:")
    print(f"  1. trainer.pretrain_source_model(source_loader, epochs=50)")
    print(f"  2. trainer.train_domain_adaptation_improved(source_loader, target_loader, epochs=100)")
    print(f"  3. trainer.evaluate_target_domain(target_test_loader)")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class SimpleDomainAdaptationTrainer:
    """简化的域适应训练器 - 基于MMD和特征对齐"""
    
    def __init__(self, source_feature_extractor, target_feature_extractor, 
                 classifier, device='auto'):
        """
        简化的域适应方案：
        1. 预训练源域模型
        2. 使用MMD损失对齐特征分布
        3. 渐进式fine-tuning
        """
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        self.source_fe = source_feature_extractor.to(self.device)
        self.target_fe = target_feature_extractor.to(self.device)
        self.classifier = classifier.to(self.device)
        
        # 损失函数
        self.classification_loss = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer_target_fe = optim.Adam(self.target_fe.parameters(), lr=0.0001)
        self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=0.0001)
        
        self.train_history = {
            'classification_loss': [],
            'mmd_loss': [],
            'total_loss': []
        }
    
    def mmd_loss(self, source_features, target_features, kernel='gaussian', sigma=1.0):
        """最大均值差异（MMD）损失"""
        def gaussian_kernel(x, y, sigma):
            """高斯核函数"""
            gamma = 1.0 / (2 * sigma**2)
            dist = torch.cdist(x, y, p=2)**2
            return torch.exp(-gamma * dist)
        
        def linear_kernel(x, y):
            """线性核函数"""
            return torch.mm(x, y.t())
        
        # 选择核函数
        if kernel == 'gaussian':
            kernel_func = lambda x, y: gaussian_kernel(x, y, sigma)
        else:
            kernel_func = linear_kernel
        
        # 计算MMD
        xx = kernel_func(source_features, source_features).mean()
        yy = kernel_func(target_features, target_features).mean()
        xy = kernel_func(source_features, target_features).mean()
        
        mmd = xx + yy - 2 * xy
        return mmd
    
    def coral_loss(self, source_features, target_features):
        """CORAL损失 - 协方差对齐"""
        def cov(x):
            """计算协方差矩阵"""
            mean = torch.mean(x, dim=0, keepdim=True)
            x_centered = x - mean
            return torch.mm(x_centered.t(), x_centered) / (x.size(0) - 1)
        
        source_cov = cov(source_features)
        target_cov = cov(target_features)
        
        # Frobenius范数
        loss = torch.mean(torch.pow(source_cov - target_cov, 2))
        return loss
    
    def pretrain_source_model(self, source_loader, epochs=50):
        """预训练源域模型"""
        print("🚀 开始预训练源域模型...")
        
        optimizer_source = optim.Adam(
            list(self.source_fe.parameters()) + list(self.classifier.parameters()), 
            lr=0.001
        )
        
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
                
                features = self.source_fe(batch_data)
                outputs = self.classifier(features)
                loss = self.classification_loss(outputs, batch_labels)
                
                loss.backward()
                optimizer_source.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct/total*100:.2f}%'
                })
            
            accuracy = correct / total
            if accuracy > best_acc:
                best_acc = accuracy
            
            print(f"预训练 Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(source_loader):.4f}, Acc: {accuracy*100:.2f}%")
        
        # 冻结源域特征提取器
        for param in self.source_fe.parameters():
            param.requires_grad = False
        
        print(f"✅ 源域预训练完成，最佳准确率: {best_acc*100:.2f}%")
    
    def train_simple_domain_adaptation(self, source_loader, target_loader, epochs=100, 
                                     mmd_weight=0.1, coral_weight=0.1):
        """简化的域适应训练"""
        print("🔄 开始简化域适应训练...")
        
        self.target_fe.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            epoch_cls_loss = 0.0
            epoch_mmd_loss = 0.0
            epoch_coral_loss = 0.0
            
            cls_correct = 0
            cls_total = 0
            
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            
            max_batches = min(len(source_loader), len(target_loader))
            progress_bar = tqdm(range(max_batches), desc=f"简化DA Epoch {epoch+1}/{epochs}")
            
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
                
                self.optimizer_target_fe.zero_grad()
                self.optimizer_classifier.zero_grad()
                
                # 特征提取
                with torch.no_grad():
                    source_features = self.source_fe(source_data)
                target_features = self.target_fe(target_data)
                
                # 分类损失（源域）
                source_pred = self.classifier(source_features)
                cls_loss = self.classification_loss(source_pred, source_labels)
                
                # MMD损失（特征对齐）
                mmd_loss_val = self.mmd_loss(source_features, target_features)
                
                # CORAL损失（协方差对齐）
                coral_loss_val = self.coral_loss(source_features, target_features)
                
                # 总损失
                total_loss = cls_loss + mmd_weight * mmd_loss_val + coral_weight * coral_loss_val
                
                total_loss.backward()
                self.optimizer_target_fe.step()
                self.optimizer_classifier.step()
                
                # 统计
                epoch_cls_loss += cls_loss.item()
                epoch_mmd_loss += mmd_loss_val.item()
                epoch_coral_loss += coral_loss_val.item()
                
                _, cls_predicted = torch.max(source_pred.data, 1)
                cls_total += source_labels.size(0)
                cls_correct += (cls_predicted == source_labels).sum().item()
                
                progress_bar.set_postfix({
                    'Cls_Loss': f'{cls_loss.item():.4f}',
                    'MMD': f'{mmd_loss_val.item():.4f}',
                    'CORAL': f'{coral_loss_val.item():.4f}',
                    'Acc': f'{cls_correct/cls_total*100:.2f}%'
                })
            
            # 记录历史
            self.train_history['classification_loss'].append(epoch_cls_loss / max_batches)
            self.train_history['mmd_loss'].append(epoch_mmd_loss / max_batches)
            self.train_history['total_loss'].append(
                (epoch_cls_loss + mmd_weight * epoch_mmd_loss + coral_weight * epoch_coral_loss) / max_batches
            )
            
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  分类损失: {epoch_cls_loss/max_batches:.4f}")
            print(f"  MMD损失: {epoch_mmd_loss/max_batches:.4f}")
            print(f"  CORAL损失: {epoch_coral_loss/max_batches:.4f}")
            print(f"  分类准确率: {cls_correct/cls_total*100:.2f}%")
            
            # 保存模型
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'target_fe_state_dict': self.target_fe.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                    'train_history': self.train_history
                }, f'simple_domain_adaptation_epoch_{epoch+1}.pth')
        
        print("✅ 简化域适应训练完成！")
    
    def evaluate_target_domain(self, target_loader_with_labels, class_names=['Normal', 'Fault']):
        """评估目标域性能"""
        print("🧪 评估目标域性能...")
        
        self.target_fe.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(target_loader_with_labels, desc="评估"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                features = self.target_fe(batch_data)
                outputs = self.classifier(features)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\n📊 目标域测试结果:")
        print(f"  准确率: {accuracy*100:.2f}%")
        print(f"  精确率: {precision*100:.2f}%")
        print(f"  召回率: {recall*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
        
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


class ProgressiveDomainAdaptationTrainer:
    """渐进式域适应训练器 - 另一种稳定方案"""
    
    def __init__(self, source_feature_extractor, target_feature_extractor, 
                 classifier, device='auto'):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        self.source_fe = source_feature_extractor.to(self.device)
        self.target_fe = target_feature_extractor.to(self.device)
        self.classifier = classifier.to(self.device)
        
        # 初始化目标域特征提取器为源域参数
        self.target_fe.load_state_dict(self.source_fe.state_dict())
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        self.train_history = {
            'classification_loss': [],
            'feature_alignment_loss': [],
            'total_loss': []
        }
    
    def pretrain_source_model(self, source_loader, epochs=50):
        """预训练源域模型"""
        print("🚀 开始预训练源域模型...")
        
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
            
            for batch_data, batch_labels in tqdm(source_loader, desc=f"预训练 {epoch+1}/{epochs}"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer_source.zero_grad()
                
                features = self.source_fe(batch_data)
                outputs = self.classifier(features)
                loss = self.classification_loss(outputs, batch_labels)
                
                loss.backward()
                optimizer_source.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            print(f"预训练 Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(source_loader):.4f}, Acc: {correct/total*100:.2f}%")
        
        # 冻结源域特征提取器
        for param in self.source_fe.parameters():
            param.requires_grad = False
        
        # 重新初始化目标域特征提取器
        self.target_fe.load_state_dict(self.source_fe.state_dict())
        
        print("✅ 源域预训练完成")
    
    def train_progressive_adaptation(self, source_loader, target_loader, 
                                   epochs=100, alignment_weight=1.0):
        """渐进式域适应训练"""
        print("🔄 开始渐进式域适应训练...")
        
        # 目标域特征提取器优化器
        optimizer_target = optim.Adam(self.target_fe.parameters(), lr=0.0001)
        optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=0.0001)
        
        self.target_fe.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            epoch_cls_loss = 0.0
            epoch_align_loss = 0.0
            
            cls_correct = 0
            cls_total = 0
            
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            
            max_batches = min(len(source_loader), len(target_loader))
            
            for batch_idx in tqdm(range(max_batches), desc=f"渐进DA Epoch {epoch+1}/{epochs}"):
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
                
                optimizer_target.zero_grad()
                optimizer_classifier.zero_grad()
                
                # 获取特征
                with torch.no_grad():
                    source_features = self.source_fe(source_data)
                target_features = self.target_fe(target_data)
                
                # 分类损失
                source_pred = self.classifier(source_features)
                cls_loss = self.classification_loss(source_pred, source_labels)
                
                # 特征对齐损失（让目标域特征接近源域特征）
                target_source_features = self.target_fe(source_data)
                align_loss = self.mse_loss(target_source_features, source_features)
                
                # 总损失
                total_loss = cls_loss + alignment_weight * align_loss
                
                total_loss.backward()
                optimizer_target.step()
                optimizer_classifier.step()
                
                # 统计
                epoch_cls_loss += cls_loss.item()
                epoch_align_loss += align_loss.item()
                
                _, cls_predicted = torch.max(source_pred.data, 1)
                cls_total += source_labels.size(0)
                cls_correct += (cls_predicted == source_labels).sum().item()
            
            # 记录历史
            self.train_history['classification_loss'].append(epoch_cls_loss / max_batches)
            self.train_history['feature_alignment_loss'].append(epoch_align_loss / max_batches)
            self.train_history['total_loss'].append((epoch_cls_loss + alignment_weight * epoch_align_loss) / max_batches)
            
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  分类损失: {epoch_cls_loss/max_batches:.4f}")
            print(f"  对齐损失: {epoch_align_loss/max_batches:.4f}")
            print(f"  分类准确率: {cls_correct/cls_total*100:.2f}%")
            
            # 逐渐减少对齐权重
            if epoch > epochs // 2:
                alignment_weight *= 0.99
        
        print("✅ 渐进式域适应训练完成！")
    
    def evaluate_target_domain(self, target_loader_with_labels, class_names=['Normal', 'Fault']):
        """评估目标域性能"""
        print("🧪 评估目标域性能...")
        
        self.target_fe.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(target_loader_with_labels, desc="评估"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                features = self.target_fe(batch_data)
                outputs = self.classifier(features)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\n📊 目标域测试结果:")
        print(f"  准确率: {accuracy*100:.2f}%")
        print(f"  精确率: {precision*100:.2f}%")
        print(f"  召回率: {recall*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
        
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


def create_stable_domain_adaptation_options():
    """创建稳定的域适应选项"""
    
    print("🔧 提供多种稳定的域适应方案:")
    print("=" * 50)
    print("1️⃣  改进版GAN域适应 (improved_domain_adaptation.py)")
    print("   - 稳定的梯度反转")
    print("   - 预热训练机制")
    print("   - 自适应权重调度")
    print("   - 梯度裁剪和正则化")
    
    print("\n2️⃣  简化MMD域适应 (SimpleDomainAdaptationTrainer)")
    print("   - 基于MMD损失的特征对齐")
    print("   - CORAL协方差对齐")
    print("   - 无对抗训练，更稳定")
    
    print("\n3️⃣  渐进式域适应 (ProgressiveDomainAdaptationTrainer)")
    print("   - 特征逐步对齐")
    print("   - 从源域参数开始")
    print("   - 简单而有效")
    
    print("\n🎯 推荐使用顺序:")
    print("   1. 先尝试简化MMD方案（最稳定）")
    print("   2. 如果效果不好，尝试渐进式方案")
    print("   3. 最后尝试改进版GAN方案")


if __name__ == "__main__":
    create_stable_domain_adaptation_options()
    
    print(f"\n📝 使用示例:")
    print(f"# 方案1：简化MMD域适应")
    print(f"trainer = SimpleDomainAdaptationTrainer(source_fe, target_fe, classifier)")
    print(f"trainer.pretrain_source_model(source_loader)")
    print(f"trainer.train_simple_domain_adaptation(source_loader, target_loader)")
    
    print(f"\n# 方案2：渐进式域适应")
    print(f"trainer = ProgressiveDomainAdaptationTrainer(source_fe, target_fe, classifier)")
    print(f"trainer.pretrain_source_model(source_loader)")
    print(f"trainer.train_progressive_adaptation(source_loader, target_loader)")
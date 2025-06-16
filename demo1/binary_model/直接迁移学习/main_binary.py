import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# 导入你的数据处理器
from data_loader import OptimizedBearingDataProcessor


class StatisticalAnomalyDetector:
    """基于统计距离的无监督异常检测器"""
    
    def __init__(self, method='mahalanobis'):
        """
        初始化检测器
        
        参数:
        - method: 'mahalanobis', 'euclidean', 'isolation_forest'
        """
        self.method = method
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, normal_data):
        """只用正常数据训练"""
        print(f"🔧 训练统计异常检测器 (方法: {self.method})")
        
        # 如果是3D数据，展平为2D
        if normal_data.ndim == 3:
            normal_data = normal_data.reshape(normal_data.shape[0], -1)
        
        # 标准化
        self.normal_scaled = self.scaler.fit_transform(normal_data)
        
        if self.method == 'mahalanobis':
            # 计算正常数据的统计特征
            self.mean = np.mean(self.normal_scaled, axis=0)
            self.cov = np.cov(self.normal_scaled.T)
            
            # 处理奇异协方差矩阵
            try:
                self.inv_cov = np.linalg.inv(self.cov)
            except np.linalg.LinAlgError:
                # 使用伪逆处理奇异矩阵
                self.inv_cov = np.linalg.pinv(self.cov)
            
            # 计算正常数据的马氏距离分布
            normal_distances = []
            for x in self.normal_scaled:
                try:
                    dist = mahalanobis(x, self.mean, self.inv_cov)
                    normal_distances.append(dist)
                except:
                    # 如果计算失败，使用欧氏距离
                    dist = np.linalg.norm(x - self.mean)
                    normal_distances.append(dist)
            
            self.threshold = np.percentile(normal_distances, 95)
            
        elif self.method == 'euclidean':
            self.mean = np.mean(self.normal_scaled, axis=0)
            # 计算正常数据的欧氏距离
            normal_distances = [np.linalg.norm(x - self.mean) for x in self.normal_scaled]
            self.threshold = np.percentile(normal_distances, 95)
            
        elif self.method == 'isolation_forest':
            self.detector = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
            self.detector.fit(self.normal_scaled)
            
        self.is_fitted = True
        print(f"✅ 训练完成，阈值: {getattr(self, 'threshold', 'N/A')}")
        
        return self
    
    def predict(self, test_data):
        """预测测试数据"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 如果是3D数据，展平为2D
        if test_data.ndim == 3:
            test_data = test_data.reshape(test_data.shape[0], -1)
            
        test_scaled = self.scaler.transform(test_data)
        
        if self.method == 'mahalanobis':
            distances = []
            for x in test_scaled:
                try:
                    dist = mahalanobis(x, self.mean, self.inv_cov)
                    distances.append(dist)
                except:
                    dist = np.linalg.norm(x - self.mean)
                    distances.append(dist)
            
            distances = np.array(distances)
            predictions = (distances > self.threshold).astype(int)
            
        elif self.method == 'euclidean':
            distances = np.array([np.linalg.norm(x - self.mean) for x in test_scaled])
            predictions = (distances > self.threshold).astype(int)
            
        elif self.method == 'isolation_forest':
            predictions = self.detector.predict(test_scaled)
            predictions = (predictions == -1).astype(int)  # -1表示异常，转换为1
            distances = -self.detector.decision_function(test_scaled)  # 负值表示异常分数
            
        return predictions, distances


class DomainAdaptiveAutoencoder(nn.Module):
    """域适应自编码器"""
    
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim)
        )
        
        # 域判别器
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        domain_pred = self.domain_classifier(encoded)
        return encoded, decoded, domain_pred


class CrossDomainFaultDiagnosis:
    """跨域故障诊断主类"""
    
    def __init__(self, source_dir='bearing_dataset', target_dir='bearing_dataset1'):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
    def load_domain_data(self, data_dir, normal_keywords=['N', 'Normal']):
        """加载指定域的数据"""
        print(f"📂 加载数据: {data_dir}")
        
        processor = OptimizedBearingDataProcessor(data_dir=data_dir, seq_length=1000)
        
        # 加载原始数据
        raw_data, raw_labels = processor.load_raw_data()
        
        normal_data = []
        fault_data = []
        
        for data, label in zip(raw_data, raw_labels):
            # 判断是否为正常数据
            is_normal = any(keyword in label for keyword in normal_keywords)
            
            # 创建滑动窗口
            windows = processor.create_sliding_windows_vectorized(data)
            
            if len(windows) > 0:
                if is_normal:
                    normal_data.extend(windows)
                else:
                    fault_data.extend(windows)
        
        normal_data = np.array(normal_data)
        fault_data = np.array(fault_data)
        
        print(f"  ✅ 正常数据: {len(normal_data):,} 样本")
        print(f"  ✅ 故障数据: {len(fault_data):,} 样本")
        
        return normal_data, fault_data
    
    def evaluate_method(self, predictions, scores, true_labels, method_name):
        """评估方法性能"""
        print(f"\n📊 {method_name} 评估结果:")
        print("=" * 50)
        
        # 分类报告
        report = classification_report(true_labels, predictions, 
                                     target_names=['正常', '故障'], 
                                     digits=4)
        print(report)
        
        # AUC分数
        try:
            auc = roc_auc_score(true_labels, scores)
            print(f"🎯 AUC Score: {auc:.4f}")
        except:
            print("⚠️ 无法计算AUC")
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['正常', '故障'], 
                   yticklabels=['正常', '故障'])
        plt.title(f'{method_name} - 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()
        
        return auc if 'auc' in locals() else 0
    
    def run_statistical_method(self):
        """运行统计方法"""
        print("\n🚀 开始统计方法测试")
        print("=" * 60)
        
        # 加载源域和目标域数据
        source_normal, source_fault = self.load_domain_data(self.source_dir)
        target_normal, target_fault = self.load_domain_data(self.target_dir)
        
        # 准备测试数据和标签
        target_test_data = np.vstack([target_normal, target_fault])
        target_test_labels = np.hstack([
            np.zeros(len(target_normal)),  # 正常=0
            np.ones(len(target_fault))     # 故障=1
        ])
        
        # 测试不同的统计方法
        methods = ['mahalanobis', 'euclidean', 'isolation_forest']
        results = {}
        
        for method in methods:
            print(f"\n🔬 测试方法: {method}")
            
            # 训练检测器
            detector = StatisticalAnomalyDetector(method=method)
            detector.fit(source_normal)
            
            # 在目标域测试
            predictions, scores = detector.predict(target_test_data)
            
            # 评估结果
            auc = self.evaluate_method(predictions, scores, target_test_labels, 
                                     f"统计方法-{method}")
            results[method] = auc
        
        # 显示最佳方法
        best_method = max(results, key=results.get)
        print(f"\n🏆 最佳统计方法: {best_method} (AUC: {results[best_method]:.4f})")
        
        return results
    
    def train_domain_adaptive_autoencoder(self, source_normal, target_unlabeled, 
                                        epochs=100, batch_size=64):
        """训练域适应自编码器"""
        print("🤖 训练域适应自编码器")
        
        # 数据预处理
        if source_normal.ndim == 3:
            source_normal = source_normal.reshape(source_normal.shape[0], -1)
        if target_unlabeled.ndim == 3:
            target_unlabeled = target_unlabeled.reshape(target_unlabeled.shape[0], -1)
        
        # 标准化
        scaler = StandardScaler()
        source_scaled = scaler.fit_transform(source_normal)
        target_scaled = scaler.transform(target_unlabeled)
        
        # 转换为tensor
        source_tensor = torch.FloatTensor(source_scaled).to(self.device)
        target_tensor = torch.FloatTensor(target_scaled).to(self.device)
        
        # 创建模型
        input_dim = source_scaled.shape[1]
        model = DomainAdaptiveAutoencoder(input_dim).to(self.device)
        
        # 损失函数和优化器
        reconstruction_loss = nn.MSELoss()
        domain_loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            # 创建批次
            source_batches = torch.split(source_tensor, batch_size)
            target_batches = torch.split(target_tensor, batch_size)
            
            epoch_loss = 0
            for s_batch, t_batch in zip(source_batches, target_batches):
                optimizer.zero_grad()
                
                # 源域重构损失
                _, decoded_s, _ = model(s_batch)
                recon_loss = reconstruction_loss(decoded_s, s_batch)
                
                # 域对抗损失
                _, _, domain_pred_s = model(s_batch)
                _, _, domain_pred_t = model(t_batch)
                
                domain_loss_val = (
                    domain_loss(domain_pred_s, torch.ones_like(domain_pred_s)) +
                    domain_loss(domain_pred_t, torch.zeros_like(domain_pred_t))
                )
                
                # 总损失
                total_loss = recon_loss - 0.1 * domain_loss_val
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        return model, scaler
    
    def run_deep_learning_method(self):
        """运行深度学习方法"""
        print("\n🤖 开始深度学习方法测试")
        print("=" * 60)
        
        # 加载数据
        source_normal, source_fault = self.load_domain_data(self.source_dir)
        target_normal, target_fault = self.load_domain_data(self.target_dir)
        
        # 使用目标域的正常数据进行域适应（实际应用中可能没有标签）
        target_unlabeled = np.vstack([target_normal[:len(target_normal)//2], 
                                     target_fault[:len(target_fault)//4]])  # 模拟无标签数据
        
        # 训练域适应自编码器
        model, scaler = self.train_domain_adaptive_autoencoder(
            source_normal, target_unlabeled, epochs=50
        )
        
        # 准备测试数据
        target_test_data = np.vstack([target_normal, target_fault])
        target_test_labels = np.hstack([
            np.zeros(len(target_normal)),
            np.ones(len(target_fault))
        ])
        
        # 测试
        model.eval()
        with torch.no_grad():
            if target_test_data.ndim == 3:
                target_test_data = target_test_data.reshape(target_test_data.shape[0], -1)
            
            target_test_scaled = scaler.transform(target_test_data)
            target_test_tensor = torch.FloatTensor(target_test_scaled).to(self.device)
            
            _, decoded, _ = model(target_test_tensor)
            reconstruction_errors = torch.mean((target_test_tensor - decoded) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        # 设置阈值
        threshold = np.percentile(reconstruction_errors, 85)
        predictions = (reconstruction_errors > threshold).astype(int)
        
        # 评估结果
        auc = self.evaluate_method(predictions, reconstruction_errors, target_test_labels, 
                                 "域适应自编码器")
        
        return auc
    
    def run_comprehensive_evaluation(self):
        """运行综合评估"""
        print("🎯 开始跨域故障诊断综合评估")
        print("=" * 80)
        
        # 统计方法
        statistical_results = self.run_statistical_method()
        
        # 深度学习方法
        try:
            dl_auc = self.run_deep_learning_method()
            statistical_results['domain_adaptive_ae'] = dl_auc
        except Exception as e:
            print(f"⚠️ 深度学习方法失败: {e}")
        
        # 总结
        print("\n🏆 最终结果总结")
        print("=" * 50)
        for method, auc in statistical_results.items():
            print(f"{method:20s}: AUC = {auc:.4f}")
        
        best_method = max(statistical_results, key=statistical_results.get)
        print(f"\n🥇 最佳方法: {best_method} (AUC: {statistical_results[best_method]:.4f})")
        
        return statistical_results


# 使用示例
if __name__ == "__main__":
    # 创建跨域故障诊断系统
    diagnosis_system = CrossDomainFaultDiagnosis(
        source_dir='bearing_dataset',    # 源域（A数据集）
        target_dir='bearing_dataset1'    # 目标域（B数据集）
    )
    
    # 运行综合评估
    results = diagnosis_system.run_comprehensive_evaluation()
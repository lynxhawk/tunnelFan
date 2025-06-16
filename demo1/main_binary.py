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
import pickle
import json
from datetime import datetime
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
    
    def save_model(self, filepath):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        
        model_data = {
            'method': self.method,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'threshold': getattr(self, 'threshold', None)
        }
        
        # 根据方法保存不同的参数
        if self.method == 'mahalanobis':
            model_data.update({
                'mean': self.mean,
                'cov': self.cov,
                'inv_cov': self.inv_cov
            })
        elif self.method == 'euclidean':
            model_data.update({
                'mean': self.mean
            })
        elif self.method == 'isolation_forest':
            model_data.update({
                'detector': self.detector
            })
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 统计模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.method = model_data['method']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        
        if 'threshold' in model_data:
            self.threshold = model_data['threshold']
        
        # 根据方法加载不同的参数
        if self.method == 'mahalanobis':
            self.mean = model_data['mean']
            self.cov = model_data['cov']
            self.inv_cov = model_data['inv_cov']
        elif self.method == 'euclidean':
            self.mean = model_data['mean']
        elif self.method == 'isolation_forest':
            self.detector = model_data['detector']
        
        print(f"📂 统计模型已加载: {filepath}")
        return self


class SimpleAutoencoder(nn.Module):
    """简单的自编码器"""
    
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
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoencoderAnomalyDetector:
    """基于自编码器的异常检测器"""
    
    def __init__(self, latent_dim=64, epochs=100, batch_size=64, lr=0.001):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, normal_data):
        """用正常数据训练自编码器"""
        print(f"🤖 训练自编码器 (设备: {self.device})")
        
        # 数据预处理
        if normal_data.ndim == 3:
            normal_data = normal_data.reshape(normal_data.shape[0], -1)
        
        # 标准化
        normal_scaled = self.scaler.fit_transform(normal_data)
        
        # 转换为tensor
        normal_tensor = torch.FloatTensor(normal_scaled).to(self.device)
        
        # 创建模型
        input_dim = normal_scaled.shape[1]
        self.model = SimpleAutoencoder(input_dim, self.latent_dim).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 训练
        self.model.train()
        for epoch in range(self.epochs):
            # 创建批次
            indices = torch.randperm(len(normal_tensor))
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(normal_tensor), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_data = normal_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # 前向传播
                _, decoded = self.model(batch_data)
                loss = criterion(decoded, batch_data)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        # 计算正常数据的重构误差分布，用于设置阈值
        self.model.eval()
        with torch.no_grad():
            _, decoded = self.model(normal_tensor)
            reconstruction_errors = torch.mean((normal_tensor - decoded) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        # 设置阈值为95%分位数
        self.threshold = np.percentile(reconstruction_errors, 95)
        self.is_fitted = True
        
        print(f"✅ 训练完成，重构误差阈值: {self.threshold:.6f}")
        return self
    
    def predict(self, test_data):
        """预测测试数据"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 数据预处理
        if test_data.ndim == 3:
            test_data = test_data.reshape(test_data.shape[0], -1)
        
        test_scaled = self.scaler.transform(test_data)
        test_tensor = torch.FloatTensor(test_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            _, decoded = self.model(test_tensor)
            reconstruction_errors = torch.mean((test_tensor - decoded) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        # 根据阈值判断异常
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions, reconstruction_errors
    
    def save_model(self, filepath):
        """保存自编码器模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        
        # 保存模型状态字典
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.encoder[0].in_features,
            'latent_dim': self.latent_dim,
            'threshold': self.threshold,
            'scaler': self.scaler
        }
        
        torch.save(model_state, filepath)
        print(f"💾 自编码器模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """加载自编码器模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 重建模型
        input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim']
        self.model = SimpleAutoencoder(input_dim, self.latent_dim).to(self.device)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.scaler = checkpoint['scaler']
        self.is_fitted = True
        
        print(f"📂 自编码器模型已加载: {filepath}")
        return self


class DatasetSpecificFaultDiagnosis:
    """针对特定数据集的故障诊断"""
    
    def __init__(self, dataset_dir, dataset_name, normal_keywords=['N', 'Normal'], 
                 save_models=True, models_dir='saved_models'):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.normal_keywords = normal_keywords
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_models = save_models
        self.models_dir = models_dir
        
        # 创建模型保存目录
        if self.save_models:
            os.makedirs(self.models_dir, exist_ok=True)
            dataset_model_dir = os.path.join(self.models_dir, self.dataset_name)
            os.makedirs(dataset_model_dir, exist_ok=True)
            self.dataset_model_dir = dataset_model_dir
        
    def load_dataset(self):
        """加载数据集"""
        print(f"📂 加载数据集: {self.dataset_name} ({self.dataset_dir})")
        
        processor = OptimizedBearingDataProcessor(data_dir=self.dataset_dir, seq_length=1000)
        
        # 加载原始数据
        raw_data, raw_labels = processor.load_raw_data()
        
        normal_data = []
        fault_data = []
        
        for data, label in zip(raw_data, raw_labels):
            # 判断是否为正常数据
            is_normal = any(keyword in label for keyword in self.normal_keywords)
            
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
        print(f"\n📊 {self.dataset_name} - {method_name} 评估结果:")
        print("=" * 60)
        
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
            auc = 0
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['正常', '故障'], 
                   yticklabels=['正常', '故障'])
        plt.title(f'{self.dataset_name} - {method_name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()
        
        return auc
    
    def save_model_with_metadata(self, model, method_name, auc_score, train_normal_size):
        """保存模型及其元数据"""
        if not self.save_models:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_filename = f"{method_name}_{timestamp}.pkl"
        if method_name == "autoencoder":
            model_filename = f"{method_name}_{timestamp}.pth"
        
        model_path = os.path.join(self.dataset_model_dir, model_filename)
        model.save_model(model_path)
        
        # 保存元数据
        metadata = {
            'dataset_name': self.dataset_name,
            'method': method_name,
            'auc_score': auc_score,
            'train_normal_size': train_normal_size,
            'timestamp': timestamp,
            'model_file': model_filename,
            'device': str(self.device)
        }
        
        metadata_path = os.path.join(self.dataset_model_dir, f"{method_name}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📁 模型和元数据已保存: {self.dataset_model_dir}")
    
    def load_best_model(self, method_name):
        """加载指定方法的最佳模型（基于AUC）"""
        if not os.path.exists(self.dataset_model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.dataset_model_dir}")
        
        # 查找所有该方法的元数据文件
        metadata_files = [f for f in os.listdir(self.dataset_model_dir) 
                         if f.startswith(f"{method_name}_") and f.endswith("_metadata.json")]
        
        if not metadata_files:
            raise FileNotFoundError(f"未找到方法 {method_name} 的模型")
        
        # 找到AUC最高的模型
        best_auc = -1
        best_metadata = None
        
        for metadata_file in metadata_files:
            metadata_path = os.path.join(self.dataset_model_dir, metadata_file)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if metadata['auc_score'] > best_auc:
                best_auc = metadata['auc_score']
                best_metadata = metadata
        
        if best_metadata is None:
            raise ValueError(f"未找到有效的 {method_name} 模型")
        
        # 加载最佳模型
        model_path = os.path.join(self.dataset_model_dir, best_metadata['model_file'])
        
        if method_name == "autoencoder":
            model = AutoencoderAnomalyDetector()
            model.load_model(model_path)
        else:
            # 统计方法
            model = StatisticalAnomalyDetector()
            model.load_model(model_path)
        
        print(f"🏆 已加载最佳 {method_name} 模型 (AUC: {best_auc:.4f})")
        return model, best_metadata
    
    def run_all_methods(self):
        """运行所有方法"""
        print(f"\n🚀 开始 {self.dataset_name} 故障诊断测试")
        print("=" * 80)
        
        # 加载数据
        normal_data, fault_data = self.load_dataset()
        
        # 准备测试数据和标签
        test_data = np.vstack([normal_data, fault_data])
        test_labels = np.hstack([
            np.zeros(len(normal_data)),  # 正常=0
            np.ones(len(fault_data))     # 故障=1
        ])
        
        # 只用部分正常数据训练（模拟实际情况）
        train_normal_ratio = 0.7
        train_size = int(len(normal_data) * train_normal_ratio)
        train_normal = normal_data[:train_size]
        
        print(f"📈 训练用正常数据: {len(train_normal):,} 样本")
        print(f"📋 测试数据: {len(test_data):,} 样本 (正常: {len(normal_data)}, 故障: {len(fault_data)})")
        
        results = {}
        
        # 1. 统计方法
        statistical_methods = ['mahalanobis', 'euclidean', 'isolation_forest']
        
        for method in statistical_methods:
            print(f"\n🔬 测试统计方法: {method}")
            
            try:
                detector = StatisticalAnomalyDetector(method=method)
                detector.fit(train_normal)
                
                predictions, scores = detector.predict(test_data)
                auc = self.evaluate_method(predictions, scores, test_labels, 
                                         f"统计方法-{method}")
                results[f"statistical_{method}"] = auc
                
                # 保存模型
                self.save_model_with_metadata(detector, f"statistical_{method}", auc, len(train_normal))
                
            except Exception as e:
                print(f"⚠️ 方法 {method} 失败: {e}")
                results[f"statistical_{method}"] = 0
        
        # 2. 自编码器方法
        print(f"\n🤖 测试自编码器方法")
        
        try:
            ae_detector = AutoencoderAnomalyDetector(
                latent_dim=64, 
                epochs=50,  # 减少训练轮数以节省时间
                batch_size=64,
                lr=0.001
            )
            ae_detector.fit(train_normal)
            
            predictions, scores = ae_detector.predict(test_data)
            auc = self.evaluate_method(predictions, scores, test_labels, "自编码器")
            results["autoencoder"] = auc
            
            # 保存模型
            self.save_model_with_metadata(ae_detector, "autoencoder", auc, len(train_normal))
            
        except Exception as e:
            print(f"⚠️ 自编码器方法失败: {e}")
            results["autoencoder"] = 0
        
        # 结果总结
        print(f"\n🏆 {self.dataset_name} 结果总结")
        print("=" * 50)
        for method, auc in results.items():
            print(f"{method:25s}: AUC = {auc:.4f}")
        
        best_method = max(results, key=results.get)
        print(f"\n🥇 最佳方法: {best_method} (AUC: {results[best_method]:.4f})")
        
        return results


class ComprehensiveFaultDiagnosis:
    """综合故障诊断系统"""
    
    def __init__(self, datasets_config, save_models=True, models_dir='saved_models'):
        """
        初始化综合诊断系统
        
        参数:
        datasets_config: 字典，格式如下
        {
            'dataset_A': {
                'dir': 'bearing_dataset',
                'normal_keywords': ['N']
            },
            'dataset_B': {
                'dir': 'bearing_dataset1', 
                'normal_keywords': ['Normal']
            }
        }
        save_models: 是否保存模型
        models_dir: 模型保存目录
        """
        self.datasets_config = datasets_config
        self.save_models = save_models
        self.models_dir = models_dir
        
    def run_comprehensive_evaluation(self):
        """运行综合评估"""
        print("🎯 开始综合故障诊断评估")
        print("=" * 100)
        
        all_results = {}
        
        # 分别测试每个数据集
        for dataset_name, config in self.datasets_config.items():
            diagnosis_system = DatasetSpecificFaultDiagnosis(
                dataset_dir=config['dir'],
                dataset_name=dataset_name,
                normal_keywords=config['normal_keywords'],
                save_models=self.save_models,
                models_dir=self.models_dir
            )
            
            dataset_results = diagnosis_system.run_all_methods()
            all_results[dataset_name] = dataset_results
        
        # 创建综合结果表格
        self.print_comprehensive_results(all_results)
        
        return all_results
    
    def print_comprehensive_results(self, all_results):
        """打印综合结果"""
        print("\n" + "="*100)
        print("🏆 综合结果总览")
        print("="*100)
        
        # 获取所有方法名
        all_methods = set()
        for dataset_results in all_results.values():
            all_methods.update(dataset_results.keys())
        all_methods = sorted(list(all_methods))
        
        # 打印表格
        print(f"{'方法':<25s}", end="")
        for dataset_name in self.datasets_config.keys():
            print(f"{dataset_name:>15s}", end="")
        print()
        print("-" * (25 + 15 * len(self.datasets_config)))
        
        for method in all_methods:
            print(f"{method:<25s}", end="")
            for dataset_name in self.datasets_config.keys():
                auc = all_results[dataset_name].get(method, 0)
                print(f"{auc:>15.4f}", end="")
            print()
        
        # 找出每个数据集的最佳方法
        print(f"\n🏅 各数据集最佳方法:")
        for dataset_name, dataset_results in all_results.items():
            best_method = max(dataset_results, key=dataset_results.get)
            best_auc = dataset_results[best_method]
            print(f"  {dataset_name}: {best_method} (AUC: {best_auc:.4f})")
    
    def load_and_test_saved_model(self, dataset_name, method_name, test_data, test_labels):
        """加载保存的模型进行测试"""
        dataset_model_dir = os.path.join(self.models_dir, dataset_name)
        
        diagnosis_system = DatasetSpecificFaultDiagnosis(
            dataset_dir="",  # 不需要数据目录，只用于加载模型
            dataset_name=dataset_name,
            save_models=False
        )
        diagnosis_system.dataset_model_dir = dataset_model_dir
        
        try:
            model, metadata = diagnosis_system.load_best_model(method_name)
            
            # 进行预测
            predictions, scores = model.predict(test_data)
            
            # 评估
            auc = diagnosis_system.evaluate_method(predictions, scores, test_labels, 
                                                 f"加载的{method_name}模型")
            
            print(f"📊 加载的模型性能 - AUC: {auc:.4f}")
            print(f"📅 模型训练时间: {metadata['timestamp']}")
            print(f"🎯 原始AUC: {metadata['auc_score']:.4f}")
            
            return model, predictions, scores, metadata
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return None, None, None, None
    
    def list_saved_models(self):
        """列出所有保存的模型"""
        print("\n📂 已保存的模型:")
        print("=" * 80)
        
        if not os.path.exists(self.models_dir):
            print("❌ 模型目录不存在")
            return
        
        for dataset_name in os.listdir(self.models_dir):
            dataset_path = os.path.join(self.models_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
                
            print(f"\n📁 {dataset_name}:")
            
            metadata_files = [f for f in os.listdir(dataset_path) 
                            if f.endswith("_metadata.json")]
            
            if not metadata_files:
                print("  ❌ 无保存的模型")
                continue
            
            # 按方法分组
            methods = {}
            for metadata_file in metadata_files:
                metadata_path = os.path.join(dataset_path, metadata_file)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                method = metadata['method']
                if method not in methods:
                    methods[method] = []
                methods[method].append(metadata)
            
            # 显示每个方法的最佳模型
            for method, model_list in methods.items():
                best_model = max(model_list, key=lambda x: x['auc_score'])
                print(f"  🏆 {method:20s}: AUC={best_model['auc_score']:.4f}, "
                      f"时间={best_model['timestamp']}")
        
        print("\n" + "=" * 80)


# 使用示例
if __name__ == "__main__":
    # 配置数据集
    datasets_config = {
        'Dataset_A': {
            'dir': 'bearing_dataset',     # A数据集目录
            'normal_keywords': ['N']      # 正常数据文件名包含的关键词
        },
        'Dataset_B': {
            'dir': 'bearing_dataset1',    # B数据集目录  
            'normal_keywords': ['Normal'] # 正常数据文件名包含的关键词
        }
    }
    
    # 创建综合诊断系统（启用模型保存）
    comprehensive_system = ComprehensiveFaultDiagnosis(
        datasets_config, 
        save_models=True, 
        models_dir='saved_models'
    )
    
    print("🚀 模式1: 训练新模型并保存")
    # 运行完整评估（会自动保存模型）
    results = comprehensive_system.run_comprehensive_evaluation()
    
    # 列出保存的模型
    comprehensive_system.list_saved_models()
    
    print("\n" + "="*80)
    print("🔍 模式2: 加载保存的模型进行测试")
    print("="*80)
    
    # 示例：加载并测试保存的模型
    # 这里需要你提供测试数据
    # test_data = your_test_data
    # test_labels = your_test_labels
    # 
    # model, predictions, scores, metadata = comprehensive_system.load_and_test_saved_model(
    #     dataset_name='Dataset_A',
    #     method_name='statistical_isolation_forest',
    #     test_data=test_data,
    #     test_labels=test_labels
    # )
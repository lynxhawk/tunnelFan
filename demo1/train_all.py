import os
import numpy as np
import pickle
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score, roc_curve)
from scipy.spatial.distance import mahalanobis
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 导入数据处理器
from data_loader import OptimizedBearingDataProcessor


class UnsupervisedDetectorBase:
    """无监督检测器基类"""
    
    def __init__(self, name):
        self.name = name
        self.is_fitted = False
        
    def fit(self, X):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
    def save_model(self, filepath):
        """保存模型到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 模型已保存: {filepath}")
        
    def load_model(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        # 复制属性到当前实例
        self.__dict__.update(loaded_model.__dict__)
        print(f"📂 模型已加载: {filepath}")
        return self


class MahalanobisDetector(UnsupervisedDetectorBase):
    """马氏距离检测器"""
    
    def __init__(self, contamination=0.1):
        super().__init__("Mahalanobis")
        self.contamination = contamination
        self.scaler = StandardScaler()
        
    def fit(self, X):
        print(f"🔧 训练 {self.name} 检测器...")
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        # 标准化并移除离群值
        X_scaled = self.scaler.fit_transform(X)
        z_scores = np.abs(stats.zscore(X_scaled, axis=0))
        outlier_mask = (z_scores < 3).all(axis=1)
        X_clean = X_scaled[outlier_mask]
        
        self.mean = np.mean(X_clean, axis=0)
        self.cov = np.cov(X_clean.T) + 1e-6 * np.eye(X_clean.shape[1])
        
        try:
            self.inv_cov = np.linalg.inv(self.cov)
        except:
            self.inv_cov = np.linalg.pinv(self.cov)
        
        # 计算阈值
        distances = []
        for x in X_clean:
            try:
                dist = mahalanobis(x, self.mean, self.inv_cov)
            except:
                dist = np.linalg.norm(x - self.mean)
            distances.append(dist)
        
        self.threshold = np.percentile(distances, (1-self.contamination) * 100)
        self.is_fitted = True
        
        print(f"✅ 训练完成，阈值: {self.threshold:.6f}")
        return self
        
    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        distances = []
        
        for x in X_scaled:
            try:
                dist = mahalanobis(x, self.mean, self.inv_cov)
            except:
                dist = np.linalg.norm(x - self.mean)
            distances.append(dist)
        
        distances = np.array(distances)
        predictions = (distances > self.threshold).astype(int)
        
        return predictions, distances


class IsolationForestDetector(UnsupervisedDetectorBase):
    """孤立森林检测器"""
    
    def __init__(self, contamination=0.1, n_estimators=100):
        super().__init__("IsolationForest")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        
    def fit(self, X):
        print(f"🌳 训练 {self.name} 检测器...")
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"✅ 训练完成，使用 {self.n_estimators} 棵树")
        return self
        
    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        
        # 预测：-1表示异常，1表示正常
        predictions_raw = self.model.predict(X_scaled)
        predictions = (predictions_raw == -1).astype(int)
        
        # 异常分数：越负越异常
        scores = -self.model.decision_function(X_scaled)
        
        return predictions, scores


class LOFDetector(UnsupervisedDetectorBase):
    """局部异常因子检测器"""
    
    def __init__(self, contamination=0.1, n_neighbors=20):
        super().__init__("LOF")
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        
    def fit(self, X):
        print(f"👥 训练 {self.name} 检测器...")
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        self.X_train = self.scaler.fit_transform(X)
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,  # 用于新样本检测
            n_jobs=-1
        )
        
        self.model.fit(self.X_train)
        self.is_fitted = True
        
        print(f"✅ 训练完成，邻居数: {self.n_neighbors}")
        return self
        
    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        
        predictions_raw = self.model.predict(X_scaled)
        predictions = (predictions_raw == -1).astype(int)
        
        scores = -self.model.decision_function(X_scaled)
        
        return predictions, scores


class OneClassSVMDetector(UnsupervisedDetectorBase):
    """单类支持向量机检测器"""
    
    def __init__(self, contamination=0.1, gamma='auto'):
        super().__init__("OneClassSVM")
        self.contamination = contamination
        self.gamma = gamma
        self.scaler = StandardScaler()
        
    def fit(self, X):
        print(f"🤖 训练 {self.name} 检测器...")
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # 由于数据量可能很大，随机采样训练
        if len(X_scaled) > 10000:
            indices = np.random.choice(len(X_scaled), 10000, replace=False)
            X_sample = X_scaled[indices]
            print("  使用10000个样本进行训练")
        else:
            X_sample = X_scaled
        
        self.model = OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma=self.gamma
        )
        
        self.model.fit(X_sample)
        self.is_fitted = True
        
        print(f"✅ 训练完成")
        return self
        
    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        
        predictions_raw = self.model.predict(X_scaled)
        predictions = (predictions_raw == -1).astype(int)
        
        scores = -self.model.decision_function(X_scaled)
        
        return predictions, scores


class EllipticEnvelopeDetector(UnsupervisedDetectorBase):
    """椭圆包络检测器(鲁棒协方差估计)"""
    
    def __init__(self, contamination=0.1):
        super().__init__("EllipticEnvelope")
        self.contamination = contamination
        self.scaler = RobustScaler()  # 使用鲁棒标准化
        
    def fit(self, X):
        print(f"📊 训练 {self.name} 检测器...")
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"✅ 训练完成")
        return self
        
    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        
        predictions_raw = self.model.predict(X_scaled)
        predictions = (predictions_raw == -1).astype(int)
        
        scores = -self.model.decision_function(X_scaled)
        
        return predictions, scores


class PCADetector(UnsupervisedDetectorBase):
    """PCA重构误差检测器"""
    
    def __init__(self, contamination=0.1, n_components=0.95):
        super().__init__("PCA")
        self.contamination = contamination
        self.n_components = n_components
        self.scaler = StandardScaler()
        
    def fit(self, X):
        print(f"📈 训练 {self.name} 检测器...")
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # 重构数据
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        # 计算重构误差
        reconstruction_errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        
        self.threshold = np.percentile(reconstruction_errors, (1-self.contamination) * 100)
        self.is_fitted = True
        
        explained_ratio = np.sum(self.pca.explained_variance_ratio_)
        print(f"✅ 训练完成，保留 {self.pca.n_components_} 个主成分")
        print(f"   解释方差比例: {explained_ratio:.3f}")
        return self
        
    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        reconstruction_errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions, reconstruction_errors


def load_saved_model(filepath):
    """加载已保存的模型"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"📂 已加载模型: {model.name} from {filepath}")
    return model


def save_models(detectors, results, timestamp):
    """保存训练好的模型"""
    print("\n💾 保存训练好的模型...")
    
    saved_models = {}
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存所有模型
    for detector in detectors:
        if detector.is_fitted:
            filename = f"{detector.name}_{timestamp}.pkl"
            filepath = os.path.join(models_dir, filename)
            detector.save_model(filepath)
    
    # 找出最佳模型并特别保存
    if results:
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['predictions', 'scores']} 
                                 for r in results])
        
        # 最佳F1分数模型
        best_f1_idx = results_df['f1'].idxmax()
        best_f1_detector = results_df.iloc[best_f1_idx]['detector']
        best_f1_model = next(d for d in detectors if d.name == best_f1_detector and d.is_fitted)
        best_f1_path = os.path.join(models_dir, f"best_f1_{best_f1_detector}_{timestamp}.pkl")
        best_f1_model.save_model(best_f1_path)
        saved_models['best_f1'] = {'model': best_f1_detector, 'path': best_f1_path, 'score': results_df.iloc[best_f1_idx]['f1']}
        
        # 最低误检率模型
        best_fpr_idx = results_df['false_positive_rate'].idxmin()
        best_fpr_detector = results_df.iloc[best_fpr_idx]['detector']
        best_fpr_model = next(d for d in detectors if d.name == best_fpr_detector and d.is_fitted)
        best_fpr_path = os.path.join(models_dir, f"best_low_fpr_{best_fpr_detector}_{timestamp}.pkl")
        best_fpr_model.save_model(best_fpr_path)
        saved_models['best_fpr'] = {'model': best_fpr_detector, 'path': best_fpr_path, 'score': results_df.iloc[best_fpr_idx]['false_positive_rate']}
        
        # 最快检测模型
        best_speed_idx = results_df['test_time'].idxmin()
        best_speed_detector = results_df.iloc[best_speed_idx]['detector']
        best_speed_model = next(d for d in detectors if d.name == best_speed_detector and d.is_fitted)
        best_speed_path = os.path.join(models_dir, f"fastest_{best_speed_detector}_{timestamp}.pkl")
        best_speed_model.save_model(best_speed_path)
        saved_models['fastest'] = {'model': best_speed_detector, 'path': best_speed_path, 'score': results_df.iloc[best_speed_idx]['test_time']}
    
    print(f"✅ 模型保存完成，共保存到 {models_dir}/ 目录")
    return saved_models


def load_data(dataset_dirs, normal_keywords=['N', 'Normal'], seq_length=1000):
    """加载数据集"""
    print(f"📂 加载数据集: {dataset_dirs}")
    
    all_normal_data = []
    all_fault_data = []
    
    for dataset_dir in dataset_dirs:
        print(f"  处理: {dataset_dir}")
        
        if not os.path.exists(dataset_dir):
            print(f"    ⚠️ 目录不存在: {dataset_dir}")
            continue
        
        processor = OptimizedBearingDataProcessor(data_dir=dataset_dir, seq_length=seq_length)
        
        try:
            raw_data, raw_labels = processor.load_raw_data()
            
            for data, label in zip(raw_data, raw_labels):
                is_normal = any(keyword in label for keyword in normal_keywords)
                windows = processor.create_sliding_windows_vectorized(data)
                
                if len(windows) > 0:
                    if is_normal:
                        all_normal_data.extend(windows)
                    else:
                        all_fault_data.extend(windows)
            
            print(f"    ✅ 处理完成")
            
        except Exception as e:
            print(f"    ❌ 处理失败: {e}")
    
    normal_data = np.array(all_normal_data)
    fault_data = np.array(all_fault_data)
    
    print(f"\n📊 数据汇总:")
    print(f"  正常样本: {len(normal_data):,}")
    print(f"  故障样本: {len(fault_data):,}")
    
    return normal_data, fault_data


def evaluate_detector(detector, test_data, test_labels):
    """评估单个检测器"""
    print(f"\n🧪 测试 {detector.name}...")
    
    start_time = time.time()
    predictions, scores = detector.predict(test_data)
    test_time = time.time() - start_time
    
    # 计算指标
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    try:
        auc = roc_auc_score(test_labels, scores)
    except:
        auc = 0
    
    # 误检率分析
    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"  ✅ 测试完成 ({test_time:.2f}秒)")
    print(f"     准确率: {accuracy:.4f}")
    print(f"     F1分数: {f1:.4f}")
    print(f"     AUC: {auc:.4f}")
    print(f"     误检率: {fpr*100:.2f}%")
    print(f"     漏检率: {fnr*100:.2f}%")
    
    return {
        'detector': detector.name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'test_time': test_time,
        'predictions': predictions,
        'scores': scores
    }


def compare_all_methods(normal_data, fault_data, contamination=0.1, train_ratio=0.7):
    """对比所有无监督方法"""
    print("\n🚀 多方法无监督异常检测对比")
    print("=" * 80)
    
    # 准备数据
    train_size = int(len(normal_data) * train_ratio)
    train_normal = normal_data[:train_size]
    test_normal = normal_data[train_size:]
    
    test_data = np.vstack([test_normal, fault_data])
    test_labels = np.hstack([
        np.zeros(len(test_normal)),
        np.ones(len(fault_data))
    ])
    
    print(f"📊 数据分布:")
    print(f"  训练样本: {len(train_normal):,} (仅正常)")
    print(f"  测试样本: {len(test_data):,} (正常: {len(test_normal)}, 故障: {len(fault_data)})")
    print(f"  预期异常比例: {contamination*100:.1f}%")
    
    # 初始化所有检测器（跳过指定的两个模型）
    detectors = [
        MahalanobisDetector(contamination=contamination),
        IsolationForestDetector(contamination=contamination, n_estimators=100),
        LOFDetector(contamination=contamination, n_neighbors=20),
        # OneClassSVMDetector(contamination=contamination),      # 跳过训练
        # EllipticEnvelopeDetector(contamination=contamination), # 跳过训练
        PCADetector(contamination=contamination, n_components=0.95)
    ]
    
    # 显示跳过的模型
    skipped_models = ["OneClassSVM", "EllipticEnvelope"]
    print(f"⏭️ 跳过训练的模型: {', '.join(skipped_models)}")
    
    results = []
    trained_detectors = []
    
    # 训练和测试每个检测器
    for detector in detectors:
        print(f"\n{'='*60}")
        print(f"🔬 {detector.name} 检测器")
        print(f"{'='*60}")
        
        try:
            # 训练
            train_start = time.time()
            detector.fit(train_normal)
            train_time = time.time() - train_start
            
            print(f"⏱️ 训练时间: {train_time:.2f}秒")
            
            # 测试
            result = evaluate_detector(detector, test_data, test_labels)
            result['train_time'] = train_time
            results.append(result)
            trained_detectors.append(detector)
            
        except Exception as e:
            print(f"❌ {detector.name} 失败: {e}")
            # 添加失败结果
            results.append({
                'detector': detector.name,
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0,
                'false_positive_rate': 1, 'false_negative_rate': 1,
                'train_time': 0, 'test_time': 0,
                'predictions': None, 'scores': None
            })
    
    return results, test_data, test_labels, trained_detectors


def visualize_results(results, test_data, test_labels):
    """可视化对比结果"""
    print("\n📊 结果可视化...")
    
    # 准备数据
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['predictions', 'scores']} for r in results])
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('无监督异常检测方法对比', fontsize=16)
    
    # 1. 准确率对比
    axes[0, 0].bar(df['detector'], df['accuracy'])
    axes[0, 0].set_title('准确率对比')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. F1分数对比
    axes[0, 1].bar(df['detector'], df['f1'])
    axes[0, 1].set_title('F1分数对比')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. AUC对比
    axes[0, 2].bar(df['detector'], df['auc'])
    axes[0, 2].set_title('AUC对比')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].set_ylim(0, 1)
    
    # 4. 误检率对比
    axes[1, 0].bar(df['detector'], df['false_positive_rate'] * 100)
    axes[1, 0].set_title('误检率对比 (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. 训练时间对比
    axes[1, 1].bar(df['detector'], df['train_time'])
    axes[1, 1].set_title('训练时间对比 (秒)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. 测试时间对比
    axes[1, 2].bar(df['detector'], df['test_time'])
    axes[1, 2].set_title('测试时间对比 (秒)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df


def print_summary(results_df):
    """打印总结报告"""
    print("\n🏆 性能排名总结")
    print("=" * 80)
    
    metrics = ['accuracy', 'f1', 'auc', 'false_positive_rate']
    metric_names = ['准确率', 'F1分数', 'AUC', '误检率']
    
    for metric, name in zip(metrics, metric_names):
        print(f"\n📈 {name} 排名:")
        if metric == 'false_positive_rate':
            # 误检率越低越好
            sorted_df = results_df.sort_values(metric)
        else:
            # 其他指标越高越好
            sorted_df = results_df.sort_values(metric, ascending=False)
        
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            value = row[metric]
            if metric == 'false_positive_rate':
                print(f"  {i}. {row['detector']:15s}: {value*100:6.2f}%")
            else:
                print(f"  {i}. {row['detector']:15s}: {value:6.4f}")
    
    # 综合推荐
    print(f"\n🎯 推荐方案:")
    
    # 最低误检率
    best_fpr = results_df.loc[results_df['false_positive_rate'].idxmin()]
    print(f"  最低误检率: {best_fpr['detector']} ({best_fpr['false_positive_rate']*100:.2f}%)")
    
    # 最高F1分数
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    print(f"  最佳综合性能: {best_f1['detector']} (F1={best_f1['f1']:.4f})")
    
    # 最快速度
    best_time = results_df.loc[results_df['test_time'].idxmin()]
    print(f"  最快检测速度: {best_time['detector']} ({best_time['test_time']:.2f}秒)")


def main():
    """主函数"""
    print("🚀 多种无监督异常检测方法对比")
    print("=" * 80)
    
    # 配置参数
    DATASET_DIRS = ['bearing_dataset', 'bearing_dataset1']
    NORMAL_KEYWORDS = ['N', 'Normal']
    SEQ_LENGTH = 1000
    CONTAMINATION = 0.1  # 预期异常比例
    TRAIN_RATIO = 0.7
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. 加载数据
        normal_data, fault_data = load_data(DATASET_DIRS, NORMAL_KEYWORDS, SEQ_LENGTH)
        
        if len(normal_data) == 0 or len(fault_data) == 0:
            print("❌ 数据加载失败")
            return None, None, None
        
        # 2. 对比所有方法
        results, test_data, test_labels, trained_detectors = compare_all_methods(
            normal_data, fault_data, CONTAMINATION, TRAIN_RATIO
        )
        
        # 3. 保存训练好的模型
        saved_models = save_models(trained_detectors, results, timestamp)
        
        # 4. 可视化结果
        results_df = visualize_results(results, test_data, test_labels)
        
        # 5. 打印总结
        print_summary(results_df)
        
        # 6. 保存结果
        results_df.to_csv(f'comparison_results_{timestamp}.csv', index=False)
        
        total_time = time.time() - start_time
        print(f"\n🎉 对比完成! 总耗时: {total_time:.2f}秒")
        print(f"📝 结果已保存: comparison_results_{timestamp}.csv")
        
        return results, results_df, saved_models
        
    except Exception as e:
        print(f"\n❌ 对比失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def demo_model_usage(saved_models):
    """演示如何使用保存的模型"""
    print("\n🔍 模型使用演示")
    print("=" * 80)
    
    if not saved_models:
        print("❌ 没有可用的保存模型")
        return
    
    print("📋 可用的预训练模型:")
    for category, info in saved_models.items():
        print(f"  {category}: {info['model']} (分数: {info['score']:.4f})")
        print(f"    文件路径: {info['path']}")
    
    print("\n💡 使用示例代码:")
    print("```python")
    print("# 1. 加载最佳F1分数模型")
    if 'best_f1' in saved_models:
        print(f"best_model = load_saved_model('{saved_models['best_f1']['path']}')")
    
    print("\n# 2. 对新数据进行预测")
    print("predictions, scores = best_model.predict(your_new_data)")
    print("print(f'检测到 {np.sum(predictions)} 个异常样本')")
    
    print("\n# 3. 批量处理")
    print("for data_batch in your_data_batches:")
    print("    preds, scores = best_model.predict(data_batch)")
    print("    # 处理预测结果...")
    print("```")


def create_model_comparison_report(results_df, saved_models, timestamp):
    """创建详细的模型对比报告"""
    print("\n📄 生成详细对比报告...")
    
    report_content = f"""
# 无监督异常检测模型对比报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 实验概述
本次实验对比了多种无监督异常检测算法在轴承故障检测任务上的性能表现。

## 2. 参与对比的模型
"""
    
    for _, row in results_df.iterrows():
        report_content += f"- **{row['detector']}**: "
        if row['f1'] > 0:
            report_content += f"F1={row['f1']:.4f}, 误检率={row['false_positive_rate']*100:.2f}%\n"
        else:
            report_content += "训练失败\n"
    
    report_content += f"""
## 3. 性能排名

### 3.1 F1分数排名
"""
    f1_sorted = results_df.sort_values('f1', ascending=False)
    for i, (_, row) in enumerate(f1_sorted.iterrows(), 1):
        report_content += f"{i}. {row['detector']}: {row['f1']:.4f}\n"
    
    report_content += f"""
### 3.2 误检率排名（越低越好）
"""
    fpr_sorted = results_df.sort_values('false_positive_rate')
    for i, (_, row) in enumerate(fpr_sorted.iterrows(), 1):
        report_content += f"{i}. {row['detector']}: {row['false_positive_rate']*100:.2f}%\n"
    
    report_content += f"""
### 3.3 检测速度排名
"""
    speed_sorted = results_df.sort_values('test_time')
    for i, (_, row) in enumerate(speed_sorted.iterrows(), 1):
        report_content += f"{i}. {row['detector']}: {row['test_time']:.2f}秒\n"
    
    if saved_models:
        report_content += f"""
## 4. 推荐模型

### 4.1 最佳综合性能
- 模型: {saved_models['best_f1']['model']}
- F1分数: {saved_models['best_f1']['score']:.4f}
- 文件: {saved_models['best_f1']['path']}

### 4.2 最低误检率
- 模型: {saved_models['best_fpr']['model']}
- 误检率: {saved_models['best_fpr']['score']*100:.2f}%
- 文件: {saved_models['best_fpr']['path']}

### 4.3 最快检测速度
- 模型: {saved_models['fastest']['model']}
- 检测时间: {saved_models['fastest']['score']:.2f}秒
- 文件: {saved_models['fastest']['path']}
"""
    
    report_content += f"""
## 5. 使用建议

1. **生产环境部署**: 推荐使用误检率最低的模型，减少误报
2. **实时监控**: 推荐使用检测速度最快的模型
3. **综合考虑**: 推荐使用F1分数最高的模型

## 6. 模型文件说明

所有训练好的模型已保存在 `saved_models/` 目录下，可直接使用 `load_saved_model()` 函数加载。
"""
    
    # 保存报告
    report_filename = f'model_comparison_report_{timestamp}.md'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📄 详细报告已保存: {report_filename}")
    return report_filename


if __name__ == "__main__":
    results, results_df, saved_models = main()
    
    # 使用示例
    if results is not None:
        print("\n" + "="*80)
        print("🎯 实验总结")
        print("="*80)
        
        # 演示模型使用
        demo_model_usage(saved_models)
        
        # 生成详细报告
        if results_df is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_model_comparison_report(results_df, saved_models, timestamp)
        
        print(f"\n✅ 所有文件已保存:")
        print(f"  📊 性能对比: comparison_results_{timestamp}.csv")
        print(f"  📄 详细报告: model_comparison_report_{timestamp}.md")
        print(f"  💾 训练模型: saved_models/ 目录")
        
        print(f"\n🚀 快速开始:")
        print(f"```python")
        print(f"from {__file__.split('.')[0]} import load_saved_model")
        print(f"")
        if saved_models and 'best_f1' in saved_models:
            print(f"# 加载最佳模型")
            print(f"model = load_saved_model('{saved_models['best_f1']['path']}')")
            print(f"predictions, scores = model.predict(your_data)")
        print(f"```")
    else:
        print("\n💥 实验失败，请检查数据路径和依赖库!")
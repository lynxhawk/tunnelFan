import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# 导入数据处理器 (假设你有这个模块)
from data_loader import OptimizedBearingDataProcessor


class SimplifiedMahalanobisDetector:
    """简化版马氏距离异常检测器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.mean = None
        self.cov = None
        self.inv_cov = None
        self.threshold = None
        
    def fit(self, normal_data):
        """训练模型"""
        print("🔧 训练马氏距离异常检测器...")
        
        # 如果是3D数据，展平为2D
        if normal_data.ndim == 3:
            normal_data = normal_data.reshape(normal_data.shape[0], -1)
        
        # 标准化
        self.normal_scaled = self.scaler.fit_transform(normal_data)
        
        # 计算统计特征
        self.mean = np.mean(self.normal_scaled, axis=0)
        self.cov = np.cov(self.normal_scaled.T)
        
        # 处理奇异协方差矩阵
        try:
            self.inv_cov = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            print("⚠️ 协方差矩阵奇异，使用伪逆")
            self.inv_cov = np.linalg.pinv(self.cov)
        
        # 计算训练数据的马氏距离，设置阈值
        normal_distances = []
        for x in self.normal_scaled:
            try:
                dist = mahalanobis(x, self.mean, self.inv_cov)
                normal_distances.append(dist)
            except:
                # 如果计算失败，使用欧氏距离
                dist = np.linalg.norm(x - self.mean)
                normal_distances.append(dist)
        
        # 设置阈值为95%分位数
        self.threshold = np.percentile(normal_distances, 95)
        self.is_fitted = True
        
        print(f"✅ 训练完成")
        print(f"   训练样本数: {len(normal_data):,}")
        print(f"   特征维数: {normal_data.shape[1] if normal_data.ndim == 2 else normal_data.shape[1] * normal_data.shape[2]}")
        print(f"   阈值: {self.threshold:.6f}")
        
        return self
    
    def predict(self, test_data):
        """预测测试数据"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 如果是3D数据，展平为2D
        if test_data.ndim == 3:
            test_data = test_data.reshape(test_data.shape[0], -1)
        
        test_scaled = self.scaler.transform(test_data)
        
        distances = []
        for x in test_scaled:
            try:
                dist = mahalanobis(x, self.mean, self.inv_cov)
                distances.append(dist)
            except:
                # 如果计算失败，使用欧氏距离
                dist = np.linalg.norm(x - self.mean)
                distances.append(dist)
        
        distances = np.array(distances)
        predictions = (distances > self.threshold).astype(int)
        
        return predictions, distances
    
    def save_model(self, filepath):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        
        model_data = {
            'scaler': self.scaler,
            'mean': self.mean,
            'cov': self.cov,
            'inv_cov': self.inv_cov,
            'threshold': self.threshold,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.mean = model_data['mean']
        self.cov = model_data['cov']
        self.inv_cov = model_data['inv_cov']
        self.threshold = model_data['threshold']
        self.is_fitted = model_data['is_fitted']
        
        print(f"📂 模型已加载: {filepath}")
        return self


def load_combined_dataset(dataset_dirs, normal_keywords=['N', 'Normal'], seq_length=1000):
    """
    加载并合并多个数据集
    
    参数:
    - dataset_dirs: 数据集目录列表
    - normal_keywords: 正常数据关键词列表
    - seq_length: 序列长度
    
    返回:
    - normal_data: 正常数据
    - fault_data: 故障数据
    """
    print(f"📂 加载数据集: {dataset_dirs}")
    
    all_normal_data = []
    all_fault_data = []
    
    for dataset_dir in dataset_dirs:
        print(f"  处理数据集: {dataset_dir}")
        
        if not os.path.exists(dataset_dir):
            print(f"    ⚠️ 数据集目录不存在: {dataset_dir}")
            continue
        
        # 创建数据处理器
        processor = OptimizedBearingDataProcessor(data_dir=dataset_dir, seq_length=seq_length)
        
        try:
            # 加载原始数据
            raw_data, raw_labels = processor.load_raw_data()
            
            dataset_normal = []
            dataset_fault = []
            
            for data, label in zip(raw_data, raw_labels):
                # 判断是否为正常数据
                is_normal = any(keyword in label for keyword in normal_keywords)
                
                # 创建滑动窗口
                windows = processor.create_sliding_windows_vectorized(data)
                
                if len(windows) > 0:
                    if is_normal:
                        dataset_normal.extend(windows)
                    else:
                        dataset_fault.extend(windows)
            
            print(f"    ✅ 正常数据: {len(dataset_normal):,} 样本")
            print(f"    ✅ 故障数据: {len(dataset_fault):,} 样本")
            
            all_normal_data.extend(dataset_normal)
            all_fault_data.extend(dataset_fault)
            
        except Exception as e:
            print(f"    ❌ 处理数据集失败: {e}")
            continue
    
    normal_data = np.array(all_normal_data)
    fault_data = np.array(all_fault_data)
    
    print(f"\n📊 合并后数据统计:")
    print(f"  总正常数据: {len(normal_data):,} 样本")
    print(f"  总故障数据: {len(fault_data):,} 样本")
    print(f"  数据形状: {normal_data.shape if len(normal_data) > 0 else 'N/A'}")
    
    return normal_data, fault_data


def evaluate_model(predictions, scores, true_labels, method_name="Mahalanobis"):
    """评估模型性能"""
    print(f"\n📊 {method_name} 模型评估结果:")
    print("=" * 60)
    
    # 计算各项指标
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    # 打印分类报告
    report = classification_report(true_labels, predictions, 
                                target_names=['正常', '故障'], 
                                digits=4)
    print(report)
    
    # 打印详细指标
    print(f"📈 详细指标:")
    print(f"   准确率 (Accuracy): {accuracy:.4f}")
    print(f"   精确率 (Precision): {precision:.4f}")
    print(f"   召回率 (Recall): {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    
    # AUC分数
    try:
        auc = roc_auc_score(true_labels, scores)
        print(f"🎯 AUC Score: {auc:.4f}")
    except Exception as e:
        print(f"⚠️ 无法计算AUC: {e}")
        auc = 0
    
    # 绘制混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '故障'], 
                yticklabels=['正常', '故障'])
    plt.title(f'{method_name} 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def main():
    """主函数"""
    print("🚀 简化版马氏距离故障检测训练")
    print("=" * 80)
    
    # 配置参数
    dataset_dirs = ['bearing_dataset', 'bearing_dataset1']  # 两个数据集目录
    normal_keywords = ['N', 'Normal']  # 正常数据关键词
    seq_length = 1000  # 序列长度
    train_ratio = 0.7  # 训练数据比例
    
    # 模型保存配置
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'mahalanobis_model_{timestamp}.pkl')
    
    # 1. 加载数据
    print("\n" + "="*60)
    print("1️⃣ 数据加载")
    print("="*60)
    
    start_time = time.time()
    normal_data, fault_data = load_combined_dataset(
        dataset_dirs=dataset_dirs,
        normal_keywords=normal_keywords,
        seq_length=seq_length
    )
    load_time = time.time() - start_time
    
    if len(normal_data) == 0:
        print("❌ 未找到正常数据，请检查数据集路径和关键词")
        return
    
    if len(fault_data) == 0:
        print("❌ 未找到故障数据，请检查数据集路径")
        return
    
    print(f"⏱️ 数据加载时间: {load_time:.2f}秒")
    
    # 2. 准备训练和测试数据
    print("\n" + "="*60)
    print("2️⃣ 数据准备")
    print("="*60)
    
    # 分割正常数据：部分用于训练，全部用于测试
    train_size = int(len(normal_data) * train_ratio)
    train_normal = normal_data[:train_size]
    
    # 测试数据包含所有正常数据和故障数据
    test_data = np.vstack([normal_data, fault_data])
    test_labels = np.hstack([
        np.zeros(len(normal_data)),  # 正常=0
        np.ones(len(fault_data))     # 故障=1
    ])
    
    print(f"📊 数据分布:")
    print(f"   训练用正常数据: {len(train_normal):,} 样本")
    print(f"   测试数据总计: {len(test_data):,} 样本")
    print(f"     - 正常: {len(normal_data):,} 样本")
    print(f"     - 故障: {len(fault_data):,} 样本")
    print(f"   故障比例: {len(fault_data)/len(test_data)*100:.1f}%")
    
    # 3. 训练模型
    print("\n" + "="*60)
    print("3️⃣ 模型训练")
    print("="*60)
    
    detector = SimplifiedMahalanobisDetector()
    
    train_start = time.time()
    detector.fit(train_normal)
    train_time = time.time() - train_start
    
    print(f"⏱️ 训练时间: {train_time:.2f}秒")
    
    # 4. 模型测试
    print("\n" + "="*60)
    print("4️⃣ 模型测试")
    print("="*60)
    
    test_start = time.time()
    predictions, scores = detector.predict(test_data)
    test_time = time.time() - test_start
    
    print(f"⏱️ 测试时间: {test_time:.2f}秒")
    print(f"📊 预测结果:")
    print(f"   预测为正常: {np.sum(predictions == 0):,} 样本")
    print(f"   预测为故障: {np.sum(predictions == 1):,} 样本")
    
    # 5. 性能评估
    print("\n" + "="*60)
    print("5️⃣ 性能评估")
    print("="*60)
    
    metrics = evaluate_model(predictions, scores, test_labels, "马氏距离检测器")
    
    # 6. 保存模型
    print("\n" + "="*60)
    print("6️⃣ 保存模型")
    print("="*60)
    
    detector.save_model(model_path)
    
    # 保存训练信息
    training_info = {
        'timestamp': timestamp,
        'dataset_dirs': dataset_dirs,
        'normal_keywords': normal_keywords,
        'seq_length': seq_length,
        'train_ratio': train_ratio,
        'train_samples': len(train_normal),
        'test_samples': len(test_data),
        'normal_samples': len(normal_data),
        'fault_samples': len(fault_data),
        'train_time': train_time,
        'test_time': test_time,
        'load_time': load_time,
        'metrics': metrics,
        'model_path': model_path
    }
    
    info_path = os.path.join(save_dir, f'training_info_{timestamp}.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("马氏距离故障检测器训练信息\n")
        f.write("="*50 + "\n")
        for key, value in training_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📝 训练信息已保存: {info_path}")
    
    # 7. 总结
    print("\n" + "="*80)
    print("🎯 训练完成总结")
    print("="*80)
    print(f"✅ 模型类型: 马氏距离异常检测")
    print(f"✅ 数据集: {', '.join(dataset_dirs)}")
    print(f"✅ 训练样本: {len(train_normal):,} (正常数据)")
    print(f"✅ 测试样本: {len(test_data):,} (正常+故障)")
    print(f"✅ 主要性能指标:")
    print(f"   - AUC: {metrics['auc']:.4f}")
    print(f"   - 准确率: {metrics['accuracy']:.4f}")
    print(f"   - 精确率: {metrics['precision']:.4f}")
    print(f"   - 召回率: {metrics['recall']:.4f}")
    print(f"   - F1分数: {metrics['f1']:.4f}")
    print(f"✅ 时间统计:")
    print(f"   - 数据加载: {load_time:.2f}秒")
    print(f"   - 模型训练: {train_time:.2f}秒")
    print(f"   - 模型测试: {test_time:.2f}秒")
    print(f"   - 总耗时: {load_time + train_time + test_time:.2f}秒")
    print(f"✅ 模型已保存: {model_path}")
    
    return detector, metrics, training_info


def load_and_test_model(model_path, test_data, test_labels):
    """加载已保存的模型进行测试"""
    print(f"\n🔍 加载模型进行测试: {model_path}")
    
    detector = SimplifiedMahalanobisDetector()
    detector.load_model(model_path)
    
    predictions, scores = detector.predict(test_data)
    metrics = evaluate_model(predictions, scores, test_labels, "加载的马氏距离检测器")
    
    return detector, predictions, scores, metrics


if __name__ == "__main__":
    # 执行训练
    try:
        detector, metrics, training_info = main()
        print("\n🎉 程序执行成功!")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 示例：如何加载已保存的模型
    # model_path = 'saved_models/mahalanobis_model_20240101_120000.pkl'  # 替换为实际路径
    # detector, predictions, scores, metrics = load_and_test_model(model_path, test_data, test_labels)
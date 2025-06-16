#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GAN的轴承故障域适应系统 - 主程序

系统架构:
1. 阶段1: 在HUST数据集(有标签)上预训练源域特征提取器 + 分类器
2. 阶段2: 固定源域特征提取器，训练目标域特征提取器使其特征无法被判别器区分
3. 阶段3: 使用训练好的目标域特征提取器 + 分类器进行CWRU数据分类

使用方法:
python domain_adaptation_main.py
"""

import os
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_binary import BinaryBearingDataProcessor
from gan_binary import (
    FeatureExtractor, Classifier, DomainDiscriminator, 
    DomainAdaptationTrainer
)


class CWRUDataProcessor(BinaryBearingDataProcessor):
    """CWRU数据集处理器 - 继承自二分类数据处理器"""
    
    def __init__(self, cwru_dir='bearing_dataset1', seq_length=1000, overlap_ratio=0.5):
        """
        初始化CWRU数据处理器
        
        参数:
        - cwru_dir: CWRU数据集目录
        - seq_length: 序列长度
        - overlap_ratio: 窗口重叠比例
        """
        # 调用父类初始化，只使用一个数据集目录
        super().__init__(
            dataset1_dir=cwru_dir,
            dataset2_dir=cwru_dir,  # 这里设置相同，后面会重写
            seq_length=seq_length,
            overlap_ratio=overlap_ratio
        )
        self.cwru_dir = cwru_dir
    
    def get_cwru_label_from_filename(self, filename):
        """
        从CWRU文件名获取标签
        根据你的CWRU数据集命名规则修改这个函数
        
        示例命名规则:
        - Normal_xxx.csv -> 0 (正常)
        - Ball_xxx.csv, Inner_xxx.csv, Outer_xxx.csv -> 1 (故障)
        """
        filename_lower = filename.lower()
        
        # 正常样本
        if 'normal' in filename_lower or 'baseline' in filename_lower:
            return 0
        # 故障样本
        elif any(fault_type in filename_lower for fault_type in 
                ['ball', 'inner', 'outer', 'fault', 'defect']):
            return 1
        else:
            # 默认根据文件名开头判断
            return 0 if filename.startswith('N') else 1
    
    def prepare_cwru_data(self, labeled=False, max_files=None):
        """
        准备CWRU数据
        
        参数:
        - labeled: 是否需要标签（测试时设为True，域适应训练时设为False）
        - max_files: 最大文件数限制
        """
        import time
        start_time = time.time()
        
        all_data = []
        all_labels = []
        
        print(f"🚀 正在加载CWRU数据集... (labeled={labeled})")
        
        if not os.path.exists(self.cwru_dir):
            raise ValueError(f"CWRU数据目录不存在: {self.cwru_dir}")
        
        csv_files = [f for f in os.listdir(self.cwru_dir) if f.endswith('.csv')]
        csv_files.sort()
        
        if max_files:
            csv_files = csv_files[:max_files]
        
        normal_count = 0
        fault_count = 0
        
        for i, file_name in enumerate(csv_files):
            print(f"\r处理文件 {i+1}/{len(csv_files)}: {file_name[:30]}...", end='', flush=True)
            file_path = os.path.join(self.cwru_dir, file_name)
            
            # 获取标签（如果需要）
            if labeled:
                label = self.get_cwru_label_from_filename(file_name)
                if label == 0:
                    normal_count += 1
                else:
                    fault_count += 1
            else:
                label = 0  # 域适应训练时不使用标签
            
            try:
                import pandas as pd
                df = pd.read_csv(file_path, header=None, dtype=np.float32, engine='c')
                data = df.values.astype(np.float32)
                
                if data.ndim > 1:
                    data = data.flatten()
                
                all_data.append(data)
                all_labels.append(label)
                
            except Exception as e:
                print(f"\n  - 错误: 无法加载文件 {file_name}: {e}")
                continue
        
        print(f"\n✅ CWRU数据加载完成:")
        print(f"  - 总文件数: {len(all_data)}")
        if labeled:
            print(f"  - 正常样本: {normal_count} 个文件")
            print(f"  - 故障样本: {fault_count} 个文件")
        
        # 创建序列数据
        all_sequences = []
        all_sequence_labels = []
        
        print(f"\n🔄 正在创建CWRU序列数据...")
        window_start = time.time()
        
        for i, (data, label) in enumerate(zip(all_data, all_labels)):
            print(f"\r处理文件 {i+1}/{len(all_data)}", end='', flush=True)
            
            windows = self.create_sliding_windows_vectorized(data)
            
            if len(windows) > 0:
                all_sequences.append(windows)
                all_sequence_labels.extend([label] * len(windows))
        
        print(f"\n")
        
        if all_sequences:
            X = np.vstack(all_sequences)
        else:
            raise ValueError("没有生成任何CWRU序列数据")
            
        y = np.array(all_sequence_labels)
        
        # 调整维度为 (batch_size, seq_length, 1)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"✅ CWRU数据准备完成:")
        print(f"  - 总序列数: {len(X):,}")
        print(f"  - 序列形状: {X.shape}")
        if labeled:
            print(f"  - 正常序列: {np.sum(y == 0):,} ({np.sum(y == 0)/len(y)*100:.1f}%)")
            print(f"  - 故障序列: {np.sum(y == 1):,} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        
        return X, y
    
    def get_cwru_dataloaders(self, labeled=False, batch_size=64, max_files=None):
        """获取CWRU数据加载器"""
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        
        X, y = self.prepare_cwru_data(labeled=labeled, max_files=max_files)
        
        # 标准化（如果需要的话）
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped).reshape(X.shape)
        else:
            X_scaled = X
        
        # 创建数据集和加载器
        dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.LongTensor(y)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if not labeled else False,  # 训练时打乱，测试时不打乱
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader


def check_data_directories():
    """检查数据目录是否存在"""
    required_dirs = ['bearing_dataset', 'bearing_dataset1']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
        else:
            csv_files = [f for f in os.listdir(dir_name) if f.endswith('.csv')]
            print(f"📁 {dir_name}: 发现 {len(csv_files)} 个CSV文件")
    
    if missing_dirs:
        print(f"❌ 缺少数据目录: {missing_dirs}")
        print("请确保数据目录存在并包含CSV文件")
        return False
    
    return True


def main():
    """主函数 - 完整的域适应流程"""
    
    print("🎯 基于GAN的轴承故障域适应系统")
    print("=" * 60)
    print("📋 系统配置:")
    print("  - 源域: bearing_dataset (HUST数据集，有标签)")
    print("  - 目标域: bearing_dataset1 (CWRU数据集，无标签)")
    print("  - 方法: 对抗域适应 (Adversarial Domain Adaptation)")
    print("  - 架构: 特征提取器 + 分类器 + 域判别器")
    print("=" * 60)
    
    # 检查数据目录
    if not check_data_directories():
        print("\n请按以下步骤准备数据:")
        print("1. 创建 'bearing_dataset' 目录，放入HUST数据集CSV文件")
        print("2. 创建 'bearing_dataset1' 目录，放入CWRU数据集CSV文件")
        print("3. 确保文件命名规则正确")
        return None
    
    # 配置参数
    config = {
        'seq_length': 1000,           # 序列长度
        'overlap_ratio': 0.5,         # 重叠比例
        'batch_size': 64,             # 批次大小
        'feature_dim': 256,           # 特征维度
        'pretrain_epochs': 50,        # 源域预训练轮数
        'adaptation_epochs': 100,     # 域适应训练轮数
        'lambda_domain': 1.0,         # 域损失权重
        'max_source_files': None,     # 最大源域文件数
        'max_target_files': None,     # 最大目标域文件数
    }
    
    print(f"\n📊 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # 步骤1: 数据准备
        print(f"\n{'='*20} 步骤1: 数据准备 {'='*20}")
        
        # HUST数据处理器（源域）
        hust_processor = BinaryBearingDataProcessor(
            dataset1_dir='bearing_dataset',
            dataset2_dir='bearing_dataset',  # 这里只用dataset1
            seq_length=config['seq_length'],
            overlap_ratio=config['overlap_ratio']
        )
        
        # 准备HUST数据（源域，有标签）
        print("📚 准备HUST源域数据...")
        X_source, y_source = hust_processor.prepare_dataset_for_training(
            dataset_type='dataset1',
            max_files=config['max_source_files']
        )
        
        # 标准化源域数据
        from sklearn.model_selection import train_test_split
        X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
            X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
        )
        
        # 标准化处理
        X_source_reshaped = X_source_train.reshape(-1, X_source_train.shape[-1])
        hust_processor.scaler.fit(X_source_reshaped)
        X_source_train_scaled = hust_processor.scaler.transform(X_source_reshaped).reshape(X_source_train.shape)
        
        X_source_val_reshaped = X_source_val.reshape(-1, X_source_val.shape[-1])
        X_source_val_scaled = hust_processor.scaler.transform(X_source_val_reshaped).reshape(X_source_val.shape)
        
        # 创建源域数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        
        source_train_dataset = TensorDataset(
            torch.FloatTensor(X_source_train_scaled),
            torch.LongTensor(y_source_train)
        )
        source_train_loader = DataLoader(
            source_train_dataset, batch_size=config['batch_size'], 
            shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available()
        )
        
        # CWRU数据处理器（目标域）
        cwru_processor = CWRUDataProcessor(
            cwru_dir='bearing_dataset1',
            seq_length=config['seq_length'],
            overlap_ratio=config['overlap_ratio']
        )
        
        # 设置CWRU处理器使用相同的标准化器
        cwru_processor.scaler = hust_processor.scaler
        
        # 准备CWRU数据（目标域，无标签用于域适应）
        print("🧪 准备CWRU目标域数据（无标签）...")
        target_train_loader = cwru_processor.get_cwru_dataloaders(
            labeled=False,  # 域适应训练时不使用标签
            batch_size=config['batch_size'],
            max_files=config['max_target_files']
        )
        
        # 准备CWRU测试数据（有标签用于评估）
        print("🔍 准备CWRU测试数据（有标签）...")
        target_test_loader = cwru_processor.get_cwru_dataloaders(
            labeled=True,   # 测试时需要标签
            batch_size=config['batch_size'],
            max_files=config['max_target_files']
        )
        
        # 步骤2: 模型初始化
        print(f"\n{'='*20} 步骤2: 模型初始化 {'='*20}")
        
        # 源域特征提取器
        source_feature_extractor = FeatureExtractor(
            input_length=config['seq_length'],
            input_channels=1,
            feature_dim=config['feature_dim']
        )
        
        # 目标域特征提取器
        target_feature_extractor = FeatureExtractor(
            input_length=config['seq_length'],
            input_channels=1,
            feature_dim=config['feature_dim']
        )
        
        # 分类器
        classifier = Classifier(
            feature_dim=config['feature_dim'],
            num_classes=2
        )
        
        # 域判别器
        discriminator = DomainDiscriminator(
            feature_dim=config['feature_dim']
        )
        
        print(f"📋 模型信息:")
        print(f"  源域特征提取器: {sum(p.numel() for p in source_feature_extractor.parameters()):,} 参数")
        print(f"  目标域特征提取器: {sum(p.numel() for p in target_feature_extractor.parameters()):,} 参数")
        print(f"  分类器: {sum(p.numel() for p in classifier.parameters()):,} 参数")
        print(f"  域判别器: {sum(p.numel() for p in discriminator.parameters()):,} 参数")
        
        # 步骤3: 域适应训练器初始化
        print(f"\n{'='*20} 步骤3: 域适应训练器初始化 {'='*20}")
        
        trainer = DomainAdaptationTrainer(
            source_feature_extractor=source_feature_extractor,
            target_feature_extractor=target_feature_extractor,
            classifier=classifier,
            discriminator=discriminator,
            device='auto'
        )
        
        # 步骤4: 源域预训练
        print(f"\n{'='*20} 步骤4: 源域预训练 {'='*20}")
        print("🚀 开始在HUST数据集上预训练源域模型...")
        
        trainer.pretrain_source_model(
            source_loader=source_train_loader,
            epochs=config['pretrain_epochs']
        )
        
        # 步骤5: 域适应训练
        print(f"\n{'='*20} 步骤5: 域适应训练 {'='*20}")
        print("🔄 开始对抗域适应训练...")
        
        trainer.train_domain_adaptation(
            source_loader=source_train_loader,
            target_loader=target_train_loader,
            epochs=config['adaptation_epochs'],
            lambda_domain=config['lambda_domain']
        )
        
        # 步骤6: 训练结果可视化
        print(f"\n{'='*20} 步骤6: 训练结果可视化 {'='*20}")
        
        trainer.plot_training_history()
        
        # 步骤7: 目标域评估
        print(f"\n{'='*20} 步骤7: 目标域评估 {'='*20}")
        
        results = trainer.evaluate_target_domain(
            target_loader_with_labels=target_test_loader,
            class_names=['Normal', 'Fault']
        )
        
        # 步骤8: 保存结果
        print(f"\n{'='*20} 步骤8: 保存结果 {'='*20}")
        
        import pickle
        with open('domain_adaptation_results.pkl', 'wb') as f:
            pickle.dump({
                'config': config,
                'results': results,
                'training_history': trainer.train_history
            }, f)
        
        print(f"\n🎉 域适应训练完成!")
        print(f"📊 最终性能指标:")
        print(f"  目标域准确率: {results['accuracy']*100:.2f}%")
        print(f"  目标域精确率: {results['precision']*100:.2f}%")
        print(f"  目标域召回率: {results['recall']*100:.2f}%")
        print(f"  目标域F1分数: {results['f1_score']*100:.2f}%")
        print(f"  域混淆度: {trainer.best_domain_confusion:.4f}")
        
        print(f"\n💾 结果已保存:")
        print(f"  模型文件: best_domain_adaptation_model.pth")
        print(f"  结果文件: domain_adaptation_results.pkl")
        print(f"  图表文件: domain_adaptation_training_history.png")
        
        return trainer, results
        
    except Exception as e:
        print(f"\n❌ 域适应训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_cwru_with_adapted_model(model_path, cwru_data_path):
    """使用域适应模型对CWRU数据进行预测"""
    
    print(f"🔍 使用域适应模型预测: {cwru_data_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    try:
        # 加载域适应模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 重建模型
        target_fe = FeatureExtractor(input_length=1000, input_channels=1, feature_dim=256)
        classifier = Classifier(feature_dim=256, num_classes=2)
        
        target_fe.load_state_dict(checkpoint['target_fe_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target_fe.to(device)
        classifier.to(device)
        
        target_fe.eval()
        classifier.eval()
        
        # 处理CWRU数据
        cwru_processor = CWRUDataProcessor(cwru_dir=cwru_data_path)
        test_loader = cwru_processor.get_cwru_dataloaders(labeled=True, batch_size=64)
        
        # 预测
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data, _ in tqdm(test_loader, desc="预测"):
                batch_data = batch_data.to(device)
                
                features = target_fe(batch_data)
                outputs = classifier(features)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        print(f"📊 预测完成:")
        print(f"  总样本数: {len(all_predictions)}")
        print(f"  预测为正常: {np.sum(np.array(all_predictions) == 0)}")
        print(f"  预测为故障: {np.sum(np.array(all_predictions) == 1)}")
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probs
        }
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode():
    """交互式模式"""
    
    print("\n🎮 进入交互式模式")
    print("=" * 40)
    
    while True:
        print("\n请选择操作:")
        print("1. 完整域适应训练流程")
        print("2. 使用已训练模型预测CWRU数据")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 开始完整域适应训练流程...")
            result = main()
            if result:
                print("✅ 域适应训练完成!")
            else:
                print("❌ 域适应训练失败!")
        
        elif choice == '2':
            model_path = input("请输入模型文件路径 (默认: best_domain_adaptation_model.pth): ").strip()
            if not model_path:
                model_path = 'best_domain_adaptation_model.pth'
                
            cwru_path = input("请输入CWRU数据目录路径 (默认: bearing_dataset1): ").strip()
            if not cwru_path:
                cwru_path = 'bearing_dataset1'
            
            predict_cwru_with_adapted_model(model_path, cwru_path)
        
        elif choice == '3':
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='基于GAN的轴承故障域适应系统')
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--model_path', default='best_domain_adaptation_model.pth', 
                       help='模型文件路径')
    parser.add_argument('--cwru_path', default='bearing_dataset1', help='CWRU数据路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 开始域适应训练模式...")
        main()
    
    elif args.mode == 'predict':
        print("🔍 开始预测模式...")
        predict_cwru_with_adapted_model(args.model_path, args.cwru_path)
    
    else:  # interactive mode
        interactive_mode()
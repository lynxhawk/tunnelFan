#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承故障检测二分类系统 - 主程序
使用三层CNN进行跨数据集训练和测试

数据集结构:
- bearing_dataset: HUST数据集 (N开头=正常, 其他=故障)
- bearing_dataset1: 另一数据集 (包含Normal=正常, 其他=故障)

训练策略: 在bearing_dataset上训练，在bearing_dataset1上测试

使用方法:
python main_binary.py
"""

import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# 检查PyTorch版本和GPU
print(f"🔍 PyTorch版本: {torch.__version__}")
print(f"🖥️  CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU设备: {torch.cuda.get_device_name(0)}")


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
    """主函数"""
    
    print("🎯 轴承故障检测二分类系统")
    print("=" * 60)
    print("📋 系统配置:")
    print("  - 训练数据集: bearing_dataset (HUST)")
    print("  - 测试数据集: bearing_dataset1")
    print("  - 模型: 三层CNN")
    print("  - 任务: 二分类 (正常 vs 故障)")
    print("=" * 60)
    
    # 检查数据目录
    if not check_data_directories():
        print("\n请按以下步骤准备数据:")
        print("1. 创建 'bearing_dataset' 目录，放入HUST数据集CSV文件")
        print("2. 创建 'bearing_dataset1' 目录，放入测试数据集CSV文件")
        print("3. 确保文件命名规则:")
        print("   - bearing_dataset: N开头的文件为正常，其他为故障")
        print("   - bearing_dataset1: 包含'Normal'的文件为正常，其他为故障")
        return None
    
    # 配置参数
    config = {
        'seq_length': 1000,           # 序列长度
        'overlap_ratio': 0.5,         # 重叠比例
        'batch_size': 64,             # 批次大小
        'learning_rate': 0.001,       # 学习率
        'epochs': 100,                # 训练轮数
        'early_stopping_patience': 15, # 早停耐心值
        'dropout_rate': 0.3,          # Dropout比例
        'weight_decay': 1e-4,         # 权重衰减
        'val_split': 0.2,             # 验证集比例
        'max_train_files': None,      # 最大训练文件数 (None=全部)
        'max_test_files': None,       # 最大测试文件数 (None=全部)
    }
    
    print(f"\n📊 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # 步骤1: 数据处理
        print(f"\n{'='*20} 步骤1: 数据处理 {'='*20}")
        
        from data_binary import BinaryBearingDataProcessor
        
        processor = BinaryBearingDataProcessor(
            dataset1_dir='bearing_dataset',
            dataset2_dir='bearing_dataset1',
            seq_length=config['seq_length'],
            overlap_ratio=config['overlap_ratio']
        )
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = processor.get_optimized_data_loaders(
            train_dataset='dataset1',         # 在bearing_dataset上训练
            test_dataset='dataset2',          # 在bearing_dataset1上测试
            batch_size=config['batch_size'],
            max_train_files=config['max_train_files'],
            max_test_files=config['max_test_files'],
            val_split=config['val_split']
        )
        
        # 步骤2: 模型初始化
        print(f"\n{'='*20} 步骤2: 模型初始化 {'='*20}")
        
        from cnn_binary import ThreeLayerCNN, BinaryBearingTrainer
        
        model = ThreeLayerCNN(
            input_length=config['seq_length'],
            input_channels=1,
            num_classes=2,
            dropout_rate=config['dropout_rate']
        )
        
        # 显示模型信息
        model_info = model.get_model_info()
        print(f"📋 模型架构信息:")
        print(f"  总参数: {model_info['total_parameters']:,}")
        print(f"  可训练参数: {model_info['trainable_parameters']:,}")
        print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
        
        # 步骤3: 训练器初始化
        print(f"\n{'='*20} 步骤3: 训练器初始化 {'='*20}")
        
        trainer = BinaryBearingTrainer(
            model=model,
            device='auto',
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 步骤4: 模型训练
        print(f"\n{'='*20} 步骤4: 模型训练 {'='*20}")
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            early_stopping_patience=config['early_stopping_patience'],
            save_best_model=True,
            model_save_path='best_binary_cnn_model.pth'
        )
        
        # 步骤5: 训练结果可视化
        print(f"\n{'='*20} 步骤5: 训练结果可视化 {'='*20}")
        
        trainer.plot_training_history()
        
        # 步骤6: 模型测试
        print(f"\n{'='*20} 步骤6: 模型测试 {'='*20}")
        
        test_results = trainer.evaluate(
            test_loader, 
            class_names=['Normal', 'Fault']
        )
        
        # 步骤7: 性能总结
        print(f"\n{'='*20} 步骤7: 性能总结 {'='*20}")
        
        processor.print_performance_summary()
        
        print(f"\n🎉 训练和测试完成!")
        print(f"📊 最终性能指标:")
        print(f"  测试准确率: {test_results['accuracy']*100:.2f}%")
        print(f"  测试精确率: {test_results['precision']*100:.2f}%")
        print(f"  测试召回率: {test_results['recall']*100:.2f}%")
        print(f"  测试F1分数: {test_results['f1_score']*100:.2f}%")
        
        # 保存详细结果
        import pickle
        with open('training_results.pkl', 'wb') as f:
            pickle.dump({
                'config': config,
                'model_info': model_info,
                'test_results': test_results,
                'training_history': trainer.train_history
            }, f)
        
        print(f"\n💾 结果已保存:")
        print(f"  模型文件: best_binary_cnn_model.pth")
        print(f"  结果文件: training_results.pkl")
        print(f"  图表文件: training_history.png, confusion_matrix.png")
        
        return trainer, test_results, processor
        
    except ImportError as e:
        print(f"\n❌ 导入模块失败: {e}")
        print("请确保以下文件存在于当前目录:")
        print("  - data_binary.py")
        print("  - cnn_binary.py")
        return None
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_single_file(model_path, csv_file_path):
    """对单个CSV文件进行预测"""
    
    print(f"🔍 对单个文件进行预测: {csv_file_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    if not os.path.exists(csv_file_path):
        print(f"❌ 数据文件不存在: {csv_file_path}")
        return None
    
    try:
        # 加载模型
        from cnn_binary import ThreeLayerCNN, BinaryBearingTrainer
        model = ThreeLayerCNN(input_length=1000, input_channels=1, num_classes=2)
        trainer = BinaryBearingTrainer(model=model, device='auto')
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载和预处理数据
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # 读取CSV文件
        df = pd.read_csv(csv_file_path, header=None, dtype=np.float32)
        data = df.values.astype(np.float32).flatten()
        
        # 创建滑动窗口
        seq_length = 1000
        overlap_ratio = 0.5
        step_size = int(seq_length * (1 - overlap_ratio))
        
        windows = []
        for i in range(0, len(data) - seq_length + 1, step_size):
            window = data[i:i + seq_length]
            windows.append(window)
        
        if not windows:
            print("❌ 文件太短，无法创建序列窗口")
            return None
        
        X = np.array(windows)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # 预测
        predictions = []
        confidences = []
        
        for window in X_scaled:
            result = trainer.predict_single_sample(window)
            predictions.append(result['prediction'][0])
            confidences.append(result['confidence'][0])
        
        # 汇总结果
        predictions = np.array(predictions)
        final_prediction = 1 if np.sum(predictions) > len(predictions) // 2 else 0
        avg_confidence = np.mean(confidences)
        
        result = {
            'file_path': csv_file_path,
            'final_prediction': 'Fault' if final_prediction == 1 else 'Normal',
            'prediction_confidence': avg_confidence,
            'total_windows': len(predictions),
            'fault_windows': np.sum(predictions == 1),
            'normal_windows': np.sum(predictions == 0)
        }
        
        print(f"📊 预测结果:")
        print(f"  文件: {os.path.basename(csv_file_path)}")
        print(f"  预测结果: {result['final_prediction']}")
        print(f"  置信度: {result['prediction_confidence']:.4f}")
        print(f"  总窗口数: {result['total_windows']}")
        print(f"  故障窗口: {result['fault_windows']} ({result['fault_windows']/result['total_windows']*100:.1f}%)")
        print(f"  正常窗口: {result['normal_windows']} ({result['normal_windows']/result['total_windows']*100:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_predict_directory(model_path, data_directory):
    """对整个目录的文件进行批量预测"""
    
    print(f"📁 批量预测目录: {data_directory}")
    
    if not os.path.exists(data_directory):
        print(f"❌ 目录不存在: {data_directory}")
        return None
    
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ 目录中没有CSV文件")
        return None
    
    print(f"📊 发现 {len(csv_files)} 个CSV文件")
    
    results = []
    correct_predictions = 0
    total_files = len(csv_files)
    
    for i, filename in enumerate(csv_files, 1):
        print(f"\n处理文件 {i}/{total_files}: {filename}")
        
        file_path = os.path.join(data_directory, filename)
        result = predict_single_file(model_path, file_path)
        
        if result:
            results.append(result)
            
            # 简单的真实标签推断（基于文件名）
            if 'Normal' in filename or filename.startswith('N'):
                true_label = 'Normal'
            else:
                true_label = 'Fault'
            
            if result['final_prediction'] == true_label:
                correct_predictions += 1
            
            print(f"  真实标签: {true_label}")
            print(f"  预测正确: {'✅' if result['final_prediction'] == true_label else '❌'}")
    
    # 批量预测总结
    if results:
        accuracy = correct_predictions / total_files
        print(f"\n🎉 批量预测完成!")
        print(f"📊 总体统计:")
        print(f"  处理文件数: {len(results)}")
        print(f"  预测准确率: {accuracy*100:.2f}% ({correct_predictions}/{total_files})")
        
        # 保存结果
        import pickle
        with open('batch_prediction_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 批量预测结果已保存: batch_prediction_results.pkl")
        
        return results
    
    return None


def interactive_mode():
    """交互式模式"""
    
    print("\n🎮 进入交互式模式")
    print("=" * 40)
    
    while True:
        print("\n请选择操作:")
        print("1. 完整训练流程")
        print("2. 单文件预测")
        print("3. 批量预测")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            print("\n🚀 开始完整训练流程...")
            result = main()
            if result:
                print("✅ 训练完成!")
            else:
                print("❌ 训练失败!")
        
        elif choice == '2':
            model_path = input("请输入模型文件路径 (默认: best_binary_cnn_model.pth): ").strip()
            if not model_path:
                model_path = 'best_binary_cnn_model.pth'
                
            csv_path = input("请输入CSV文件路径: ").strip()
            
            if csv_path:
                predict_single_file(model_path, csv_path)
            else:
                print("❌ 请提供CSV文件路径")
        
        elif choice == '3':
            model_path = input("请输入模型文件路径 (默认: best_binary_cnn_model.pth): ").strip()
            if not model_path:
                model_path = 'best_binary_cnn_model.pth'
                
            data_dir = input("请输入数据目录路径: ").strip()
            
            if data_dir:
                batch_predict_directory(model_path, data_dir)
            else:
                print("❌ 请提供数据目录路径")
        
        elif choice == '4':
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='轴承故障检测二分类系统')
    parser.add_argument('--mode', choices=['train', 'predict', 'batch', 'interactive'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--model_path', default='best_binary_cnn_model.pth', 
                       help='模型文件路径')
    parser.add_argument('--file_path', help='单文件预测的CSV文件路径')
    parser.add_argument('--data_dir', help='批量预测的数据目录路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 开始训练模式...")
        main()
    
    elif args.mode == 'predict':
        if args.file_path:
            print("🔍 开始单文件预测...")
            predict_single_file(args.model_path, args.file_path)
        else:
            print("❌ 请提供 --file_path 参数")
    
    elif args.mode == 'batch':
        if args.data_dir:
            print("📁 开始批量预测...")
            batch_predict_directory(args.model_path, args.data_dir)
        else:
            print("❌ 请提供 --data_dir 参数")
    
    else:  # interactive mode
        interactive_mode()
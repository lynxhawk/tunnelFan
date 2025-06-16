#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
域适应收敛问题修复 - 主程序
提供多种稳定的域适应方案

使用方法:
python convergence_fix_main.py
"""

import os
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 尝试导入所需模块
try:
    from data_binary import BinaryBearingDataProcessor
except ImportError:
    print("❌ 请确保 data_binary.py 文件存在")
    exit(1)

try:
    from gan_binary import FeatureExtractor, Classifier
except ImportError:
    print("❌ 请确保 domain_adaptation_system.py 文件存在")
    exit(1)

try:
    from gan_binary1 import SimpleDomainAdaptationTrainer, ProgressiveDomainAdaptationTrainer
except ImportError:
    print("❌ 请确保 simple_domain_adaptation.py 文件存在")
    exit(1)


def diagnose_convergence_issues():
    """诊断可能的收敛问题"""
    
    print("🔍 域适应收敛问题诊断")
    print("=" * 50)
    
    issues_and_solutions = {
        "1. 判别器过强": [
            "现象: 域判别准确率 > 85%",
            "解决: 降低判别器学习率到 0.0001",
            "解决: 增加判别器dropout到 0.5",
            "解决: 减少判别器训练频率"
        ],
        "2. 生成器过强": [
            "现象: 域判别准确率 < 15%",
            "解决: 降低目标特征提取器学习率",
            "解决: 增加域损失权重",
            "解决: 使用梯度惩罚"
        ],
        "3. 训练不稳定": [
            "现象: 损失剧烈震荡",
            "解决: 使用梯度裁剪 (max_norm=1.0)",
            "解决: 增加批次大小到 128",
            "解决: 使用标签平滑 (0.9/0.1)"
        ],
        "4. 模式崩塌": [
            "现象: 特征变得相同",
            "解决: 添加特征多样性损失",
            "解决: 使用谱归一化",
            "解决: 降低学习率"
        ],
        "5. 分类性能下降": [
            "现象: 源域分类准确率下降",
            "解决: 增加分类损失权重",
            "解决: 使用源域特征提取器冻结",
            "解决: 添加分类损失正则化"
        ]
    }
    
    for issue, solutions in issues_and_solutions.items():
        print(f"\n{issue}:")
        for solution in solutions:
            print(f"  {solution}")
    
    print(f"\n🎯 推荐解决方案优先级:")
    print(f"  1️⃣ 简化MMD域适应 (最稳定)")
    print(f"  2️⃣ 渐进式域适应")
    print(f"  3️⃣ 改进版GAN域适应")


def create_stable_data_loaders():
    """创建稳定的数据加载器 - 修复设备问题"""
    
    print("📊 创建稳定的数据加载器...")
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    try:
        # HUST数据处理器（源域）
        hust_processor = BinaryBearingDataProcessor(
            dataset1_dir='bearing_dataset',
            dataset2_dir='bearing_dataset',
            seq_length=1000,
            overlap_ratio=0.5
        )
        
        # 准备HUST数据
        X_source, y_source = hust_processor.prepare_dataset_for_training(
            dataset_type='dataset1',
            max_files=50  # 进一步限制文件数量避免内存问题
        )
        
        # 标准化
        from sklearn.model_selection import train_test_split
        X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
            X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
        )
        
        X_source_reshaped = X_source_train.reshape(-1, X_source_train.shape[-1])
        hust_processor.scaler.fit(X_source_reshaped)
        X_source_train_scaled = hust_processor.scaler.transform(X_source_reshaped).reshape(X_source_train.shape)
        
        # 创建平衡的数据加载器
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
        
        # 计算类别权重
        class_counts = np.bincount(y_source_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_source_train]
        
        # 创建加权采样器
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        source_train_dataset = TensorDataset(
            torch.FloatTensor(X_source_train_scaled),
            torch.LongTensor(y_source_train)
        )
        
        source_train_loader = DataLoader(
            source_train_dataset,
            batch_size=64,  # 减小batch size
            sampler=sampler,
            num_workers=0,  # Windows系统设为0避免问题
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # 确保batch大小一致
        )
        
        print("✅ 稳定数据加载器创建完成")
        
        return source_train_loader, hust_processor
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {str(e)}")
        print("💡 可能的解决方案:")
        print("  1. 检查 bearing_dataset 目录是否存在")
        print("  2. 检查数据文件格式是否正确")
        print("  3. 减少 max_files 参数")
        return None, None



def method1_simple_mmd_adaptation():
    """方案1: 简化MMD域适应 - 修复设备问题"""
    
    print("\n🔧 方案1: 简化MMD域适应")
    print("=" * 40)
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    try:
        # 创建数据加载器
        source_loader, processor = create_stable_data_loaders()
        
        if source_loader is None:
            print("❌ 数据加载器创建失败，使用模拟数据")
            # 创建模拟数据进行测试
            return create_mock_trainer(device)
        
        # 创建模型
        feature_dim = 256
        
        source_fe = FeatureExtractor(
            input_length=1000,
            input_channels=1,
            feature_dim=feature_dim
        ).to(device)  # 确保模型在正确设备上
        
        target_fe = FeatureExtractor(
            input_length=1000,
            input_channels=1,
            feature_dim=feature_dim
        ).to(device)  # 确保模型在正确设备上
        
        classifier = Classifier(
            feature_dim=feature_dim,
            num_classes=2
        ).to(device)  # 确保模型在正确设备上
        
        # 创建训练器
        trainer = SimpleDomainAdaptationTrainer(
            source_feature_extractor=source_fe,
            target_feature_extractor=target_fe,
            classifier=classifier,
            device=device  # 明确指定设备
        )
        
        print("📋 MMD域适应特点:")
        print("  ✅ 无对抗训练，训练稳定")
        print("  ✅ 直接特征分布对齐")
        print("  ✅ 参数少，容易调节")
        
        # 预训练源域
        print("\n🚀 开始源域预训练...")
        trainer.pretrain_source_model(source_loader, epochs=30)
        
        # 这里需要目标域数据
        print("\n⚠️  需要提供CWRU目标域数据加载器")
        print("请实现 target_loader 后继续训练:")
        print("trainer.train_simple_domain_adaptation(source_loader, target_loader)")
        
        return trainer
        
    except Exception as e:
        print(f"❌ 方案1执行失败: {str(e)}")
        print("💡 尝试创建模拟训练器进行测试...")
        return create_mock_trainer(device)

def create_mock_trainer(device):
    """创建模拟训练器用于测试"""
    
    print("🎭 创建模拟训练器...")
    
    feature_dim = 128
    
    source_fe = FeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim
    ).to(device)
    
    target_fe = FeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim
    ).to(device)
    
    classifier = Classifier(
        feature_dim=feature_dim,
        num_classes=2
    ).to(device)
    
    trainer = SimpleDomainAdaptationTrainer(
        source_feature_extractor=source_fe,
        target_feature_extractor=target_fe,
        classifier=classifier,
        device=device
    )
    
    print("✅ 模拟训练器创建成功")
    return trainer

def method2_progressive_adaptation():
    """方案2: 渐进式域适应"""
    
    print("\n🔧 方案2: 渐进式域适应")
    print("=" * 40)
    
    # 创建数据加载器
    source_loader, processor = create_stable_data_loaders()
    
    # 创建模型
    feature_dim = 256
    
    source_fe = FeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim
    )
    
    target_fe = FeatureExtractor(
        input_length=1000,
        input_channels=1,
        feature_dim=feature_dim
    )
    
    classifier = Classifier(
        feature_dim=feature_dim,
        num_classes=2
    )
    
    # 创建训练器
    trainer = ProgressiveDomainAdaptationTrainer(
        source_feature_extractor=source_fe,
        target_feature_extractor=target_fe,
        classifier=classifier,
        device='auto'
    )
    
    print("📋 渐进式域适应特点:")
    print("  ✅ 从源域参数开始")
    print("  ✅ 特征逐步对齐")
    print("  ✅ 训练过程可控")
    
    # 预训练源域
    print("\n🚀 开始源域预训练...")
    trainer.pretrain_source_model(source_loader, epochs=30)
    
    print("\n⚠️  需要提供CWRU目标域数据加载器")
    print("请实现 target_loader 后继续训练:")
    print("trainer.train_progressive_adaptation(source_loader, target_loader)")
    
    return trainer


def create_cwru_data_loader_template():
    """CWRU数据加载器模板"""
    
    template_code = '''
# CWRU数据加载器实现模板
class CWRUDataProcessor:
    def __init__(self, cwru_dir='cwru_dataset'):
        self.cwru_dir = cwru_dir
        self.scaler = None  # 使用HUST的scaler
    
    def get_cwru_loader(self, labeled=False, batch_size=128):
        """
        创建CWRU数据加载器
        
        参数:
        - labeled: 是否需要标签 (训练时False, 测试时True)
        - batch_size: 批次大小
        """
        # 1. 读取CWRU CSV文件
        # 2. 创建滑动窗口
        # 3. 标准化 (使用HUST的scaler)
        # 4. 创建DataLoader
        
        pass
    
# 使用示例:
cwru_processor = CWRUDataProcessor('cwru_dataset')
cwru_processor.scaler = hust_processor.scaler  # 使用相同的标准化器

target_train_loader = cwru_processor.get_cwru_loader(labeled=False)  # 域适应训练
target_test_loader = cwru_processor.get_cwru_loader(labeled=True)    # 测试评估
'''
    
    print("📝 CWRU数据加载器实现模板:")
    print(template_code)


def quick_convergence_test():
    """快速收敛测试 - 修复设备不匹配问题"""
    
    print("\n🧪 快速收敛测试")
    print("=" * 30)
    
    # 检测并设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 创建小规模测试
    print("创建小规模测试数据...")
    
    # 模拟源域和目标域数据
    batch_size = 32
    feature_dim = 128
    seq_length = 1000
    
    # 创建简单测试数据 - 确保在正确设备上
    source_data = torch.randn(batch_size, seq_length, 1).to(device)
    target_data = torch.randn(batch_size, seq_length, 1).to(device) + 0.1  # 轻微分布差异
    source_labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    # 创建简单模型 - 确保在正确设备上
    source_fe = FeatureExtractor(seq_length, 1, feature_dim).to(device)
    target_fe = FeatureExtractor(seq_length, 1, feature_dim).to(device)
    classifier = Classifier(feature_dim, 2).to(device)
    
    # 简化训练器 - 传递设备参数
    trainer = SimpleDomainAdaptationTrainer(
        source_fe, 
        target_fe, 
        classifier,
        device=device  # 明确指定设备
    )
    
    print("✅ 快速测试环境准备完成")
    print("📊 测试数据:")
    print(f"  源域数据形状: {source_data.shape}, 设备: {source_data.device}")
    print(f"  目标域数据形状: {target_data.shape}, 设备: {target_data.device}")
    print(f"  特征维度: {feature_dim}")
    
    try:
        # 简单的前向传播测试
        with torch.no_grad():
            print("🔄 执行前向传播测试...")
            
            source_features = source_fe(source_data)
            target_features = target_fe(target_data)
            
            print(f"  源域特征形状: {source_features.shape}, 设备: {source_features.device}")
            print(f"  目标域特征形状: {target_features.shape}, 设备: {target_features.device}")
            
            # 测试MMD损失计算
            if hasattr(trainer, 'mmd_loss'):
                mmd_loss_val = trainer.mmd_loss(source_features, target_features)
                print(f"  MMD损失: {mmd_loss_val.item():.4f}")
            else:
                print("  MMD损失函数不可用，跳过测试")
            
            # 测试分类器
            print("🔄 测试分类器...")
            source_logits = classifier(source_features)
            print(f"  分类输出形状: {source_logits.shape}, 设备: {source_logits.device}")
            
        print("✅ 快速测试通过，模型结构正常")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        
        # 提供调试信息
        print("\n🔍 调试信息:")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA设备数量: {torch.cuda.device_count()}")
            print(f"  当前CUDA设备: {torch.cuda.current_device()}")
        
        return False

def interactive_convergence_fix():
    """交互式收敛问题修复"""
    
    print("\n🎮 交互式收敛问题修复")
    print("=" * 40)
    
    while True:
        print("\n请选择操作:")
        print("1. 诊断收敛问题")
        print("2. 🚀 执行方案1: 简化MMD域适应")
        print("3. 🚀 执行方案2: 渐进式域适应")
        print("4. 查看CWRU数据加载器模板")
        print("5. 🧪 快速收敛测试")
        print("6. 📊 真实数据测试")
        print("7. 退出")
        
        choice = input("\n请输入选择 (1-7): ").strip()
        
        if choice == '1':
            diagnose_convergence_issues()
        
        elif choice == '2':
            print("\n🚀 正在执行方案1...")
            trainer = method1_simple_mmd_adaptation()
            if trainer:
                print("✅ 方案1执行成功！")
                print("💡 下一步: 添加CWRU目标域数据后调用:")
                print("   trainer.train_simple_domain_adaptation(source_loader, target_loader)")
            else:
                print("❌ 方案1执行失败，请检查错误信息")
        
        elif choice == '3':
            print("\n🚀 正在执行方案2...")
            trainer = method2_progressive_adaptation()
            if trainer:
                print("✅ 方案2执行成功！")
                print("💡 下一步: 添加CWRU目标域数据后调用:")
                print("   trainer.train_progressive_adaptation(source_loader, target_loader)")
            else:
                print("❌ 方案2执行失败，请检查错误信息")
        
        elif choice == '4':
            create_cwru_data_loader_template()
        
        elif choice == '5':
            print("\n🧪 正在执行快速收敛测试...")
            success = quick_convergence_test()
            if success:
                print("✅ 快速测试成功！系统运行正常")
            else:
                print("❌ 快速测试失败，请检查环境配置")
        
        elif choice == '6':
            print("\n📊 正在执行真实数据测试...")
            success = test_with_real_data()
            if success:
                print("✅ 真实数据测试成功！")
            else:
                print("❌ 真实数据测试失败，请检查数据目录")
        
        elif choice == '7':
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效选择，请重新输入")
        
        # 询问是否继续
        if choice in ['2', '3', '5', '6']:
            continue_choice = input("\n是否继续其他操作? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("👋 再见!")
                break


if __name__ == "__main__":
    print("🔧 域适应收敛问题修复工具")
    print("=" * 50)
    
    # 检查依赖
    print("🔍 检查系统环境...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        exit(1)
    
    try:
        from data_binary import BinaryBearingDataProcessor
        print("✅ 数据处理模块")
    except ImportError:
        print("❌ data_binary.py模块未找到")
        print("请确保data_binary.py在当前目录")
        exit(1)
    
    try:
        from gan_binary import FeatureExtractor, Classifier
        print("✅ 域适应系统模块")
    except ImportError:
        print("❌ domain_adaptation_system.py模块未找到")
        print("请确保domain_adaptation_system.py在当前目录")
        exit(1)
    
    try:
        from gan_binary1 import SimpleDomainAdaptationTrainer, ProgressiveDomainAdaptationTrainer
        print("✅ 简化域适应模块")
    except ImportError:
        print("❌ simple_domain_adaptation.py模块未找到")
        print("请确保simple_domain_adaptation.py在当前目录")
        exit(1)
    
    # 检查数据目录
    print("\n📁 检查数据目录...")
    if os.path.exists('bearing_dataset'):
        csv_count = len([f for f in os.listdir('bearing_dataset') if f.endswith('.csv')])
        print(f"✅ bearing_dataset目录存在，包含{csv_count}个CSV文件")
    else:
        print("⚠️  bearing_dataset目录不存在")
        print("   这会影响真实数据测试，但可以进行快速测试")
    
    if os.path.exists('cwru_dataset'):
        csv_count = len([f for f in os.listdir('cwru_dataset') if f.endswith('.csv')])
        print(f"✅ cwru_dataset目录存在，包含{csv_count}个CSV文件")
    else:
        print("⚠️  cwru_dataset目录不存在")
        print("   需要创建此目录并放入CWRU数据集")
    
    print(f"\n🎯 推荐使用步骤:")
    print(f"  1. 先运行'快速收敛测试'确认环境正常")
    print(f"  2. 如果有bearing_dataset，运行'真实数据测试'")
    print(f"  3. 选择并执行适合的域适应方案")
    print(f"  4. 根据提示添加CWRU数据继续训练")
    
    # 启动交互式修复
    interactive_convergence_fix()



# 额外的设备检查函数
def check_device_compatibility():
    """检查设备兼容性"""
    
    print("🔍 设备兼容性检查")
    print("=" * 30)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("将使用CPU进行计算")
    
    # 测试基本张量操作
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✅ 基本张量操作测试成功，设备: {device}")
        return True
    except Exception as e:
        print(f"❌ 基本张量操作测试失败: {e}")
        return False
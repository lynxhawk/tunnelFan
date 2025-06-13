#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CWRU数据集MAT文件批量提取和转换工具
- 递归遍历多层文件夹
- 提取所有MAT文件到同一目录
- 批量转换MAT为CSV
"""

import scipy.io
import pandas as pd
import numpy as np
import os
import glob
import shutil
from pathlib import Path

def find_all_mat_files(root_dir):
    """递归查找所有MAT文件"""
    mat_files = []
    root_path = Path(root_dir)
    
    print(f"🔍 正在递归搜索目录: {root_dir}")
    
    # 递归查找所有.mat文件
    for mat_file in root_path.rglob("*.mat"):
        mat_files.append(mat_file)
        print(f"  📄 找到: {mat_file.relative_to(root_path)}")
    
    print(f"🎯 总共找到 {len(mat_files)} 个MAT文件")
    return mat_files

def extract_mat_files_to_folder(mat_files, output_dir="extracted_mats"):
    """提取所有MAT文件到同一个文件夹"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n📁 开始提取MAT文件到: {output_path.absolute()}")
    
    extracted_files = []
    name_conflicts = {}
    
    for mat_file in mat_files:
        # 获取原始文件名
        original_name = mat_file.name
        
        # 处理重名文件
        if original_name in name_conflicts:
            name_conflicts[original_name] += 1
            # 添加序号避免重名
            base_name = mat_file.stem
            extension = mat_file.suffix
            new_name = f"{base_name}_{name_conflicts[original_name]}{extension}"
        else:
            name_conflicts[original_name] = 0
            new_name = original_name
        
        # 目标路径
        target_path = output_path / new_name
        
        try:
            # 复制文件
            shutil.copy2(mat_file, target_path)
            extracted_files.append(target_path)
            print(f"  ✅ 提取: {mat_file.name} -> {new_name}")
        except Exception as e:
            print(f"  ❌ 提取失败 {mat_file.name}: {e}")
    
    print(f"\n🎉 提取完成! 共提取 {len(extracted_files)} 个文件")
    return extracted_files, output_path

def convert_mat_to_csv(mat_file, variable_name=None):
    """转换单个MAT文件为CSV"""
    try:
        print(f"正在处理: {mat_file.name}")
        
        # 读取MAT文件
        mat_data = scipy.io.loadmat(mat_file)
        
        # 如果没指定变量名，找第一个数据变量
        if variable_name is None:
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not data_keys:
                print(f"  ⚠️ 没有找到数据变量")
                return False
            variable_name = data_keys[0]
            print(f"  📊 使用变量: {variable_name}")
        
        # 检查变量是否存在
        if variable_name not in mat_data:
            print(f"  ❌ 变量 '{variable_name}' 不存在")
            available_vars = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"  可用变量: {available_vars}")
            return False
        
        # 获取数据
        data = mat_data[variable_name]
        print(f"  📏 数据形状: {data.shape}")
        print(f"  🔢 数据类型: {data.dtype}")
        
        # 转换为DataFrame
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                # 一维数组转换为单列
                df = pd.DataFrame(data, columns=[variable_name])
            elif data.ndim == 2:
                # 二维数组
                df = pd.DataFrame(data)
            else:
                # 多维数组，展平为二维
                print(f"  🔄 多维数组，重新整形为二维")
                reshaped_data = data.reshape(data.shape[0], -1)
                df = pd.DataFrame(reshaped_data)
        else:
            print(f"  ❌ 数据类型不支持: {type(data)}")
            return False
        
        # 生成输出文件名
        csv_file = mat_file.with_suffix('.csv')
        
        # 保存CSV
        df.to_csv(csv_file, index=False)
        
        print(f"  ✅ 成功转换: {csv_file.name}")
        print(f"  📊 CSV形状: {df.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        return False

def batch_convert_extracted_files(extracted_files, variable_name=None):
    """批量转换提取的MAT文件为CSV"""
    
    print(f"\n🚀 开始批量转换 {len(extracted_files)} 个MAT文件...")
    
    successful = 0
    failed = 0
    
    for mat_file in extracted_files:
        if convert_mat_to_csv(mat_file, variable_name):
            successful += 1
        else:
            failed += 1
        print()  # 空行分隔
    
    print("=" * 50)
    print(f"🎉 批量转换完成!")
    print(f"✅ 成功: {successful} 个文件")
    print(f"❌ 失败: {failed} 个文件")

def explore_mat_structure(mat_files, max_files=5):
    """探索MAT文件结构"""
    print(f"\n🔍 探索MAT文件结构 (显示前{max_files}个文件):\n")
    
    for i, mat_file in enumerate(mat_files[:max_files]):
        print(f"📄 {mat_file.name}")
        try:
            mat_data = scipy.io.loadmat(mat_file)
            variables = [k for k in mat_data.keys() if not k.startswith('__')]
            
            for var in variables:
                data = mat_data[var]
                print(f"  📊 变量: {var}")
                print(f"     类型: {type(data).__name__}")
                if hasattr(data, 'shape'):
                    print(f"     形状: {data.shape}")
                if hasattr(data, 'dtype'):
                    print(f"     数据类型: {data.dtype}")
        
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
        
        print()

def main():
    print("🎯 CWRU数据集MAT文件批量提取和转换工具")
    print("=" * 50)
    
    # 检查依赖
    try:
        import pandas as pd
        import scipy.io
        print("✅ 依赖库检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install pandas scipy")
        return
    
    # 获取CWRU数据集路径
    print("\n请输入数据集的根目录路径:")
    print("(直接回车使用当前目录)")
    root_dir = input("路径: ").strip()
    
    if not root_dir:
        root_dir = "."
    
    if not os.path.exists(root_dir):
        print(f"❌ 路径不存在: {root_dir}")
        return
    
    print(f"📁 使用目录: {os.path.abspath(root_dir)}")
    
    # 查找所有MAT文件
    mat_files = find_all_mat_files(root_dir)
    
    if not mat_files:
        print("❌ 没有找到MAT文件")
        return
    
    # 选择操作
    print("\n请选择操作:")
    print("1. 仅探索MAT文件结构")
    print("2. 提取所有MAT文件到同一文件夹")
    print("3. 提取并转换为CSV（自动选择变量）")
    print("4. 提取并转换为CSV（指定变量名）")
    
    choice = input("\n请输入选择 (1/2/3/4): ").strip()
    
    if choice == "1":
        explore_mat_structure(mat_files)
    
    elif choice == "2":
        output_dir = input("输入输出文件夹名称 (默认: extracted_mats): ").strip()
        if not output_dir:
            output_dir = "extracted_mats"
        
        extracted_files, output_path = extract_mat_files_to_folder(mat_files, output_dir)
        print(f"\n📁 所有MAT文件已提取到: {output_path.absolute()}")
    
    elif choice == "3":
        output_dir = input("输入输出文件夹名称 (默认: extracted_mats): ").strip()
        if not output_dir:
            output_dir = "extracted_mats"
        
        # 先探索文件结构
        print("\n📋 先探索文件结构...")
        explore_mat_structure(mat_files, max_files=3)
        
        # 提取文件
        extracted_files, output_path = extract_mat_files_to_folder(mat_files, output_dir)
        
        # 转换为CSV
        batch_convert_extracted_files(extracted_files)
        
        print(f"\n🎉 完成! 所有文件在: {output_path.absolute()}")
    
    elif choice == "4":
        output_dir = input("输入输出文件夹名称 (默认: extracted_mats): ").strip()
        if not output_dir:
            output_dir = "extracted_mats"
        
        var_name = input("请输入变量名: ").strip()
        if not var_name:
            print("❌ 变量名不能为空")
            return
        
        # 提取文件
        extracted_files, output_path = extract_mat_files_to_folder(mat_files, output_dir)
        
        # 转换为CSV
        batch_convert_extracted_files(extracted_files, variable_name=var_name)
        
        print(f"\n🎉 完成! 所有文件在: {output_path.absolute()}")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
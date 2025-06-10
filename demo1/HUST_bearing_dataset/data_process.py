#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速MAT转CSV脚本 - 放在MAT文件同一目录下使用
"""

import scipy.io
import pandas as pd
import numpy as np
import os
import glob

def convert_mat_to_csv(mat_file, variable_name=None):
    """转换单个MAT文件为CSV"""
    try:
        print(f"正在处理: {mat_file}")
        
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
        base_name = os.path.splitext(mat_file)[0]
        csv_file = f"{base_name}.csv"
        
        # 保存CSV
        df.to_csv(csv_file, index=False)
        
        print(f"  ✅ 成功转换: {csv_file}")
        print(f"  📊 CSV形状: {df.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        return False

def batch_convert_current_folder(variable_name=None, output_folder=None):
    """批量转换当前文件夹的MAT文件"""
    
    # 找到所有MAT文件
    mat_files = glob.glob("*.mat")
    
    if not mat_files:
        print("❌ 当前文件夹没有找到MAT文件")
        print(f"当前目录: {os.getcwd()}")
        print("文件列表:")
        all_files = os.listdir('.')
        for f in all_files[:10]:  # 显示前10个文件
            print(f"  {f}")
        return
    
    print(f"🔍 找到 {len(mat_files)} 个MAT文件:")
    for f in mat_files:
        print(f"  📄 {f}")
    
    print(f"\n🚀 开始批量转换...")
    
    successful = 0
    failed = 0
    
    for mat_file in mat_files:
        if convert_mat_to_csv(mat_file, variable_name):
            successful += 1
        else:
            failed += 1
        print()  # 空行分隔
    
    print("=" * 50)
    print(f"🎉 批量转换完成!")
    print(f"✅ 成功: {successful} 个文件")
    print(f"❌ 失败: {failed} 个文件")
    
    # 显示生成的CSV文件
    csv_files = glob.glob("*.csv")
    if csv_files:
        print(f"\n📁 生成的CSV文件:")
        for csv_file in csv_files:
            size = os.path.getsize(csv_file)
            print(f"  📊 {csv_file} ({size} 字节)")

def explore_mat_files():
    """探索当前文件夹的MAT文件内容"""
    mat_files = glob.glob("*.mat")
    
    if not mat_files:
        print("❌ 当前文件夹没有找到MAT文件")
        return
    
    print(f"🔍 探索 {len(mat_files)} 个MAT文件:\n")
    
    for mat_file in mat_files:
        print(f"📄 {mat_file}")
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

if __name__ == "__main__":
    print("🎯 MAT文件快速转换为CSV")
    print("=" * 30)
    
    # 检查依赖
    try:
        import pandas as pd
        import scipy.io
        print("✅ 依赖库检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install pandas scipy")
        exit(1)
    
    print(f"📁 当前目录: {os.getcwd()}")
    
    # 选择操作
    print("\n请选择操作:")
    print("1. 探索MAT文件内容")
    print("2. 批量转换为CSV（自动选择变量）")
    print("3. 批量转换为CSV（指定变量名）")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        explore_mat_files()
    
    elif choice == "2":
        batch_convert_current_folder()
    
    elif choice == "3":
        var_name = input("请输入变量名: ").strip()
        if var_name:
            batch_convert_current_folder(variable_name=var_name)
        else:
            print("❌ 变量名不能为空")
    
    else:
        print("❌ 无效选择，直接进行批量转换")
        batch_convert_current_folder()
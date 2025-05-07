import os
import glob
import re
import pandas as pd
import numpy as np

def recursive_find_csv_files(data_dir):
    """
    递归查找给定目录及其所有子目录中的所有CSV文件
    
    参数:
    - data_dir: 数据根目录
    
    返回:
    - 所有CSV文件的完整路径列表
    """
    csv_files = []
    
    # 搜索直接位于data_dir中的CSV文件
    csv_files.extend(glob.glob(os.path.join(data_dir, "*.csv")))
    
    # 搜索子目录
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            # 从子目录名称中提取故障信息
            fault_info = extract_fault_info_from_dirname(subdir)
            
            # 搜索子目录中的CSV文件
            for csv_file in glob.glob(os.path.join(subdir_path, "*.csv")):
                csv_files.append((csv_file, fault_info))
    
    return csv_files

def extract_fault_info_from_dirname(dirname):
    """
    从目录名称中提取故障信息
    
    参数:
    - dirname: 目录名称
    
    返回:
    - 故障信息字典
    """
    fault_info = {}
    
    # 检测故障严重程度（如0.7mm, 0.9mm等）
    severity_match = re.search(r'(\d+\.\d+)mm', dirname)
    if severity_match:
        fault_info['severity'] = severity_match.group(1)
    
    # 检测故障类型（内圈、外圈、健康）
    if 'inner' in dirname.lower():
        fault_info['type'] = 'inner'
    elif 'outer' in dirname.lower():
        fault_info['type'] = 'outer'
    elif 'healthy' in dirname.lower():
        fault_info['type'] = 'healthy'
        fault_info['severity'] = '0.0'  # 健康状态没有故障严重程度
    
    return fault_info

def parse_filename_with_folder_info(filename, folder_info=None):
    """
    从文件名和文件夹信息中解析故障类型、严重程度和负载信息
    
    参数:
    - filename: 文件名
    - folder_info: 从文件夹名称中提取的信息
    
    返回:
    - 解析后的标签
    """
    # 初始化默认值
    fault_type = 'unknown'
    severity = 'unknown'
    load = 'unknown'
    pulley = 'unknown'
    
    # 如果有文件夹信息，首先使用它
    if folder_info:
        if 'type' in folder_info:
            fault_type = folder_info['type']
        if 'severity' in folder_info:
            severity = folder_info['severity']
    
    # 从文件名中提取额外信息，可能会覆盖文件夹信息
    # 提取故障类型
    if 'healthy' in filename.lower():
        fault_type = 'healthy'
        severity = '0.0'  # 健康状态的严重程度为0
    elif 'inner' in filename.lower():
        fault_type = 'inner'
    elif 'outer' in filename.lower():
        fault_type = 'outer'
    
    # 从文件名中提取严重程度（如果文件夹信息中没有）
    if severity == 'unknown':
        severity_match = re.search(r'(\d+\.\d+)', filename)
        if severity_match:
            severity = severity_match.group(1)
    
    # 提取负载信息
    if '100w' in filename.lower() or '100watt' in filename.lower():
        load = '100W'
    elif '200w' in filename.lower() or '200watt' in filename.lower():
        load = '200W'
    elif '300w' in filename.lower() or '300watt' in filename.lower():
        load = '300W'
    
    # 确定是否有皮带轮
    if 'pulley' in filename.lower():
        pulley = 'with_pulley'
    else:
        pulley = 'without_pulley'
    
    # 组合成标签
    if fault_type == 'healthy':
        label = f"healthy_{load}_{pulley}"
    else:
        label = f"{fault_type}_{severity}mm_{load}"
    
    return label

def load_and_process_data_from_nested_dirs(data_dir, window_size=1000, overlap=0.5):
    """
    从嵌套目录结构中加载并处理数据
    
    参数:
    - data_dir: 数据根目录
    - window_size: 窗口大小
    - overlap: 重叠比例
    
    返回:
    - 处理后的数据段和标签
    """
    # 创建标签映射
    label_map = create_label_mapping()
    
    X = []
    y = []
    
    # 递归查找所有CSV文件
    csv_files_with_info = recursive_find_csv_files(data_dir)
    
    for file_info in csv_files_with_info:
        if isinstance(file_info, tuple):
            file_path, folder_info = file_info
        else:
            file_path = file_info
            folder_info = None
        
        filename = os.path.basename(file_path)
        
        # 解析文件名获取标签
        label_str = parse_filename_with_folder_info(filename, folder_info)
        
        # 如果标签在映射中存在
        if label_str in label_map:
            label = label_map[label_str]
            
            try:
                # 加载数据
                data = pd.read_csv(file_path)
                
                # 重命名列（根据数据描述）
                if len(data.columns) >= 4:
                    data.columns = ['Time Stamp', 'X-axis', 'Y-axis', 'Z-axis'] + list(data.columns[4:])
                
                # 处理缺失值
                data = data.dropna()
                
                # 分割数据
                segments = segment_data(data, window_size, overlap)
                
                # 添加到数据集
                X.extend(segments)
                y.extend([label] * len(segments))
                
                print(f"处理文件: {filename}, 标签: {label_str}, 添加了 {len(segments)} 个样本")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    return X, y, label_map

def segment_data(data, window_size=1000, overlap=0.5):
    """
    将数据分割成固定大小的窗口
    
    参数:
    - data: 原始数据DataFrame
    - window_size: 窗口大小
    - overlap: 重叠比例(0-1之间)
    
    返回:
    - 分割后的数据段列表
    """
    segments = []
    step = int(window_size * (1 - overlap))
    
    # 提取特征列
    features = data[['X-axis', 'Y-axis', 'Z-axis']].values
    
    # 分割数据
    for i in range(0, len(features) - window_size + 1, step):
        segment = features[i:i + window_size]
        segments.append(segment)
    
    return np.array(segments)

def create_label_mapping():
    """
    创建故障类型和标签的映射
    
    返回:
    - 标签映射字典
    """
    # 故障类型
    fault_types = ['healthy', 'inner', 'outer']
    
    # 故障严重程度
    severities = ['0.0', '0.7', '0.9', '1.1', '1.3', '1.5', '1.7']  # 0.0用于健康状态
    
    # 负载条件
    loads = ['100W', '200W', '300W']
    
    # 创建映射
    label_map = {}
    label_count = 0
    
    # 健康状态 (无故障)
    for load in loads:
        for pulley in ['with_pulley', 'without_pulley']:
            key = f"healthy_{load}_{pulley}"
            label_map[key] = label_count
            label_count += 1
    
    # 故障状态
    for fault in ['inner', 'outer']:
        for severity in severities[1:]:  # 跳过0.0
            for load in loads:
                key = f"{fault}_{severity}mm_{load}"
                label_map[key] = label_count
                label_count += 1
    
    return label_map

# 示例使用
if __name__ == "__main__":
    data_dir = "path/to/your/bearing_data"
    X, y, label_map = load_and_process_data_from_nested_dirs(data_dir)
    print(f"加载了 {len(X)} 个样本，{len(np.unique(y))} 个不同类别")
import os
import requests
import time
from pathlib import Path

def test_single_folder(folder_path, api_url="http://localhost:8000"):
    """测试单个文件夹中的所有文件"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return []
    
    # 获取所有txt和csv文件
    files = list(folder.glob('*.txt')) + list(folder.glob('*.csv'))
    print(f"在 {folder_path} 中找到 {len(files)} 个文件")
    
    results = []
    
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 测试: {file_path.name}")
        
        try:
            # 发送请求
            with open(file_path, 'rb') as f:
                files_data = {'file': (file_path.name, f, 'text/plain')}
                response = requests.post(f"{api_url}/diagnose", files=files_data)
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'filename': file_path.name,
                    'status': result['status'],
                    'confidence': result['confidence_score'],
                    'distance': result['euclidean_distance'],
                    'data_points': result['data_points']
                })
                print(f"  结果: {result['status']} (置信度: {result['confidence_score']:.4f})")
            else:
                print(f"  失败: {response.status_code}")
                results.append({
                    'filename': file_path.name,
                    'status': 'ERROR',
                    'confidence': 0,
                    'distance': 0,
                    'data_points': 0
                })
                
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'filename': file_path.name,
                'status': 'ERROR',
                'confidence': 0,
                'distance': 0,
                'data_points': 0
            })
        
        time.sleep(0.1)  # 短暂延迟避免过于频繁的请求
    
    return results

def save_results_to_csv(results, dataset_name, output_file):
    """保存结果到CSV文件"""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['dataset', 'filename', 'status', 'confidence', 'distance', 'data_points']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            result['dataset'] = dataset_name
            writer.writerow(result)

def main():
    """主函数"""
    print("轴承数据集批量测试脚本")
    print("="*50)
    
    # 检查API是否可用
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("API服务正常")
        else:
            print("API服务异常")
            return
    except:
        print("无法连接到API服务，请确保服务已启动")
        return
    
    # 测试两个数据集
    datasets = [
        ('bearing_dataset', 'bearing_dataset'),
        ('bearing_dataset1', 'bearing_dataset1')
    ]
    
    all_results = []
    
    for dataset_name, folder_path in datasets:
        print(f"\n测试数据集: {dataset_name}")
        print("-" * 30)
        
        results = test_single_folder(folder_path)
        
        if results:
            # 保存单个数据集结果
            csv_filename = f"{dataset_name}_results.csv"
            save_results_to_csv(results, dataset_name, csv_filename)
            print(f"结果已保存到: {csv_filename}")
            
            # 添加到总结果
            for result in results:
                result['dataset'] = dataset_name
                all_results.append(result)
            
            # 打印统计
            healthy_count = len([r for r in results if r['status'] == '健康'])
            fault_count = len([r for r in results if r['status'] == '故障'])
            error_count = len([r for r in results if r['status'] == 'ERROR'])
            
            print(f"统计: 健康({healthy_count}) 故障({fault_count}) 错误({error_count})")
    
    # 保存总结果
    if all_results:
        save_results_to_csv(all_results, 'All', 'all_results.csv')
        print(f"\n总结果已保存到: all_results.csv")
        
        # 打印总体统计
        print(f"\n总体统计:")
        print(f"总文件数: {len(all_results)}")
        total_healthy = len([r for r in all_results if r['status'] == '健康'])
        total_fault = len([r for r in all_results if r['status'] == '故障'])
        total_error = len([r for r in all_results if r['status'] == 'ERROR'])
        print(f"健康: {total_healthy}, 故障: {total_fault}, 错误: {total_error}")

if __name__ == "__main__":
    main()
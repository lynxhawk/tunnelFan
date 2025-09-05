
# 无监督异常检测模型对比报告
生成时间: 2025-09-05 09:41:11

## 1. 实验概述
本次实验对比了多种无监督异常检测算法在轴承故障检测任务上的性能表现。

## 2. 参与对比的模型
- **Mahalanobis**: F1=0.9668, 误检率=20.03%
- **IsolationForest**: F1=0.8217, 误检率=7.05%
- **LOF**: F1=0.8543, 误检率=7.43%
- **PCA**: F1=0.8780, 误检率=20.03%

## 3. 性能排名

### 3.1 F1分数排名
1. Mahalanobis: 0.9668
2. PCA: 0.8780
3. LOF: 0.8543
4. IsolationForest: 0.8217

### 3.2 误检率排名（越低越好）
1. IsolationForest: 7.05%
2. LOF: 7.43%
3. Mahalanobis: 20.03%
4. PCA: 20.03%

### 3.3 检测速度排名
1. IsolationForest: 6.66秒
2. PCA: 10.23秒
3. Mahalanobis: 39.57秒
4. LOF: 89.20秒

## 4. 推荐模型

### 4.1 最佳综合性能
- 模型: Mahalanobis
- F1分数: 0.9668
- 文件: saved_models\best_f1_Mahalanobis_20250905_093740.pkl

### 4.2 最低误检率
- 模型: IsolationForest
- 误检率: 7.05%
- 文件: saved_models\best_low_fpr_IsolationForest_20250905_093740.pkl

### 4.3 最快检测速度
- 模型: IsolationForest
- 检测时间: 6.66秒
- 文件: saved_models\fastest_IsolationForest_20250905_093740.pkl

## 5. 使用建议

1. **生产环境部署**: 推荐使用误检率最低的模型，减少误报
2. **实时监控**: 推荐使用检测速度最快的模型
3. **综合考虑**: 推荐使用F1分数最高的模型

## 6. 模型文件说明

所有训练好的模型已保存在 `saved_models/` 目录下，可直接使用 `load_saved_model()` 函数加载。

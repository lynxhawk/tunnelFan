# 模型评估结果
train : val : test = 0.6 : 0.2 : 0.2
| 模型名称 | 准确率 |
|---------|--------|
| mlp   | 44.38%  |
| mlp_use_features   | 99.15%  |
| svm   | 44.08%  |
| svm_use_features   | 98.8%  |
| cnn   | 99.65%  |
| cnn_use_features   | 99.05%  |
| cnn_attention   | 94.06%  |
| cnn_bilstm   | 99.8%  |
| cnn_bilstm_attention   | 99.9%  |
| cnn_bigru_attention  | 99.9%  |
| cnn_bilstm_attention_enhanced  | 99.85%  |
| cnn_bilstm_attention_hybrid  | 99.8%  |
| cnn_bilstm_attention_multihead  | 99.85%  |
| informer  | 3.05%  |
| informer_use_features  | 99.1%  |
| light_cnn_informer  | 5.34%  |
| cnn_informer  | 99.35%  |
| cnn_informer_attention  | 99.85%  |
| se_informer_attention  | 3.25%  |
| se_cnn_informer_attention  | 99.65%  |
| patchtst  | 98.8%  |
| se_patchtst  | ?  |
| se_patchtst_attention  | 99.45%  |
| se_cnn_patchtst  | ?  |
| se_cnn_patchtst_attention  | ?  |
| se_resnet_patchtst  | ?  |
| se_resnet_patchtst_attention  | ?  |
| multi_feature_patchtst  | 98.75%  |
| multi_feature_patchtst use_fft | 98.6%  |
| multiscale_feature_patchtst  | 98.3%  |
| lstf_linear | 40.89%  |
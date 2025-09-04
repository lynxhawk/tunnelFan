from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="轴承故障诊断API",
    description="基于统计方法的轴承故障诊断服务",
    version="1.0.0"
)

# 数据模型定义
class DiagnosisResult(BaseModel):
    """诊断结果模型"""
    status: str  # "健康" 或 "故障"
    confidence_score: float  # 置信度分数
    euclidean_distance: float  # 欧氏距离
    threshold: float  # 判断阈值
    timestamp: str  # 诊断时间
    data_points: int  # 数据点数

class FileResult(BaseModel):
    """单个文件诊断结果"""
    filename: str
    success: bool
    result: Optional[DiagnosisResult] = None
    error_message: Optional[str] = None

class BatchDiagnosisResult(BaseModel):
    """批量诊断结果"""
    total_files: int
    successful_diagnoses: int
    failed_diagnoses: int
    results: List[FileResult]
    batch_timestamp: str

class StatisticalEuclideanDetector:
    """欧氏距离异常检测器（简化版）"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.mean = None
        self.threshold = None
        self.is_fitted = False
        
    def predict_single(self, test_data):
        """预测单个样本"""
        if not self.is_fitted:
            raise ValueError("模型未加载")
        
        # 如果是3D数据，展平为2D
        if test_data.ndim == 3:
            test_data = test_data.reshape(test_data.shape[0], -1)
        elif test_data.ndim == 1:
            test_data = test_data.reshape(1, -1)
        
        # 标准化
        test_scaled = self.scaler.transform(test_data)
        
        # 计算欧氏距离
        distances = np.array([np.linalg.norm(x - self.mean) for x in test_scaled])
        
        # 判断异常
        predictions = (distances > self.threshold).astype(int)
        
        return predictions, distances

class BearingFaultDiagnosisAPI:
    """轴承故障诊断API主类"""
    
    def __init__(self, model_path: str, seq_length: int = 1000):
        self.model_path = model_path
        self.seq_length = seq_length
        self.detector = None
        self.load_model()
    
    def load_model(self):
        """加载预训练的统计模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 重建检测器
            self.detector = StatisticalEuclideanDetector()
            self.detector.scaler = model_data['scaler']
            self.detector.mean = model_data['mean']
            self.detector.threshold = model_data['threshold']
            self.detector.is_fitted = model_data['is_fitted']
            
            logger.info(f"✅ 模型加载成功: {self.model_path}")
            logger.info(f"   阈值: {self.detector.threshold:.6f}")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def preprocess_data(self, raw_data: np.ndarray) -> np.ndarray:
        """预处理原始数据"""
        try:
            # 确保数据是1维的
            if raw_data.ndim > 1:
                raw_data = raw_data.flatten()
            
            # 创建滑动窗口
            if len(raw_data) < self.seq_length:
                # 如果数据长度不足，进行填充
                padding_length = self.seq_length - len(raw_data)
                raw_data = np.pad(raw_data, (0, padding_length), mode='constant', constant_values=0)
            
            # 创建滑动窗口
            windows = []
            step_size = self.seq_length // 4  # 25%重叠
            
            for i in range(0, len(raw_data) - self.seq_length + 1, step_size):
                window = raw_data[i:i + self.seq_length]
                windows.append(window)
            
            if len(windows) == 0:
                # 如果无法创建窗口，使用整个数据
                windows = [raw_data[:self.seq_length]]
            
            return np.array(windows)
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            raise
    
    def read_data_file(self, file_content: bytes, filename: str) -> np.ndarray:
        """读取数据文件（支持TXT、CSV格式）"""
        try:
            # 将字节转换为字符串
            content = file_content.decode('utf-8')
            file_ext = filename.lower().split('.')[-1]
            
            if file_ext == 'csv':
                return self._read_csv_data(content)
            elif file_ext == 'txt':
                return self._read_txt_data(content)
            else:
                # 尝试自动检测格式
                return self._auto_detect_format(content)
                
        except Exception as e:
            logger.error(f"读取数据文件失败: {e}")
            raise ValueError(f"数据文件格式错误: {e}")
    
    def _read_csv_data(self, content: str) -> np.ndarray:
        """读取CSV格式数据"""
        try:
            # 尝试不同的分隔符
            separators = [',', ';', '\t', ' ']
            
            for sep in separators:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep, header=None)
                    if len(df) > 0 and len(df.columns) > 0:
                        # 取第一列数据
                        data = df.iloc[:, 0].values.astype(float)
                        logger.info(f"CSV读取成功，分隔符: '{sep}', 数据点: {len(data)}")
                        return data
                except:
                    continue
            
            raise ValueError("无法解析CSV格式")
            
        except Exception as e:
            raise ValueError(f"CSV格式错误: {e}")
    
    def _read_txt_data(self, content: str) -> np.ndarray:
        """读取TXT格式数据"""
        try:
            # 方式1: 使用pandas读取
            try:
                data = pd.read_csv(io.StringIO(content), header=None, sep=None, engine='python')
                if len(data.columns) == 1:
                    return data.iloc[:, 0].values.astype(float)
                else:
                    # 如果多列，取第一列
                    return data.iloc[:, 0].values.astype(float)
                    
            except:
                # 方式2: 按行分割，每行一个数字
                lines = content.strip().split('\n')
                data = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):  # 忽略空行和注释行
                        try:
                            # 尝试分割多个数字（空格或制表符分隔）
                            numbers = line.replace('\t', ' ').split()
                            if numbers:
                                data.append(float(numbers[0]))  # 取第一个数字
                        except ValueError:
                            continue
                
                if not data:
                    raise ValueError("无法解析TXT数据")
                
                return np.array(data)
                
        except Exception as e:
            raise ValueError(f"TXT格式错误: {e}")
    
    def _auto_detect_format(self, content: str) -> np.ndarray:
        """自动检测数据格式"""
        # 先尝试CSV格式
        try:
            return self._read_csv_data(content)
        except:
            pass
        
        # 再尝试TXT格式
        try:
            return self._read_txt_data(content)
        except:
            pass
        
        raise ValueError("无法识别的数据格式")
    
    def diagnose(self, raw_data: np.ndarray) -> DiagnosisResult:
        """执行故障诊断"""
        try:
            # 预处理数据
            processed_data = self.preprocess_data(raw_data)
            logger.info(f"处理后数据形状: {processed_data.shape}")
            
            # 预测
            predictions, distances = self.detector.predict_single(processed_data)
            
            # 计算平均距离作为最终判断依据
            avg_distance = np.mean(distances)
            final_prediction = 1 if avg_distance > self.detector.threshold else 0
            
            # 计算置信度（基于距离与阈值的比值）
            confidence_score = min(abs(avg_distance - self.detector.threshold) / self.detector.threshold, 1.0)
            
            # 生成结果
            status = "故障" if final_prediction == 1 else "健康"
            
            result = DiagnosisResult(
                status=status,
                confidence_score=round(confidence_score, 4),
                euclidean_distance=round(avg_distance, 6),
                threshold=round(self.detector.threshold, 6),
                timestamp=datetime.now().isoformat(),
                data_points=len(raw_data)
            )
            
            logger.info(f"诊断结果: {status}, 距离: {avg_distance:.6f}, 阈值: {self.detector.threshold:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"诊断过程失败: {e}")
            raise
    
    def diagnose_multiple(self, files_data: List[tuple]) -> BatchDiagnosisResult:
        """批量诊断多个文件"""
        results = []
        successful_count = 0
        failed_count = 0
        
        for filename, raw_data in files_data:
            try:
                # 执行单个文件诊断
                result = self.diagnose(raw_data)
                
                file_result = FileResult(
                    filename=filename,
                    success=True,
                    result=result
                )
                
                successful_count += 1
                
            except Exception as e:
                logger.error(f"文件 {filename} 诊断失败: {e}")
                
                file_result = FileResult(
                    filename=filename,
                    success=False,
                    error_message=str(e)
                )
                
                failed_count += 1
            
            results.append(file_result)
        
        # 生成批量结果
        batch_result = BatchDiagnosisResult(
            total_files=len(files_data),
            successful_diagnoses=successful_count,
            failed_diagnoses=failed_count,
            results=results,
            batch_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"批量诊断完成: {successful_count} 成功, {failed_count} 失败")
        
        return batch_result

# 全局API实例（需要配置模型路径）
MODEL_PATH = "statistical_mahalanobis.pkl"  # 请修改为实际模型路径
api_instance = None

def initialize_api(model_path: str = MODEL_PATH):
    """初始化API实例"""
    global api_instance
    try:
        api_instance = BearingFaultDiagnosisAPI(model_path)
        logger.info("🚀 轴承故障诊断API初始化成功")
    except Exception as e:
        logger.error(f"❌ API初始化失败: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    try:
        initialize_api()
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        # 注意：这里不抛出异常，允许服务启动，但会在调用时报错

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "轴承故障诊断API",
        "version": "1.0.0",
        "status": "运行中" if api_instance else "未初始化",
        "endpoints": [
            "/diagnose - POST 单文件上传诊断",
            "/diagnose-batch - POST 批量文件上传诊断",
            "/diagnose-raw - POST 原始数据诊断",
            "/health - GET 健康检查",
            "/model-info - GET 模型信息"
        ]
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    return {
        "status": "healthy",
        "model_loaded": api_instance.detector is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """获取模型信息"""
    if api_instance is None:
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    return {
        "model_path": api_instance.model_path,
        "method": "statistical_euclidean",
        "sequence_length": api_instance.seq_length,
        "threshold": api_instance.detector.threshold if api_instance.detector else None,
        "is_fitted": api_instance.detector.is_fitted if api_instance.detector else False
    }

@app.post("/diagnose", response_model=DiagnosisResult)
async def diagnose_bearing(file: UploadFile = File(...)):
    """
    轴承故障诊断接口
    
    上传TXT/CSV格式的单轴轴承振动数据，返回健康/故障判断结果
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    # 检查文件类型
    allowed_extensions = ['.txt', '.csv']
    file_ext = '.' + file.filename.lower().split('.')[-1]
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"只支持以下格式文件: {', '.join(allowed_extensions)}")
    
    try:
        # 读取文件内容
        file_content = await file.read()
        logger.info(f"收到文件: {file.filename}, 大小: {len(file_content)} 字节")
        
        # 解析数据
        raw_data = api_instance.read_data_file(file_content, file.filename)
        logger.info(f"解析到 {len(raw_data)} 个数据点")
        
        # 执行诊断
        result = api_instance.diagnose(raw_data)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"诊断失败: {e}")
        raise HTTPException(status_code=500, detail=f"诊断过程出错: {str(e)}")

@app.post("/diagnose-batch", response_model=BatchDiagnosisResult)
async def diagnose_batch(files: List[UploadFile] = File(...)):
    """
    批量轴承故障诊断接口
    
    上传多个TXT/CSV格式的轴承振动数据文件，返回每个文件的诊断结果
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")
    
    if len(files) > 50:  # 限制最大文件数
        raise HTTPException(status_code=400, detail="一次最多上传50个文件")
    
    # 检查文件格式并读取数据
    files_data = []
    allowed_extensions = ['.txt', '.csv']
    
    for file in files:
        # 检查文件类型
        file_ext = '.' + file.filename.lower().split('.')[-1]
        
        if file_ext not in allowed_extensions:
            # 对于格式错误的文件，添加错误记录
            files_data.append((file.filename, None, f"不支持的文件格式: {file_ext}"))
            continue
        
        try:
            # 读取文件内容
            file_content = await file.read()
            logger.info(f"读取文件: {file.filename}, 大小: {len(file_content)} 字节")
            
            # 解析数据
            raw_data = api_instance.read_data_file(file_content, file.filename)
            files_data.append((file.filename, raw_data, None))
            
        except Exception as e:
            logger.error(f"文件 {file.filename} 读取失败: {e}")
            files_data.append((file.filename, None, f"文件读取错误: {str(e)}"))
    
    # 执行批量诊断
    try:
        # 准备诊断数据（过滤掉有错误的文件）
        valid_files_data = [(filename, data) for filename, data, error in files_data if error is None]
        error_files = [(filename, error) for filename, data, error in files_data if error is not None]
        
        # 执行诊断
        batch_result = api_instance.diagnose_multiple(valid_files_data)
        
        # 添加读取阶段的错误文件
        for filename, error_msg in error_files:
            error_file_result = FileResult(
                filename=filename,
                success=False,
                error_message=error_msg
            )
            batch_result.results.append(error_file_result)
            batch_result.total_files += 1
            batch_result.failed_diagnoses += 1
        
        logger.info(f"批量诊断完成: 总文件数 {batch_result.total_files}, "
                   f"成功 {batch_result.successful_diagnoses}, 失败 {batch_result.failed_diagnoses}")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"批量诊断失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量诊断过程出错: {str(e)}")

@app.post("/diagnose-raw")
async def diagnose_raw_data(data: List[float]):
    """
    直接接收数组数据进行诊断
    
    用于测试或直接传入数值数组
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    if not data:
        raise HTTPException(status_code=400, detail="数据不能为空")
    
    try:
        # 转换为numpy数组
        raw_data = np.array(data, dtype=float)
        logger.info(f"接收到 {len(raw_data)} 个数据点")
        
        # 执行诊断
        result = api_instance.diagnose(raw_data)
        
        return result
        
    except Exception as e:
        logger.error(f"诊断失败: {e}")
        raise HTTPException(status_code=500, detail=f"诊断过程出错: {str(e)}")

@app.post("/reload-model")
async def reload_model(model_path: str = None):
    """
    重新加载模型
    
    用于更换模型或重新初始化
    """
    global api_instance
    
    try:
        if model_path:
            initialize_api(model_path)
        else:
            initialize_api()
        
        return {
            "message": "模型重新加载成功",
            "model_path": api_instance.model_path if api_instance else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"模型重新加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # 可以通过环境变量设置模型路径
    model_path = os.getenv("MODEL_PATH", MODEL_PATH)
    
    # 启动前初始化
    try:
        initialize_api(model_path)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("⚠️  请确认模型路径正确")
    
    # 启动服务
    uvicorn.run(
        "main:app",  # 如果文件名是main.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
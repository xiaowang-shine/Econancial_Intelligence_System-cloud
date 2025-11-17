# error_handler.py - 统一错误处理模块
import logging
import traceback
from functools import wraps
from flask import jsonify, request
from typing import Dict, Any, Optional, Tuple
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BusinessError(Exception):
    """业务错误"""
    def __init__(self, message: str, error_code: str = None, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'BUSINESS_ERROR'
        self.status_code = status_code

class ValidationError(BusinessError):
    """验证错误"""
    def __init__(self, message: str, field: str = None):
        super().__init__(message, 'VALIDATION_ERROR', 400)
        self.field = field

class AuthenticationError(BusinessError):
    """认证错误"""
    def __init__(self, message: str = '认证失败'):
        super().__init__(message, 'AUTH_ERROR', 401)

class AuthorizationError(BusinessError):
    """授权错误"""
    def __init__(self, message: str = '权限不足'):
        super().__init__(message, 'AUTHZ_ERROR', 403)

class NotFoundError(BusinessError):
    """资源未找到错误"""
    def __init__(self, message: str = '资源未找到'):
        super().__init__(message, 'NOT_FOUND', 404)

class FileProcessingError(BusinessError):
    """文件处理错误"""
    def __init__(self, message: str, file_name: str = None):
        super().__init__(message, 'FILE_ERROR', 400)
        self.file_name = file_name

class DataProcessingError(BusinessError):
    """数据处理错误"""
    def __init__(self, message: str, step: str = None):
        super().__init__(message, 'DATA_ERROR', 500)
        self.step = step

def handle_error(func):
    """
    统一错误处理装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            return func(*args, **kwargs)
            
        except BusinessError as e:
            # 业务错误
            logger.warning(f"业务错误: {e.message}, 错误代码: {e.error_code}")
            return _create_error_response(
                message=e.message,
                error_code=e.error_code,
                status_code=e.status_code,
                details=getattr(e, 'field', None) or getattr(e, 'file_name', None) or getattr(e, 'step', None)
            )
            
        except ValueError as e:
            # 值错误
            logger.warning(f"值错误: {str(e)}")
            return _create_error_response(
                message=str(e),
                error_code='VALUE_ERROR',
                status_code=400
            )
            
        except KeyError as e:
            # 键错误
            logger.warning(f"键错误: {str(e)}")
            return _create_error_response(
                message=f"缺少必要参数: {str(e)}",
                error_code='KEY_ERROR',
                status_code=400
            )
            
        except FileNotFoundError as e:
            # 文件未找到
            logger.warning(f"文件未找到: {str(e)}")
            return _create_error_response(
                message="文件未找到或已被删除",
                error_code='FILE_NOT_FOUND',
                status_code=404
            )
            
        except PermissionError as e:
            # 权限错误
            logger.warning(f"权限错误: {str(e)}")
            return _create_error_response(
                message="权限不足",
                error_code='PERMISSION_ERROR',
                status_code=403
            )
            
        except Exception as e:
            # 系统错误
            execution_time = time.time() - start_time
            logger.error(f"系统错误: {str(e)}, 执行时间: {execution_time:.2f}秒")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            
            # 记录请求信息
            logger.error(f"请求路径: {request.path}")
            logger.error(f"请求方法: {request.method}")
            logger.error(f"请求参数: {dict(request.args)}")
            if request.is_json:
                logger.error(f"请求体: {request.get_json()}")
            
            return _create_error_response(
                message="系统内部错误，请稍后重试",
                error_code='INTERNAL_ERROR',
                status_code=500
            )
    
    return wrapper

def _create_error_response(message: str, error_code: str, status_code: int, 
                          details: Any = None) -> Tuple[Dict, int]:
    """
    创建错误响应
    
    Args:
        message: 错误消息
        error_code: 错误代码
        status_code: HTTP状态码
        details: 错误详情
        
    Returns:
        (响应字典, 状态码)
    """
    response = {
        'success': False,
        'error': {
            'message': message,
            'code': error_code,
            'timestamp': time.time()
        }
    }
    
    if details is not None:
        response['error']['details'] = details
    
    # 开发环境下添加调试信息
    if logger.level <= logging.DEBUG:
        response['error']['debug'] = {
            'request_path': request.path,
            'request_method': request.method
        }
    
    return response, status_code

def validate_request_data(required_fields: list = None, optional_fields: list = None):
    """
    请求数据验证装饰器
    
    Args:
        required_fields: 必需字段列表
        optional_fields: 可选字段列表
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 获取请求数据
                if request.is_json:
                    data = request.get_json() or {}
                else:
                    data = request.form.to_dict()
                
                # 验证必需字段
                if required_fields:
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        raise ValidationError(f"缺少必需字段: {', '.join(missing_fields)}")
                
                # 验证字段类型
                if optional_fields:
                    for field in optional_fields:
                        if field in data and data[field] is None:
                            raise ValidationError(f"字段 {field} 不能为空")
                
                # 将验证后的数据添加到kwargs
                kwargs['validated_data'] = data
                
                return func(*args, **kwargs)
                
            except ValidationError as e:
                raise e
            except Exception as e:
                raise ValidationError(f"请求数据验证失败: {str(e)}")
        
        return wrapper
    return decorator

def log_performance(func):
    """
    性能监控装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 记录性能信息
            logger.info(f"函数 {func.__name__} 执行完成, 耗时: {execution_time:.2f}秒")
            
            # 如果执行时间超过阈值，记录警告
            if execution_time > 10:  # 10秒阈值
                logger.warning(f"函数 {func.__name__} 执行时间过长: {execution_time:.2f}秒")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func.__name__} 执行失败, 耗时: {execution_time:.2f}秒, 错误: {str(e)}")
            raise
    
    return wrapper

class ErrorReporter:
    """错误报告器"""
    
    @staticmethod
    def report_error(error: Exception, context: Dict[str, Any] = None):
        """
        报告错误
        
        Args:
            error: 错误对象
            context: 错误上下文信息
        """
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'timestamp': time.time()
        }
        
        # 记录到日志
        logger.error(f"错误报告: {error_info}")
        
        # 这里可以添加其他错误报告方式，如发送邮件、通知等
    
    @staticmethod
    def report_performance(func_name: str, execution_time: float, 
                          context: Dict[str, Any] = None):
        """
        报告性能信息
        
        Args:
            func_name: 函数名
            execution_time: 执行时间
            context: 上下文信息
        """
        performance_info = {
            'function': func_name,
            'execution_time': execution_time,
            'context': context or {},
            'timestamp': time.time()
        }
        
        # 记录到日志
        logger.info(f"性能报告: {performance_info}")
        
        # 如果执行时间过长，记录警告
        if execution_time > 10:
            logger.warning(f"性能警告: {performance_info}")

# 全局错误处理器
def setup_global_error_handlers(app):
    """
    设置全局错误处理器
    
    Args:
        app: Flask应用实例
    """
    
    @app.errorhandler(404)
    def not_found_error(error):
        return _create_error_response(
            message="请求的资源未找到",
            error_code='NOT_FOUND',
            status_code=404
        )
    
    @app.errorhandler(405)
    def method_not_allowed_error(error):
        return _create_error_response(
            message="请求方法不被允许",
            error_code='METHOD_NOT_ALLOWED',
            status_code=405
        )
    
    @app.errorhandler(500)
    def internal_error(error):
        return _create_error_response(
            message="服务器内部错误",
            error_code='INTERNAL_ERROR',
            status_code=500
        )
    
    @app.errorhandler(Exception)
    def handle_unhandled_exception(error):
        logger.error(f"未处理的异常: {str(error)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        
        return _create_error_response(
            message="系统发生未知错误",
            error_code='UNKNOWN_ERROR',
            status_code=500
        )
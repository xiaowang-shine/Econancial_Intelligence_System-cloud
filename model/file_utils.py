# file_utils.py
import os
import uuid
import time
from datetime import datetime
from werkzeug.datastructures import FileStorage
from typing import Tuple, Optional

# 尝试导入magic库，如果失败则使用备用验证
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    print("警告: python-magic库未安装，将使用基础文件验证")

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 目录常量 - 使用绝对路径
UPLOAD_DIR = os.path.join(BASE_DIR, 'Uploads')
LOG_DIR = os.path.join(BASE_DIR, 'Logs')

# 文件安全配置
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_MIME_TYPES = {
    'application/vnd.ms-excel',  # .xls
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
    'application/octet-stream',  # 有时Excel文件会被识别为此类型
    'application/zip'  # .xlsx文件实际上是ZIP格式
}

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def validate_file_security(file_storage: FileStorage) -> Tuple[bool, str]:
    """
    验证文件安全性
    
    Args:
        file_storage: 上传的文件对象
        
    Returns:
        (是否有效, 错误信息)
    """
    # 检查文件是否存在
    if not file_storage or not file_storage.filename:
        return False, "未选择文件"
    
    filename = file_storage.filename
    
    # 检查文件名安全性
    if not _is_safe_filename(filename):
        return False, "文件名包含不安全字符"
    
    # 检查文件扩展名
    if not is_valid_excel_file(filename):
        return False, "只支持Excel文件(.xlsx, .xls)"
    
    # 检查文件大小
    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"文件大小超过限制({MAX_FILE_SIZE // (1024*1024)}MB)"
    
    if file_size == 0:
        return False, "文件为空"
    
    # 检查MIME类型（如果magic库可用）
    if HAS_MAGIC:
        try:
            file_content = file_storage.read(1024)  # 读取前1KB用于类型检测
            file_storage.seek(0)
            
            mime_type = magic.from_buffer(file_content, mime=True)
            if mime_type not in ALLOWED_MIME_TYPES:
                return False, f"不支持的文件类型: {mime_type}"
                
        except Exception as e:
            print(f"MIME类型检测失败: {e}")
            # 继续使用基础验证
    
    return True, ""

def _is_safe_filename(filename: str) -> bool:
    """
    检查文件名是否安全
    
    Args:
        filename: 文件名
        
    Returns:
        是否安全
    """
    # 检查危险字符
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        if char in filename:
            return False
    
    # 检查文件名长度
    if len(filename) > 255:
        return False
    
    # 检查是否为Windows保留名称
    reserved_names = [
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]
    
    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in reserved_names:
        return False
    
    return True

def save_uploaded_file(file_storage: FileStorage) -> str:
    """
    保存上传的文件到指定目录（增强安全版本）
    """
    # 首先进行安全验证
    is_valid, error_msg = validate_file_security(file_storage)
    if not is_valid:
        raise ValueError(error_msg)

    # 生成唯一文件名
    filename = file_storage.filename
    unique_filename = f"{int(time.time())}_{uuid.uuid4().hex}_{filename}"
    path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # 安全保存文件
    try:
        file_storage.save(path)
        # 验证文件是否成功保存
        if not os.path.exists(path):
            raise RuntimeError("文件保存失败")
        
        # 验证文件大小
        saved_size = os.path.getsize(path)
        if saved_size == 0:
            os.remove(path)
            raise RuntimeError("保存的文件为空")
            
        return path
    except Exception as e:
        # 清理可能创建的文件
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
        raise RuntimeError(f"文件保存失败: {str(e)}")

def log_message(task_id: str, message: str, log_dir: str = LOG_DIR) -> None:
    """
    记录任务日志
    """
    log_file = os.path.join(log_dir, f'task_{task_id}.log')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


# 在 file_utils.py 中修改 read_excel_file 函数

def read_excel_file(file_path, sheet_name=0, **kwargs):
    """
    读取Excel文件，支持多工作表

    Args:
        file_path: 文件路径
        sheet_name: 工作表名称或索引，None表示读取所有工作表
        **kwargs: 传递给pd.read_excel的其他参数

    Returns:
        如果sheet_name为None，返回字典{sheet_name: DataFrame}；
        否则返回单个DataFrame
    """
    try:
        import pandas as pd

        if sheet_name is None:
            # 读取所有工作表
            return pd.read_excel(file_path, sheet_name=None, **kwargs)
        else:
            # 读取指定工作表
            return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return None

def get_file_extension(filename: str) -> str:
    """
    获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        文件扩展名（小写）
    """
    return os.path.splitext(filename)[1].lower()


# 在 file_utils.py 中确保 is_valid_excel_file 函数可用
def is_valid_excel_file(filename: str) -> bool:
    """
    检查是否为有效的Excel文件

    Args:
        filename: 文件名

    Returns:
        是否为有效的Excel文件
    """
    ext = get_file_extension(filename)
    return ext in ['.xlsx', '.xls']


def generate_unique_filename(original_filename: str) -> str:
    """
    生成唯一的文件名

    Args:
        original_filename: 原始文件名

    Returns:
        唯一的文件名
    """
    ext = get_file_extension(original_filename)
    base_name = os.path.splitext(original_filename)[0]
    return f"{base_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"


def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """
    清理指定目录中的旧文件

    Args:
        directory: 目录路径
        max_age_hours: 最大保留时间（小时）

    Returns:
        删除的文件数量
    """
    count = 0
    current_time = time.time()

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # 检查文件修改时间
            file_mtime = os.path.getmtime(file_path)
            if current_time - file_mtime > max_age_hours * 3600:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")

    return count
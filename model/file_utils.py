# file_utils.py
import os
import uuid
import time
from datetime import datetime
from werkzeug.datastructures import FileStorage
from typing import Tuple, Optional

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 目录常量 - 使用绝对路径
UPLOAD_DIR = os.path.join(BASE_DIR, 'Uploads')
LOG_DIR = os.path.join(BASE_DIR, 'Logs')

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def save_uploaded_file(file_storage: FileStorage) -> str:
    """
    保存上传的文件到指定目录
    """
    # 检查文件类型
    filename = file_storage.filename
    if not (filename and (filename.endswith('.xlsx') or filename.endswith('.xls'))):
        raise ValueError("只支持Excel文件(.xlsx, .xls)")

    # 生成唯一文件名
    unique_filename = f"{int(time.time())}_{uuid.uuid4().hex}_{filename}"
    path = os.path.join(UPLOAD_DIR, unique_filename)
    file_storage.save(path)
    return path

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
# data_preprocessor.py
import pandas as pd
import numpy as np


def detect_column_types(df):
    """
    检测DataFrame中的列类型

    Args:
        df: 输入DataFrame

    Returns:
        列类型字典
    """
    result = {
        'time_col': None,
        'financial_cols': [],
        'health_score_col': None
    }

    # 尝试检测时间列
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['date', '时间', '日期', 'month', 'year']):
            result['time_col'] = col
            break

    # 尝试检测财务列
    financial_keywords = [
        '资产', '负债', '收入', '利润', '现金', '存货',
        'asset', 'liability', 'revenue', 'profit', 'cash', 'inventory'
    ]

    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in financial_keywords):
            result['financial_cols'].append(col)

    # 尝试检测健康度评分列
    health_keywords = ['健康度', 'score', 'health', 'rating']
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in health_keywords):
            result['health_score_col'] = col
            break

    return result


def validate_data(df, time_col, required_cols):
    """
    验证数据是否满足基本要求

    Args:
        df: 输入DataFrame
        time_col: 时间列名
        required_cols: 需要的列名列表

    Returns:
        验证结果和错误消息
    """
    errors = []

    # 检查时间列
    if time_col not in df.columns:
        errors.append(f"时间列 '{time_col}' 不存在")

    # 检查必需列
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"必需列 '{col}' 不存在")

    # 检查数据量
    if len(df) < 3:
        errors.append("数据量过少，至少需要3个月的数据")

    # 检查时间列格式
    try:
        pd.to_datetime(df[time_col])
    except:
        errors.append("时间列格式不正确，无法转换为日期")

    return len(errors) == 0, errors
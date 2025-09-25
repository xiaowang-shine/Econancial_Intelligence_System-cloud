# model_training.py
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Any, Tuple, Optional
import traceback

# 导入机器学习相关库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import shap

# 尝试导入贝叶斯模型
try:
    from model.bayesian_model import run_bayesian_forecast
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False
    print("贝叶斯模型不可用")

# 尝试导入 Prophet（可选）
try:
    from prophet import Prophet
    import cmdstanpy
    HAS_PROPHET = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Prophet 不可用: {e}")
    HAS_PROPHET = False

# 常量
FORECAST_MONTHS = 12
MIN_TRAINING_ROWS = 5
DEFAULT_CONFIDENCE_LEVEL = 1.96  # 95% 置信区间


def infer_time_col(df: pd.DataFrame) -> str:
    """推断时间列"""
    for c in df.columns:
        if any(k in str(c).lower() for k in ['date', '时间', '日期']):
            return c
    return df.columns[0]


def infer_target_col(df: pd.DataFrame) -> str:
    """推断目标列"""
    preferred = ['revenue', '营业收入', '收入', 'amount', 'value']
    for c in df.columns:
        if any(pk in str(c).lower() for pk in preferred):
            return c
    nums = df.select_dtypes(include=[np.number]).columns
    return nums[0] if len(nums) else df.columns[-1]


def prepare_training_data(df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    """准备训练数据"""
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    if df[time_col].isna().all():
        raise ValueError("时间列无法解析")

    train_df = df[[time_col, target_col]].rename(
        columns={time_col: 'date', target_col: 'target'}
    ).dropna()

    train_df = train_df.sort_values('date').reset_index(drop=True)

    if len(train_df) < MIN_TRAINING_ROWS:
        raise ValueError(f"训练数据过少（<{MIN_TRAINING_ROWS} 行）")

    return train_df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """创建特征"""
    df_feat = df.copy()
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['year'] = df_feat['date'].dt.year

    # 添加滞后特征
    for lag in [1, 2, 3, 6, 12]:
        if lag < len(df_feat):
            df_feat[f'lag_{lag}'] = df_feat['target'].shift(lag)

    # 添加滚动平均特征
    df_feat['rolling_3'] = df_feat['target'].rolling(min(3, len(df_feat))).mean().shift(1)
    df_feat['rolling_6'] = df_feat['target'].rolling(min(6, len(df_feat))).mean().shift(1)

    # 添加傅里叶项（季节性）
    t = np.arange(len(df_feat)) / max(1, len(df_feat))
    for i in range(1, 5):
        df_feat[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t / 12)
        df_feat[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t / 12)

    # 删除缺失值并重置索引
    df_feat = df_feat.dropna().reset_index(drop=True)

    # 分离特征和目标
    X = df_feat.drop(columns=['date', 'target'])
    y = df_feat['target']

    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Optional[Any]:
    """训练随机森林模型"""
    if len(X) < 5:
        return None

    try:
        n_splits = max(2, min(3, len(X) // 3))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        models, scores = [], []

        for tr_idx, val_idx in tscv.split(X):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(mean_absolute_error(y.iloc[val_idx], pred))
            models.append(model)

        best_model = models[int(np.argmin(scores))]
        return best_model
    except Exception as e:
        print(f"随机森林模型训练失败: {e}")
        return None


def train_linear_model(X: pd.DataFrame, y: pd.Series) -> Any:
    """训练线性回归模型"""
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # 将scaler存储在模型中以便后续使用
    model.scaler = scaler
    return model


def random_forest_forecast(model: Any, X: pd.DataFrame, train_df: pd.DataFrame,
                           hist_mae: float) -> List[Dict[str, Any]]:
    """使用随机森林进行预测"""
    preds = []
    last_date = train_df['date'].max()
    base_window = train_df['target'].tolist()

    # 逐步预测
    for m in range(1, FORECAST_MONTHS + 1):
        next_date = last_date + relativedelta(months=m)
        window = base_window + [p['value'] for p in preds]

        # 构建特征
        feat = {
            'month': next_date.month,
            'year': next_date.year,
            'rolling_3': np.mean(window[-3:]) if len(window) >= 3 else np.mean(base_window),
            'rolling_6': np.mean(window[-6:]) if len(window) >= 6 else np.mean(base_window)
        }

        # 添加滞后特征
        for lag in [1, 2, 3, 6, 12]:
            feat[f'lag_{lag}'] = window[-lag] if len(window) >= lag else np.mean(base_window)

        # 添加傅里叶项
        t = (len(train_df) + m) / max(1, len(train_df) + FORECAST_MONTHS)
        for i in range(1, 5):
            feat[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t / 12)
            feat[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t / 12)

        # 预测
        Xpred = pd.DataFrame([feat])[X.columns].fillna(0)
        pred_val = float(model.predict(Xpred)[0])

        # 计算置信区间
        lower = max(0.0, pred_val - DEFAULT_CONFIDENCE_LEVEL * hist_mae)
        upper = pred_val + DEFAULT_CONFIDENCE_LEVEL * hist_mae

        preds.append({
            'date': next_date.strftime('%Y-%m-%d'),
            'value': pred_val,
            'lower': lower,
            'upper': upper
        })

    return preds


def linear_model_forecast(model: Any, train_df: pd.DataFrame,
                          hist_mae: float) -> List[Dict[str, Any]]:
    """使用线性模型进行预测"""
    preds = []
    last_date = train_df['date'].max()

    # 提取特征名
    feature_names = ['month', 'year']
    for lag in [1, 2, 3, 6, 12]:
        feature_names.extend([f'lag_{lag}', f'rolling_{lag}'])
    for i in range(1, 5):
        feature_names.extend([f'fourier_sin_{i}', f'fourier_cos_{i}'])

    for m in range(1, FORECAST_MONTHS + 1):
        next_date = last_date + relativedelta(months=m)

        # 构建特征
        feat = {
            'month': next_date.month,
            'year': next_date.year,
            'rolling_3': train_df['target'].rolling(3).mean().iloc[-1] if len(train_df) >= 3 else train_df[
                'target'].mean(),
            'rolling_6': train_df['target'].rolling(6).mean().iloc[-1] if len(train_df) >= 6 else train_df[
                'target'].mean()
        }

        # 添加滞后特征
        for lag in [1, 2, 3, 6, 12]:
            feat[f'lag_{lag}'] = train_df['target'].iloc[-lag] if len(train_df) >= lag else train_df['target'].mean()

        # 添加傅里叶项
        t = (len(train_df) + m) / max(1, len(train_df) + FORECAST_MONTHS)
        for i in range(1, 5):
            feat[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t / 12)
            feat[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t / 12)

        # 转换为DataFrame并标准化
        Xpred = pd.DataFrame([feat])
        Xpred = Xpred.reindex(columns=feature_names, fill_value=0)

        # 标准化特征
        if hasattr(model, 'scaler'):
            Xpred_scaled = model.scaler.transform(Xpred)
        else:
            Xpred_scaled = Xpred.values

        # 预测
        pred_val = float(model.predict(Xpred_scaled)[0])

        # 计算置信区间
        lower = max(0.0, pred_val - DEFAULT_CONFIDENCE_LEVEL * hist_mae)
        upper = pred_val + DEFAULT_CONFIDENCE_LEVEL * hist_mae

        preds.append({
            'date': next_date.strftime('%Y-%m-%d'),
            'value': pred_val,
            'lower': lower,
            'upper': upper
        })

    return preds


def calculate_feature_importance(model: Any, X: pd.DataFrame) -> List[Dict[str, Any]]:
    """计算特征重要性"""
    importances = []

    if hasattr(model, 'feature_importances_'):
        # 随机森林的特征重要性
        arr = model.feature_importances_
        importances = [{'name': n, 'importance': float(v)} for n, v in zip(X.columns, arr)]
        importances = sorted(importances, key=lambda x: -x['importance'])[:10]
    elif hasattr(model, 'coef_'):
        # 线性模型的系数
        arr = np.abs(model.coef_)
        if len(arr) == len(X.columns):
            importances = [{'name': n, 'importance': float(v)} for n, v in zip(X.columns, arr)]
            importances = sorted(importances, key=lambda x: -x['importance'])[:10]

    # 如果没有特征重要性，使用默认值
    if not importances:
        importances = [
            {'name': 'month', 'importance': 0.4},
            {'name': 'year', 'importance': 0.3},
            {'name': 'lag_1', 'importance': 0.2},
            {'name': 'rolling_3', 'importance': 0.1}
        ]

    return importances


def run_training_task(monthly_df: pd.DataFrame, health_df: pd.DataFrame,
                      mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行训练任务 - 作为核心算法的辅助

    Args:
        monthly_df: 月度数据DataFrame
        health_df: 健康度数据DataFrame
        mapping: 列映射字典

    Returns:
        训练结果字典
    """
    try:
        # 推断时间列和目标列
        time_col = mapping.get('fileMonthly', {}).get('timeCol') or infer_time_col(monthly_df)
        target_col = mapping.get('fileMonthly', {}).get('targetCol') or infer_target_col(monthly_df)

        # 准备训练数据
        train_df = prepare_training_data(monthly_df, time_col, target_col)

        # 创建特征
        X, y = create_features(train_df)

        # 计算历史MAE
        hist_mae = float(np.mean(np.abs(np.diff(train_df['target'])))) if len(train_df) > 2 else float(
            train_df['target'].std() or 1.0)

        # 选择并训练模型
        model = None
        model_type = 'LinearRegression'

        # 首先尝试随机森林
        rf_model = train_random_forest(X, y)
        if rf_model is not None:
            model = rf_model
            model_type = 'RandomForest'
        else:
            # 回退到线性模型
            linear_model = train_linear_model(X, y)
            model = linear_model
            model_type = 'LinearRegression'

        # 进行预测
        if model_type == 'RandomForest':
            preds = random_forest_forecast(model, X, train_df, hist_mae)
        else:
            preds = linear_model_forecast(model, train_df, hist_mae)

        # 计算特征重要性
        importances = calculate_feature_importance(model, X)

        # 计算波动度
        vals = [p['value'] for p in preds]
        fluct = round((max(vals) - min(vals)) / (np.mean(vals) or 1) * 100, 2) if vals else 0.0

        # 生成建议
        suggestions = [{
            'priority': '中',
            'text': '使用辅助模型进行预测，建议增加数据量以提高准确性',
            'impact': '提升预测精度'
        }]

        # 返回结果
        return {
            'forecast': preds,
            'kpi': {
                'debt_ratio': None,
                'current_ratio': None,
                'quick_ratio': None,
                'health_score': None
            },
            'explain': {'feature_importance': importances},
            'suggestions': suggestions,
            'meta': {
                'model': model_type,
                'model_version': 'v1.0',
                'data_rows': len(train_df),
                'fluctuation': fluct,
                'is_fallback': True  # 标记为备用模型
            }
        }

    except Exception as e:
        error_msg = f"训练任务执行失败: {str(e)}\n{traceback.format_exc()}"
        raise Exception(error_msg)


def simple_forecast(train_df, time_col, target_col, periods=12):
    """
    简单预测函数（用于数据量极少的情况）

    Args:
        train_df: 训练数据
        time_col: 时间列名
        target_col: 目标列名
        periods: 预测期数

    Returns:
        预测结果列表
    """
    # 确保时间列是datetime类型
    train_df[time_col] = pd.to_datetime(train_df[time_col])
    train_df = train_df.sort_values(time_col)

    # 提取时间和目标值
    dates = train_df[time_col].values
    values = train_df[target_col].values

    # 计算趋势（简单线性回归）
    x = np.arange(len(dates)).reshape(-1, 1)
    y = values

    model = LinearRegression()
    model.fit(x, y)

    # 预测未来值
    future_x = np.arange(len(dates), len(dates) + periods).reshape(-1, 1)
    future_y = model.predict(future_x)

    # 计算历史波动率
    hist_mae = mean_absolute_error(y[1:], y[:-1]) if len(y) > 1 else np.std(y) if len(y) > 0 else 1.0

    # 生成预测结果
    preds = []
    last_date = dates[-1]

    for i in range(periods):
        next_date = last_date + relativedelta(months=i + 1)
        pred_val = float(future_y[i])
        lower = max(0.0, pred_val - DEFAULT_CONFIDENCE_LEVEL * hist_mae)
        upper = pred_val + DEFAULT_CONFIDENCE_LEVEL * hist_mae

        preds.append({
            'date': pd.Timestamp(next_date).strftime('%Y-%m-%d'),
            'value': pred_val,
            'lower': lower,
            'upper': upper
        })

    return preds
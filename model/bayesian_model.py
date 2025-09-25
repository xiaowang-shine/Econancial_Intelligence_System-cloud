# bayesian_model.py
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def run_bayesian_forecast(df, date_col="date", target_col="value", periods=12):
    """
    用 PyMC 进行简单的贝叶斯线性回归预测
    """
    # 保持原有实现不变
    df = df.copy()
    df = df.dropna(subset=[date_col, target_col])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 用时间索引作为自变量 (归一化以避免数值问题)
    X = np.arange(len(df))
    X = (X - X.mean()) / X.std()
    y = df[target_col].values

    with pm.Model() as model:
        # 先验分布
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha + beta * X

        # 似然函数
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # 采样
        idata = pm.sample(1000, tune=1000, target_accept=0.9, cores=2, random_seed=42)

    # 未来预测点
    future_idx = np.arange(len(df), len(df) + periods)
    future_X = (future_idx - X.mean()) / X.std()

    with model:
        posterior_predictive = pm.sample_posterior_predictive(idata, var_names=["alpha", "beta", "sigma"])

    # 计算未来预测分布
    alpha_samples = posterior_predictive["alpha"]
    beta_samples = posterior_predictive["beta"]
    sigma_samples = posterior_predictive["sigma"]

    preds = []
    for i, fx in enumerate(future_X):
        draws = alpha_samples + beta_samples * fx
        median = np.median(draws)
        lower = np.percentile(draws, 2.5)
        upper = np.percentile(draws, 97.5)
        preds.append({
            "date": df[date_col].max() + pd.DateOffset(months=i+1),
            "value": float(median),
            "lower": float(lower),
            "upper": float(upper)
        })

    return preds

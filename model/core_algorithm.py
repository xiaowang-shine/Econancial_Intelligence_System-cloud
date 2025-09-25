# core_algorithm.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
import itertools
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pymc as pm
import arviz as az
from model.model_training import HAS_BAYESIAN as TRAINING_BAYESIAN  # 重命名避免冲突
from sklearn.linear_model import LinearRegression

# 尝试导入 Prophet（可选）
try:
    from prophet import Prophet
    import cmdstanpy

    HAS_PROPHET = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Prophet 不可用: {e}")
    HAS_PROPHET = False

# 尝试导入贝叶斯模型
try:
    import pymc as pm
    import arviz as az

    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False
    print("贝叶斯模型不可用")


# 设置中文字体和日志级别
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')


# 在 CoreAlgorithm 类中添加缺失的方法
class CoreAlgorithm:
    def __init__(self, desktop_path=None, seed=None, add_noise=True, noise_scale=0.01):
        """初始化核心算法"""
        self.desktop_path = desktop_path or os.getcwd()
        os.makedirs(self.desktop_path, exist_ok=True)

        # 随机数生成器
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.add_noise = add_noise
        self.noise_scale = noise_scale

        # 权重设置
        self.weights = {
            'w1': 0.4108,  # 流动比率权重
            'w2': 0.4108,  # 速动比率权重
            'w3': 0.1558,  # (1-资产负债率)权重
            'w4': 0.1558,  # 净利润率权重
            'w5': 0.0668  # 资产周转率权重
        }

        # 参数设置
        self.MAX_RATIO = 5.0  # 流动比率和速动比率的最大合理值
        self.MIN_CL = 1000.0  # 流动负债的最小合理值

        # 数据存储
        self.monthly_data = None
        self.health_scores = None

    # 添加缺失的方法
    def generate_monthly_data(self):
        """
        生成月度财务数据（示例方法）
        """
        # 这里实现生成示例数据的逻辑
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
        data = {
            'date': dates,
            'revenue': np.random.normal(100000, 20000, len(dates)),
            'expenses': np.random.normal(80000, 15000, len(dates)),
            'profit': np.random.normal(20000, 5000, len(dates))
        }
        return pd.DataFrame(data)


    def prepare_data(self, df, time_col, target_cols, min_months=12):
        """
        准备数据，根据需要决定是否进行插值

        Args:
            df: 原始数据DataFrame
            time_col: 时间列名
            target_cols: 目标列名列表
            min_months: 最小需要的数据月数

        Returns:
            处理后的数据DataFrame
        """
        # 确保时间列是datetime类型
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        # 检查数据量
        if len(df) >= min_months:
            # 数据量足够，直接使用
            return df
        else:
            # 数据量不足，进行线性插值
            print(f"数据量不足 ({len(df)}个月)，进行线性插值至{min_months}个月")
            return self._linear_interpolate(df, time_col, target_cols, min_months)

    def _linear_interpolate(self, df, time_col, target_cols, target_months):
        """
        线性插值函数

        Args:
            df: 原始数据DataFrame
            time_col: 时间列名
            target_cols: 目标列名列表
            target_months: 目标月数

        Returns:
            插值后的数据DataFrame
        """
        # 创建完整的日期范围
        start_date = df[time_col].min()
        end_date = start_date + pd.DateOffset(months=target_months - 1)
        full_dates = pd.date_range(start=start_date, end=end_date, freq='M')

        # 准备结果DataFrame
        result_df = pd.DataFrame({time_col: full_dates})

        # 对每个目标列进行线性插值
        for col in target_cols:
            if col in df.columns:
                # 获取原始数据点
                original_dates = df[time_col]
                original_values = df[col]

                # 计算每个原始数据点在整个时间范围内的位置
                date_positions = [(d - start_date).days for d in original_dates]
                target_positions = [(d - start_date).days for d in full_dates]

                # 线性插值
                interpolated_values = np.interp(target_positions, date_positions, original_values)

                # 添加轻微扰动
                if self.add_noise:
                    perturbation = self.rng.uniform(1 - self.noise_scale, 1 + self.noise_scale,
                                                    len(interpolated_values))
                    interpolated_values = interpolated_values * perturbation

                # 确保原始值保持不变
                for i, orig_date in enumerate(original_dates):
                    if orig_date in full_dates:
                        idx = list(full_dates).index(orig_date)
                        interpolated_values[idx] = original_values.iloc[i]

                # 添加到结果DataFrame
                result_df[col] = interpolated_values

        return result_df

    def add_fourier_terms(self, df, period=12, order=4):
        """
        添加傅里叶项（季节性）

        Args:
            df: 输入DataFrame
            period: 周期长度
            order: 傅里叶阶数

        Returns:
            添加了傅里叶项的DataFrame
        """
        t = np.arange(len(df)) / max(1, len(df))
        for i in range(1, order + 1):
            df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t / period)
            df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t / period)
        return df

    def bayesian_forecast(self, df, date_col="date", target_col="value", periods=12):
        """
        贝叶斯预测

        Args:
            df: 包含时间序列数据的 DataFrame
            date_col: 时间列名
            target_col: 目标变量列名
            periods: 预测未来的周期数

        Returns:
            预测结果列表
        """
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
                "date": df[date_col].max() + pd.DateOffset(months=i + 1),
                "value": float(median),
                "lower": float(lower),
                "upper": float(upper)
            })

        return preds

    def forecast_financials(self, df=None, time_col=None, target_cols=None,
                            forecast_months=12, use_bayesian=False):
        """
        预测财务指标 - 修改为可以处理多种调用方式
        """
        if df is None:
            # 如果没有提供数据，使用示例数据
            return self.generate_monthly_data()

        # 使用提供的数据进行预测
        return self.forecast_financials_with_data(df, time_col, target_cols,
                                                  forecast_months, use_bayesian)

    def forecast_financials_with_data(self, df, time_col, target_cols, forecast_months=12, use_bayesian=False):
        """
        使用稳健的预测方法 - 完全重写版本
        """
        try:
            print("=== 开始财务预测 ===")
            print(f"输入数据形状: {df.shape}")
            print(f"时间列: {time_col}")
            print(f"目标列: {target_cols}")

            # 确保数据正确排序
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)

            print(f"数据时间范围: {df[time_col].min()} 到 {df[time_col].max()}")

            # 创建预测日期
            last_date = df[time_col].max()
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_months,
                freq='M'
            )

            # 使用稳健的趋势预测方法
            forecast_results = {}

            for column in target_cols:
                if column in df.columns:
                    print(f"\n--- 预测列: {column} ---")
                    series = df[column].dropna()

                    if len(series) < 3:
                        print(f"数据不足，使用最后值: {series.iloc[-1]}")
                        # 数据不足，使用最后值加上轻微增长
                        last_val = series.iloc[-1]
                        forecast_values = [last_val * (1 + 0.01 * i) for i in range(forecast_months)]
                    else:
                        # 使用稳健的线性趋势预测
                        x = np.arange(len(series))
                        y = series.values

                        # 计算稳健的趋势（使用最近12个月的数据）
                        recent_data = series.tail(min(12, len(series)))
                        if len(recent_data) > 1:
                            # 计算月度增长率
                            growth_rates = []
                            for i in range(1, len(recent_data)):
                                if recent_data.iloc[i - 1] != 0:
                                    growth_rate = (recent_data.iloc[i] - recent_data.iloc[i - 1]) / recent_data.iloc[
                                        i - 1]
                                    growth_rates.append(growth_rate)

                            if growth_rates:
                                # 使用平均增长率，但限制在合理范围内
                                avg_growth = np.mean(growth_rates)
                                # 限制增长率在 -10% 到 +10% 之间
                                avg_growth = max(-0.1, min(0.1, avg_growth))

                                print(f"检测到增长率: {avg_growth:.3%}")

                                # 基于趋势预测
                                last_val = series.iloc[-1]
                                forecast_values = []
                                for i in range(forecast_months):
                                    # 逐渐衰减增长效应，避免过度外推
                                    decay_factor = max(0.5, 1 - i * 0.1)  # 逐渐衰减
                                    monthly_growth = avg_growth * decay_factor
                                    next_val = last_val * (1 + monthly_growth)
                                    forecast_values.append(next_val)
                                    last_val = next_val
                            else:
                                # 没有明显趋势，使用最后值
                                last_val = series.iloc[-1]
                                forecast_values = [last_val] * forecast_months
                        else:
                            # 数据不足，使用最后值
                            last_val = series.iloc[-1]
                            forecast_values = [last_val] * forecast_months

                    print(f"预测值范围: {min(forecast_values):.0f} 到 {max(forecast_values):.0f}")
                    forecast_results[column] = forecast_values

            # 创建预测数据框
            forecast_df = pd.DataFrame(forecast_results, index=forecast_dates)
            forecast_df.index.name = time_col

            # 历史数据
            historical_df = df.set_index(time_col)

            print("=== 预测完成 ===")
            print(f"预测结果形状: {forecast_df.shape}")

            return historical_df, forecast_df

        except Exception as e:
            print(f"预测失败: {e}")
            import traceback
            traceback.print_exc()

            # 紧急回退：使用历史趋势进行简单预测
            return self.emergency_fallback(df, time_col, target_cols, forecast_months)

    def emergency_fallback(self, df, time_col, target_cols, forecast_months):
        """紧急回退预测方法"""
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

        last_date = df[time_col].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_months,
            freq='M'
        )

        forecast_results = {}

        for column in target_cols:
            if column in df.columns:
                series = df[column].dropna()
                if len(series) >= 2:
                    # 计算年度增长率
                    first_year_avg = series.head(12).mean() if len(series) >= 12 else series.iloc[0]
                    last_year_avg = series.tail(12).mean() if len(series) >= 12 else series.iloc[-1]

                    if first_year_avg > 0:
                        annual_growth = (last_year_avg - first_year_avg) / first_year_avg
                        monthly_growth = annual_growth / 12
                    else:
                        monthly_growth = 0.01  # 默认1%月度增长

                    # 限制增长率
                    monthly_growth = max(-0.05, min(0.05, monthly_growth))

                    last_val = series.iloc[-1]
                    forecast_values = [last_val * (1 + monthly_growth * (i + 1)) for i in range(forecast_months)]
                else:
                    # 数据不足，使用最后值
                    last_val = series.iloc[-1] if len(series) > 0 else 1000
                    forecast_values = [last_val] * forecast_months

                forecast_results[column] = forecast_values

        forecast_df = pd.DataFrame(forecast_results, index=forecast_dates)
        forecast_df.index.name = time_col

        return df.set_index(time_col), forecast_df

    def forecast_health_score(self, health_data, time_col, score_col, forecast_months=12, use_bayesian=False):
        """
        预测健康度评分

        Args:
            health_data: 健康度数据DataFrame
            time_col: 时间列名
            score_col: 评分列名
            forecast_months: 预测月数
            use_bayesian: 是否使用贝叶斯方法

        Returns:
            预测结果
        """
        # 准备数据
        health_data[time_col] = pd.to_datetime(health_data[time_col])
        health_data = health_data.sort_values(time_col)

        if use_bayesian:
            # 使用贝叶斯方法预测健康度
            return self.bayesian_forecast(
                health_data.rename(columns={time_col: 'date', score_col: 'value'}),
                periods=forecast_months
            )
        else:
            # 使用Prophet预测健康度
            prophet_df = health_data[[time_col, score_col]].rename(
                columns={time_col: 'ds', score_col: 'y'}
            )

            # 定义中国节假日
            def create_holidays_df():
                # 获取数据中的年份范围
                years = pd.DatetimeIndex(prophet_df['ds']).year.unique()
                spring_festival_dates = []
                national_day_dates = []

                for year in years:
                    # 春节（近似计算）
                    spring_festival = pd.Timestamp(f'{year}-02-01') + pd.DateOffset(years=year - 2022)
                    spring_festival_dates.append(spring_festival)

                    # 国庆节
                    national_day = pd.Timestamp(f'{year}-10-01')
                    national_day_dates.append(national_day)

                spring_festival_df = pd.DataFrame({
                    'holiday': 'spring_festival',
                    'ds': spring_festival_dates,
                    'lower_window': -7,
                    'upper_window': 7
                })

                national_day_df = pd.DataFrame({
                    'holiday': 'national_day',
                    'ds': national_day_dates,
                    'lower_window': -2,
                    'upper_window': 7
                })

                return pd.concat([spring_festival_df, national_day_df])

            # 参数调优
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                'seasonality_prior_scale': [0.1, 1.0, 10.0],
                'holidays_prior_scale': [0.1, 1.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative']
            }

            # 生成所有参数组合
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            maes = []  # 存储平均绝对误差

            # 使用交叉验证评估所有参数组合
            for params in all_params:
                try:
                    model = Prophet(**params, yearly_seasonality=True, weekly_seasonality=False,
                                    daily_seasonality=False)
                    model.holidays = create_holidays_df()
                    model.fit(prophet_df)

                    # 交叉验证
                    df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='90 days')
                    df_p = performance_metrics(df_cv, rolling_window=1)
                    maes.append(df_p['mae'].values[0])
                except:
                    maes.append(1000)  # 如果失败，使用很大的误差值

            # 找到最佳参数（MAE最小）
            best_params = all_params[np.argmin(maes)]

            # 使用最佳参数训练最终模型
            final_model = Prophet(**best_params, yearly_seasonality=True, weekly_seasonality=False,
                                  daily_seasonality=False)
            final_model.holidays = create_holidays_df()
            final_model.fit(prophet_df)

            # 创建未来数据框
            future = final_model.make_future_dataframe(periods=forecast_months, freq='M')

            # 进行预测
            forecast = final_model.predict(future)

            # 提取未来预测值
            future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()][
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            # 转换为统一格式
            preds = []
            for _, row in future_forecast.iterrows():
                preds.append({
                    'date': row['ds'],
                    'value': row['yhat'],
                    'lower': row['yhat_lower'],
                    'upper': row['yhat_upper']
                })

            return preds

    def calculate_health_score(self, financial_data):
        """
        计算健康度得分

        Args:
            financial_data: 包含财务指标的字典

        Returns:
            健康度得分和各项比率
        """
        # 确保流动负债不低于最小值
        cl = max(financial_data['CL'], self.MIN_CL)

        # 计算各项比率
        CR = min(financial_data['CA'] / cl, self.MAX_RATIO)  # 流动比率，设置上限
        QR = min((financial_data['CA'] - financial_data['Inv']) / cl, self.MAX_RATIO)  # 速动比率，设置上限
        DR = financial_data['TD'] / financial_data['TA'] if financial_data['TA'] > 0.0001 else 0  # 资产负债率
        NPM = financial_data['NI'] / financial_data['Rev'] if financial_data['Rev'] > 0.0001 else 0  # 净利润率
        AT = financial_data['Rev'] / financial_data['TA'] if financial_data['TA'] > 0.0001 else 0  # 资产周转率

        # 计算健康度得分
        H = (self.weights['w1'] * CR +
             self.weights['w2'] * QR +
             self.weights['w3'] * (1 - DR) +
             self.weights['w4'] * NPM +
             self.weights['w5'] * AT)

        return H, CR, QR, DR, NPM, AT

    def optimize_funds(self, X, financial_data):
        """
        优化资金分配

        Args:
            X: 可用资金
            financial_data: 财务数据

        Returns:
            优化后的财务数据和分配方案
        """
        # 计算长期负债
        LTD = max(financial_data['TD'] - financial_data['CL'], 0.0)

        # 分配资金
        if X <= financial_data['CL'] - self.MIN_CL:  # 确保保留最小流动负债
            y3 = X  # 用于偿还流动负债
            y4 = 0  # 不偿还长期负债
            excess = 0  # 无剩余资金
        elif X <= financial_data['CL'] - self.MIN_CL + LTD:
            y3 = financial_data['CL'] - self.MIN_CL  # 偿还部分流动负债，保留MIN_CL
            y4 = X - y3  # 剩余用于偿还长期负债
            excess = 0  # 无剩余资金
        else:
            y3 = financial_data['CL'] - self.MIN_CL  # 偿还部分流动负债，保留MIN_CL
            y4 = LTD  # 全部长期负债
            excess = X - (y3 + y4)  # 剩余资金增加流动资产

        # 更新财务数据
        new_data = financial_data.copy()
        new_data['CL'] = financial_data['CL'] - y3
        new_data['TD'] = financial_data['TD'] - y3 - y4
        new_data['TA'] = financial_data['TA'] - y3 - y4 + excess
        new_data['CA'] = financial_data['CA'] - y3 - y4 + excess

        return new_data, y3, y4, excess

    def run_optimization(self, financial_data, year="2024"):
        """
        运行资金优化分析

        Args:
            financial_data: 财务数据字典
            year: 年份标识

        Returns:
            优化结果和最优方案
        """
        # 计算当前健康度
        current_H, current_CR, current_QR, current_DR, current_NPM, current_AT = self.calculate_health_score(
            financial_data)

        # 测试不同资金规模的效果
        X_values = [5000, 10000, 15000, 20000, 25000, 30000]
        results = []

        for X in X_values:
            # 优化资金分配
            new_data, y3, y4, excess = self.optimize_funds(X, financial_data)

            # 计算新健康度
            new_H, new_CR, new_QR, new_DR, new_NPM, new_AT = self.calculate_health_score(new_data)

            # 存储结果
            results.append({
                'X': X,
                'y3': y3,
                'y4': y4,
                'excess': excess,
                'H': new_H,
                'CR': new_CR,
                'QR': new_QR,
                'DR': new_DR,
                'NPM': new_NPM,
                'AT': new_AT
            })

        # 转换为DataFrame
        df_results = pd.DataFrame(results)

        # 找到最优资金分配
        max_H_idx = df_results['H'].idxmax()
        optimal_result = df_results.loc[max_H_idx]

        return df_results, optimal_result, {
            'current_H': current_H,
            'current_CR': current_CR,
            'current_QR': current_QR,
            'current_DR': current_DR,
            'current_NPM': current_NPM,
            'current_AT': current_AT
        }

    # 在 core_algorithm.py 中添加以下方法
    def fallback_to_training_model(self, df, time_col, target_cols, forecast_months=12):
        """
        回退到训练模型作为备用方案

        Args:
            df: 输入数据DataFrame
            time_col: 时间列名
            target_cols: 目标列名列表
            forecast_months: 预测月数

        Returns:
            预测结果DataFrame
        """
        try:
            from model.model_training import run_training_task

            # 对每个目标列分别预测
            forecast_results = {}
            for column in target_cols:
                if column in df.columns:
                    # 准备单列数据
                    single_col_df = df[[time_col, column]].copy()

                    # 运行训练任务
                    result = run_training_task(
                        single_col_df,
                        None,  # 不需要健康度数据
                        {'fileMonthly': {'timeCol': time_col, 'targetCol': column}}
                    )

                    # 提取预测值
                    forecast_values = [p['value'] for p in result['forecast']]
                    forecast_results[column] = forecast_values

            # 创建预测日期索引
            last_date = pd.to_datetime(df[time_col]).max()
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_months,
                freq='M'
            )

            # 创建预测数据框
            forecast_df = pd.DataFrame(forecast_results, index=forecast_dates)

            return df.set_index(time_col), forecast_df

        except Exception as e:
            print(f"回退到训练模型失败: {e}")
            # 如果连备用模型都失败，使用最简单的预测
            return self.simple_fallback(df, time_col, target_cols, forecast_months)

    def simple_fallback(self, df, time_col, target_cols, forecast_months=12):
        """
        最简单的回退方案

        Args:
            df: 输入数据DataFrame
            time_col: 时间列名
            target_cols: 目标列名列表
            forecast_months: 预测月数

        Returns:
            预测结果DataFrame
        """
        from model.model_training import simple_forecast

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

        forecast_results = {}
        for column in target_cols:
            if column in df.columns:
                preds = simple_forecast(df, time_col, column, forecast_months)
                forecast_values = [p['value'] for p in preds]
                forecast_results[column] = forecast_values

        # 创建预测日期索引
        last_date = df[time_col].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_months,
            freq='M'
        )

        # 创建预测数据框
        forecast_df = pd.DataFrame(forecast_results, index=forecast_dates)

        return df.set_index(time_col), forecast_df
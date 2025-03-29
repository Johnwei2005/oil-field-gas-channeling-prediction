#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统不确定性量化模块

本模块负责量化预测结果的不确定性，为决策提供风险评估。
"""

import os
import numpy as np
import pandas as pd
import logging
from scipy import stats
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# 导入配置
from config import UNCERTAINTY_CONFIG, PATHS, VIZ_CONFIG
from physics_ml import PhysicsGuidedXGBoost, BayesianPhysicsModel, GaussianProcessPhysicsModel

# 设置日志
logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """
    预测不确定性量化器
    
    使用多种方法量化模型预测的不确定性，包括蒙特卡洛模拟、自助法和模型内置不确定性估计。
    """
    
    def __init__(self, 
                 model=None, 
                 monte_carlo_samples=None, 
                 bootstrap_samples=None,
                 confidence_level=None,
                 prediction_intervals=None):
        """
        初始化不确定性量化器
        
        Args:
            model: 预测模型
            monte_carlo_samples: 蒙特卡洛采样次数
            bootstrap_samples: 自助法样本数
            confidence_level: 置信水平
            prediction_intervals: 是否计算预测区间
        """
        self.model = model
        self.monte_carlo_samples = monte_carlo_samples if monte_carlo_samples else UNCERTAINTY_CONFIG.get('monte_carlo_samples', 1000)
        self.bootstrap_samples = bootstrap_samples if bootstrap_samples else UNCERTAINTY_CONFIG.get('bootstrap_samples', 100)
        self.confidence_level = confidence_level if confidence_level else UNCERTAINTY_CONFIG.get('confidence_level', 0.95)
        self.prediction_intervals = prediction_intervals if prediction_intervals is not None else UNCERTAINTY_CONFIG.get('prediction_intervals', True)
        
        # 方法映射
        self.methods = {
            'model_based': self._model_based_uncertainty,
            'bootstrap': self._bootstrap_uncertainty,
            'monte_carlo': self._monte_carlo_uncertainty
        }
    
    def _model_based_uncertainty(self, X):
        """
        使用模型内置的不确定性估计方法
        
        Args:
            X: 特征数据
            
        Returns:
            tuple: (预测均值, 预测标准差)
        """
        logger.info("使用模型内置方法估计不确定性")
        
        # 检查模型是否支持不确定性估计
        if isinstance(self.model, (BayesianPhysicsModel, GaussianProcessPhysicsModel)):
            # 这些模型支持直接返回预测标准差
            return self.model.predict(X, return_std=True)
        elif hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba')):
            # 使用predict_proba方法估计不确定性
            try:
                proba = self.model.predict_proba(X)
                mean = np.sum(proba * np.arange(proba.shape[1]), axis=1)
                std = np.sqrt(np.sum(proba * (np.arange(proba.shape[1]) - mean.reshape(-1, 1))**2, axis=1))
                return mean, std
            except:
                logger.warning("模型不支持预测不确定性，使用蒙特卡洛方法代替")
                return self._monte_carlo_uncertainty(X)
        else:
            # 模型不支持内置不确定性估计，使用其他方法
            logger.warning("模型不支持预测不确定性，使用蒙特卡洛方法代替")
            return self._monte_carlo_uncertainty(X)
    
    def _bootstrap_uncertainty(self, X, y=None, X_train=None, y_train=None):
        """
        使用自助法估计不确定性
        
        通过对训练数据进行重采样，训练多个模型，估计预测不确定性。
        
        Args:
            X: 预测特征
            y: 真实标签（可选）
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            tuple: (预测均值, 预测标准差)
        """
        logger.info(f"使用自助法估计不确定性，样本数: {self.bootstrap_samples}")
        
        if X_train is None or y_train is None:
            logger.warning("未提供训练数据，无法使用自助法估计不确定性")
            return self._monte_carlo_uncertainty(X)
        
        # 存储各个样本的预测结果
        predictions = []
        
        # 对训练数据进行重采样并训练多个模型
        for i in range(self.bootstrap_samples):
            # 重采样
            X_boot, y_boot = resample(X_train, y_train)
            
            # 训练模型
            model_instance = self.model.__class__()
            model_instance.fit(X_boot, y_boot)
            
            # 预测
            pred = model_instance.predict(X)
            predictions.append(pred)
        
        # 将预测结果转换为数组
        predictions = np.array(predictions)
        
        # 计算预测均值和标准差
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    # 在 uncertainty.py 中，改进 _monte_carlo_uncertainty 方法（大约第144行）
    def _monte_carlo_uncertainty(self, X):
        """
        增强的蒙特卡洛模拟估计不确定性
        
        使用自适应噪声水平和多种采样策略。
        
        参数:
            X: 预测特征
            
        返回:
            tuple: (预测均值, 预测标准差)
        """
        logger.info(f"使用蒙特卡洛模拟估计不确定性，样本数: {self.monte_carlo_samples}")
        
        # 存储各个样本的预测结果
        predictions = []
        
        # 根据特征的数据类型和分布确定噪声尺度
        if isinstance(X, pd.DataFrame):
            feature_types = {col: X[col].dtype for col in X.columns}
            scales = {}
            for col in X.columns:
                if np.issubdtype(feature_types[col], np.number):
                    # 使用百分比噪声，最小为0.01
                    scale = max(0.03 * X[col].std(), 0.01)
                    scales[col] = scale
        
        # 对X进行多次随机扰动并预测
        for i in range(self.monte_carlo_samples):
            # 添加分层随机噪声
            X_noise = X.copy()
            
            if isinstance(X, pd.DataFrame):
                for col in X.select_dtypes(include=[np.number]).columns:
                    # 为不同的特征使用不同的噪声尺度
                    noise_scale = scales.get(col, 0.05)
                    
                    # 根据特征的取值范围调整噪声
                    col_min, col_max = X[col].min(), X[col].max()
                    col_range = col_max - col_min
                    
                    if col_range > 0:
                        # 为较大的值添加较大的噪声，为较小的值添加较小的噪声
                        relative_values = (X[col] - col_min) / col_range
                        adaptive_noise = np.random.normal(
                            0, 
                            noise_scale * (0.5 + 0.5 * relative_values), 
                            size=X[col].shape
                        )
                        X_noise[col] = X[col] + adaptive_noise * X[col].std()
                    else:
                        # 对于常量特征，使用小的固定噪声
                        X_noise[col] = X[col] + np.random.normal(0, 0.01, size=X[col].shape)
            else:
                # 对于非DataFrame的X，使用基于列方差的噪声
                col_stds = np.std(X, axis=0)
                # 对于零方差的列，设置一个小的正数以避免乘以零
                col_stds = np.where(col_stds > 0, col_stds, 0.01)
                
                noise = np.random.normal(0, 0.05, size=X.shape)
                # 缩放噪声与每列的标准差成比例
                scaled_noise = noise * col_stds
                X_noise = X + scaled_noise
            
            # 使用模型预测
            try:
                pred = self.model.predict(X_noise)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"蒙特卡洛预测出错 (样本 {i}): {str(e)}")
                # 跳过这个样本
                continue
        
        # 如果所有样本都失败，回退到单一预测
        if len(predictions) == 0:
            logger.warning("所有蒙特卡洛样本都失败，回退到单一预测")
            mean_pred = self.model.predict(X)
            # 使用一个小的固定值作为标准差
            std_pred = np.ones_like(mean_pred) * 0.01
            return mean_pred, std_pred
        
        # 将预测结果转换为数组
        predictions = np.array(predictions)
        
        # 计算预测均值和标准差
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 应用物理约束：标准差不应过大也不应为零
        # 计算合理的标准差上限（基于均值的一定比例）
        std_upper_limit = np.maximum(mean_pred * 0.3, 0.05)
        # 标准差应有一个小的下限，避免过于确定的预测
        std_lower_limit = np.maximum(mean_pred * 0.01, 0.001)
        
        # 限制标准差在合理范围内
        std_pred = np.minimum(std_pred, std_upper_limit)
        std_pred = np.maximum(std_pred, std_lower_limit)
        
        # 确保均值为非负
        mean_pred = np.maximum(mean_pred, 0)
        
        return mean_pred, std_pred
    
    # 在 uncertainty.py 中，更新 quantify_uncertainty 方法
    def quantify_uncertainty(self, X, method='model_based', y=None, X_train=None, y_train=None):
        """
        增强的不确定性量化方法
        
        参数:
            X: 预测特征
            method: 不确定性估计方法，可选 'model_based', 'bootstrap', 'monte_carlo'
            y: 真实标签（可选）
            X_train: 训练特征（适用于bootstrap方法）
            y_train: 训练标签（适用于bootstrap方法）
            
        返回:
            dict: 包含均值、标准差、预测区间等的结果字典
        """
        if self.model is None:
            raise ValueError("未指定模型")
        
        # 使用指定方法估计不确定性
        if method not in self.methods:
            logger.warning(f"不支持的不确定性估计方法: {method}，使用monte_carlo方法代替")
            method = 'monte_carlo'
        
        # 调用相应的方法
        if method == 'bootstrap':
            mean_pred, std_pred = self.methods[method](X, y, X_train, y_train)
        else:
            mean_pred, std_pred = self.methods[method](X)
        
        # 计算预测区间 - 调整z值以实现更准确的覆盖率
        alpha = 1 - self.confidence_level
        # 使用t分布而不是正态分布，适用于小样本
        if y is not None and len(y) < 30:
            from scipy import stats
            # 自由度为样本大小减1
            df = len(y) - 1
            t_score = stats.t.ppf(1 - alpha / 2, df)
            z_score = t_score
        else:
            # 对于大样本或未知样本大小，使用正态分布
            from scipy import stats
            z_score = stats.norm.ppf(1 - alpha / 2)
        
        # 额外的校准因子，以提高覆盖率
        calibration_factor = 1.2  # 略微扩大预测区间
        
        lower_bound = mean_pred - z_score * std_pred * calibration_factor
        upper_bound = mean_pred + z_score * std_pred * calibration_factor
        
        # 确保下界非负（对于PV数值）
        lower_bound = np.maximum(lower_bound, 0)
        
        # 生成结果字典
        results = {
            'mean': mean_pred,
            'std': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'z_score': z_score,
            'confidence_level': self.confidence_level,
            'calibration_factor': calibration_factor  # 添加校准因子
        }
        
        # 如果有实际值，计算覆盖率
        if y is not None:
            within_interval = ((y >= lower_bound) & (y <= upper_bound))
            coverage_rate = np.mean(within_interval)
            results['actual_coverage_rate'] = coverage_rate
            logger.info(f"实际覆盖率: {coverage_rate:.2%}, 目标置信水平: {self.confidence_level:.2%}")
            
            # 如果覆盖率太低，记录警告
            if coverage_rate < self.confidence_level * 0.8:
                logger.warning(f"覆盖率({coverage_rate:.2%})显著低于目标置信水平({self.confidence_level:.2%})，考虑增加校准因子")
        
        return results
    
    def analyze_sensitivities(self, X, feature_names=None, n_points=20):
        """
        分析特征敏感性
        
        通过改变每个特征的值，分析其对预测结果的影响。
        
        Args:
            X: 基准特征数据（单个样本）
            feature_names: 特征名称列表
            n_points: 每个特征的采样点数
            
        Returns:
            dict: 特征敏感性分析结果
        """
        logger.info("开始特征敏感性分析")
        
        if self.model is None:
            raise ValueError("未指定模型")
        
        # 确保X是DataFrame且只有一行
        if isinstance(X, pd.DataFrame):
            if len(X) > 1:
                logger.warning(f"提供了多个样本({len(X)}行)，只使用第一个样本进行敏感性分析")
                X = X.iloc[[0]]
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            if len(X.shape) > 1 and X.shape[0] > 1:
                logger.warning(f"提供了多个样本({X.shape[0]}行)，只使用第一个样本进行敏感性分析")
                X = X[[0], :]
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 将X转换为DataFrame便于处理
        # Convert to DataFrame with feature names
        if not isinstance(X, pd.DataFrame):
            if feature_names is not None and len(feature_names) != X.shape[1]:
                logger.warning(f"特征名数量({len(feature_names)})与特征维度({X.shape[1]})不匹配，使用默认特征名")
                feature_names = None
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            X = pd.DataFrame(X, columns=feature_names)
        
        # 特征敏感性结果
        sensitivities = {}
        
        # 分析每个特征的敏感性
        for feature in feature_names:
            # 获取特征的范围
            if X[feature].dtype in [np.int64, np.int32, np.int16, np.int8]:
                # 整型特征
                min_val = X[feature].iloc[0] * 0.5 if X[feature].iloc[0] > 0 else X[feature].iloc[0] * 1.5
                max_val = X[feature].iloc[0] * 1.5 if X[feature].iloc[0] > 0 else X[feature].iloc[0] * 0.5
                feature_range = np.linspace(min_val, max_val, n_points).astype(int)
            else:
                # 浮点型特征
                min_val = X[feature].iloc[0] * 0.5 if X[feature].iloc[0] > 0 else X[feature].iloc[0] * 1.5
                max_val = X[feature].iloc[0] * 1.5 if X[feature].iloc[0] > 0 else X[feature].iloc[0] * 0.5
                feature_range = np.linspace(min_val, max_val, n_points)
            
            # 对每个特征值进行预测
            predictions = []
            for val in feature_range:
                X_copy = X.copy()
                X_copy[feature] = val
                
                # 使用模型的特征名创建预测数据
                if hasattr(self.model, 'feature_names') and self.model.feature_names:
                    # 确保使用相同的特征名格式进行预测
                    # 创建一个与模型特征匹配的DataFrame
                    pred_data = pd.DataFrame(columns=self.model.feature_names)
                    # 复制所有共有特征的值
                    for feat in set(X_copy.columns) & set(self.model.feature_names):
                        pred_data[feat] = X_copy[feat]
                    # 为缺失特征设置默认值0
                    for feat in set(self.model.feature_names) - set(X_copy.columns):
                        pred_data[feat] = 0
                    
                    pred = self.model.predict(pred_data)
                else:
                    pred = self.model.predict(X_copy)
                
                predictions.append(pred[0])
            
            # 计算敏感性指标
            predictions = np.array(predictions)
            baseline = self.model.predict(X)[0]
            sensitivity = (predictions.max() - predictions.min()) / baseline if baseline != 0 else 0
            
            # 保存结果
            sensitivities[feature] = {
                'feature_range': feature_range,
                'predictions': predictions,
                'sensitivity': sensitivity,
                'baseline': baseline
            }
        
        # 对特征按敏感性排序
        sorted_features = sorted(
            sensitivities.keys(), 
            key=lambda x: sensitivities[x]['sensitivity'],
            reverse=True
        )
        
        # 返回结果
        results = {
            'sensitivities': sensitivities,
            'sorted_features': sorted_features
        }
        
        logger.info(f"特征敏感性分析完成，排名前三的特征: {', '.join(sorted_features[:3])}")
        
        return results
    
    def calculate_uncertainty_metrics(self, mean_pred, std_pred, threshold):
        """
        计算基于不确定性的风险指标
        
        Args:
            mean_pred: 预测均值
            std_pred: 预测标准差
            threshold: 风险阈值
            
        Returns:
            dict: 风险指标
        """
        # 计算超过阈值的概率
        prob_exceeds = 1 - stats.norm.cdf(threshold, loc=mean_pred, scale=std_pred)
        
        # 计算风险指数（越高风险越大）
        risk_index = prob_exceeds * (mean_pred / threshold)
        
        # 计算可靠性指数（越高越可靠）
        reliability_index = (threshold - mean_pred) / std_pred
        
        # 风险等级分类
        risk_levels = np.zeros_like(mean_pred, dtype=np.int32)
        risk_levels[prob_exceeds > 0.75] = 3  # 高风险
        risk_levels[(prob_exceeds > 0.25) & (prob_exceeds <= 0.75)] = 2  # 中风险
        risk_levels[prob_exceeds <= 0.25] = 1  # 低风险
        
        return {
            'probability_exceeds_threshold': prob_exceeds,
            'risk_index': risk_index,
            'reliability_index': reliability_index,
            'risk_levels': risk_levels
        }
    
    def create_risk_map(self, X, y=None, uncertainty_method='model_based', risk_threshold=None):
        """
        创建风险图
        
        Args:
            X: 特征数据
            y: 真实标签（可选）
            uncertainty_method: 不确定性估计方法
            risk_threshold: 风险阈值
            
        Returns:
            dict: 风险图数据
        """
        # 量化不确定性
        uncertainty_results = self.quantify_uncertainty(X, method=uncertainty_method, y=y)
        
        # 获取预测均值和标准差
        mean_pred = uncertainty_results['mean']
        std_pred = uncertainty_results['std']
        
        # 如果未指定风险阈值，则使用预测均值的中位数
        if risk_threshold is None:
            risk_threshold = np.median(mean_pred)
        
        # 计算风险指标
        risk_metrics = self.calculate_uncertainty_metrics(mean_pred, std_pred, risk_threshold)
        
        # 结合原始特征和风险指标
        if isinstance(X, pd.DataFrame):
            risk_data = X.copy()
        else:
            if hasattr(self.model, 'feature_names') and self.model.feature_names:
                risk_data = pd.DataFrame(X, columns=self.model.feature_names)
            else:
                risk_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # 添加预测和风险指标
        risk_data['predicted_value'] = mean_pred
        risk_data['prediction_std'] = std_pred
        risk_data['lower_bound'] = uncertainty_results['lower_bound']
        risk_data['upper_bound'] = uncertainty_results['upper_bound']
        risk_data['prob_exceeds_threshold'] = risk_metrics['probability_exceeds_threshold']
        risk_data['risk_index'] = risk_metrics['risk_index']
        risk_data['reliability_index'] = risk_metrics['reliability_index']
        risk_data['risk_level'] = risk_metrics['risk_levels']
        
        # 如果有真实标签，添加到数据中
        if y is not None:
            risk_data['actual_value'] = y
            risk_data['prediction_error'] = y - mean_pred
            risk_data['within_interval'] = ((y >= uncertainty_results['lower_bound']) & 
                                            (y <= uncertainty_results['upper_bound']))
            
            # 计算预测区间覆盖率
            coverage_rate = risk_data['within_interval'].mean()
            logger.info(f"预测区间覆盖率: {coverage_rate:.2%}, 目标置信水平: {self.confidence_level:.2%}")
        
        return {
            'risk_data': risk_data,
            'threshold': risk_threshold,
            'coverage_rate': coverage_rate if y is not None else None
        }
    
    def plot_uncertainty(self, X, y=None, uncertainty_method='model_based', 
                       save_path=None, show_plot=True):
        """
        绘制不确定性可视化图
        
        Args:
            X: 特征数据
            y: 真实标签（可选）
            uncertainty_method: 不确定性估计方法
            save_path: 图像保存路径
            show_plot: 是否显示图像
            
        Returns:
            matplotlib.figure.Figure: 图像对象
        """
        # 量化不确定性
        uncertainty_results = self.quantify_uncertainty(X, method=uncertainty_method, y=y)
        
        # 获取预测均值和置信区间
        mean_pred = uncertainty_results['mean']
        lower_bound = uncertainty_results['lower_bound']
        upper_bound = uncertainty_results['upper_bound']
        
        # 创建图像
        fig, ax = plt.subplots(figsize=VIZ_CONFIG['figure_size'])
        
        # 绘制预测均值
        ax.plot(range(len(mean_pred)), mean_pred, 'b-', label='预测均值')
        
        # 绘制置信区间
        ax.fill_between(range(len(mean_pred)), lower_bound, upper_bound, 
                       color='blue', alpha=0.2, label=f'{self.confidence_level:.0%}置信区间')
        
        # 如果有真实标签，则绘制
        if y is not None:
            ax.plot(range(len(y)), y, 'ro', label='实际值')
            
            # 计算在置信区间内的点数
            within_interval = ((y >= lower_bound) & (y <= upper_bound))
            coverage_rate = within_interval.mean()
            
            # 绘制区间内外的点
            ax.plot(np.where(within_interval)[0], y[within_interval], 'go', label='区间内点')
            ax.plot(np.where(~within_interval)[0], y[~within_interval], 'ro', label='区间外点')
            
            ax.set_title(f'预测不确定性 (区间覆盖率: {coverage_rate:.2%})')
        else:
            ax.set_title('预测不确定性')
        
        ax.set_xlabel('样本索引')
        ax.set_ylabel('预测值')
        ax.legend(loc='best')
        ax.grid(True)
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"不确定性可视化图已保存至 {save_path}")
        
        # 显示图像
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_sensitivity(self, sensitivity_results, top_n=5, save_path=None, show_plot=True):
        """
        绘制特征敏感性图
        
        Args:
            sensitivity_results: 特征敏感性分析结果
            top_n: 显示前N个最敏感特征
            save_path: 图像保存路径
            show_plot: 是否显示图像
            
        Returns:
            matplotlib.figure.Figure: 图像对象
        """
        # 获取排序后的特征
        sorted_features = sensitivity_results['sorted_features'][:top_n]
        sensitivities = sensitivity_results['sensitivities']
        
        # 创建图像
        fig, axes = plt.subplots(len(sorted_features), 1, figsize=(VIZ_CONFIG['figure_size'][0], 
                                                                 VIZ_CONFIG['figure_size'][1] * len(sorted_features) / 2))
        
        # 确保axes是列表
        if len(sorted_features) == 1:
            axes = [axes]
        
        # 绘制每个特征的敏感性曲线
        for i, feature in enumerate(sorted_features):
            feature_data = sensitivities[feature]
            feature_range = feature_data['feature_range']
            predictions = feature_data['predictions']
            baseline = feature_data['baseline']
            sensitivity = feature_data['sensitivity']
            
            ax = axes[i]
            ax.plot(feature_range, predictions, 'b-')
            ax.axhline(y=baseline, color='r', linestyle='--', label='基准预测')
            
            ax.set_title(f'{feature} (敏感度: {sensitivity:.4f})')
            ax.set_xlabel('特征值')
            ax.set_ylabel('预测值')
            ax.legend(loc='best')
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"特征敏感性图已保存至 {save_path}")
        
        # 显示图像
        if show_plot:
            plt.show()
        
        return fig
    
    def save_results(self, risk_data, filepath=None):
        """
        保存风险评估结果
        
        Args:
            risk_data: 风险评估数据
            filepath: 保存路径
        """
        if filepath is None:
            filepath = os.path.join(PATHS['results_dir'], 'risk_assessment_results.csv')
        
        risk_data.to_csv(filepath, index=False)
        logger.info(f"风险评估结果已保存至 {filepath}")


def monte_carlo_simulation(model, X, n_samples=1000, noise_scale=0.05):
    """
    使用蒙特卡洛模拟估计预测不确定性
    
    Args:
        model: 预测模型
        X: 预测特征
        n_samples: 蒙特卡洛采样次数
        noise_scale: 噪声尺度
        
    Returns:
        tuple: (预测均值, 预测标准差)
    """
    # 创建不确定性量化器
    uq = UncertaintyQuantifier(model, monte_carlo_samples=n_samples)
    
    # 使用蒙特卡洛方法量化不确定性
    results = uq.quantify_uncertainty(X, method='monte_carlo')
    
    return results['mean'], results['std']


def calculate_prediction_interval(mean, std, confidence_level=0.95):
    """
    计算预测区间
    
    Args:
        mean: 预测均值
        std: 预测标准差
        confidence_level: 置信水平
        
    Returns:
        tuple: (下界, 上界)
    """
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    lower_bound = mean - z_score * std
    upper_bound = mean + z_score * std
    
    return lower_bound, upper_bound


def evaluate_uncertainty(y_true, mean_pred, lower_bound, upper_bound, confidence_level=0.95):
    """
    评估不确定性估计的质量
    
    Args:
        y_true: 真实标签
        mean_pred: 预测均值
        lower_bound: 预测下界
        upper_bound: 预测上界
        confidence_level: 置信水平
        
    Returns:
        dict: 评估指标
    """
    # 计算在预测区间内的样本比例
    within_interval = ((y_true >= lower_bound) & (y_true <= upper_bound))
    coverage_rate = within_interval.mean()
    
    # 计算评估指标
    metrics = {
        'coverage_rate': coverage_rate,
        'target_coverage': confidence_level,
        'coverage_error': abs(coverage_rate - confidence_level),
        'mean_interval_width': (upper_bound - lower_bound).mean(),
        'rmse': np.sqrt(mean_squared_error(y_true, mean_pred))
    }
    
    return metrics


def main():
    """不确定性量化模块主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(PATHS['log_dir'], PATHS['log_filename'])
    )
    
    logger.info("不确定性量化模块测试")
    
    # 创建并测试不确定性量化器
    from data_processor import load_data, preprocess_data, engineer_features, split_dataset
    
    # 加载数据
    df = load_data()
    
    # 预处理
    df = preprocess_data(df)
    
    # 特征工程
    df = engineer_features(df)
    
    # 分割数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    
    # 加载已训练的模型
    model_path = os.path.join(PATHS['model_dir'], PATHS['model_filename'])
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"已加载模型: {model_path}")
    else:
        # 如果没有已训练的模型，则训练一个简单模型
        from physics_ml import train_model
        model = train_model(X_train, y_train, X_val, y_val)
        logger.info("已训练新模型")
    
    # 创建不确定性量化器
    uq = UncertaintyQuantifier(model)
    
    # 量化测试集的预测不确定性
    uncertainty_results = uq.quantify_uncertainty(X_test, method='monte_carlo', y=y_test)
    
    # 评估不确定性估计的质量
    evaluation = evaluate_uncertainty(
        y_test, 
        uncertainty_results['mean'], 
        uncertainty_results['lower_bound'], 
        uncertainty_results['upper_bound']
    )
    
    logger.info(f"不确定性评估: 目标覆盖率={evaluation['target_coverage']:.2%}, "
                f"实际覆盖率={evaluation['coverage_rate']:.2%}, "
                f"区间宽度={evaluation['mean_interval_width']:.4f}")
    
    # 创建风险图
    risk_map = uq.create_risk_map(X_test, y_test)
    
    # 保存结果
    uq.save_results(risk_map['risk_data'])
    
    # 绘制不确定性可视化图
    fig = uq.plot_uncertainty(
        X_test, 
        y_test, 
        save_path=os.path.join(PATHS['results_dir'], 'uncertainty_visualization.png')
    )
    
    # 特征敏感性分析
    sensitivity_results = uq.analyze_sensitivities(X_test.iloc[0:1])
    
    # 绘制特征敏感性图
    fig = uq.plot_sensitivity(
        sensitivity_results, 
        save_path=os.path.join(PATHS['results_dir'], 'sensitivity_analysis.png')
    )
    
    return uq

if __name__ == "__main__":
    main()
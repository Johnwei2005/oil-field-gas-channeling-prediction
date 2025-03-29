#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统残差建模模块

本模块实现了基于物理模型的残差建模方法，将物理模型与机器学习模型结合，
提高预测精度，特别适用于小样本量数据集。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import logging
import joblib

# 导入配置
from config import PHYSICS_CONFIG, FEATURE_CONFIG, PATHS, DATA_CONFIG, MODEL_CONFIG

# 设置日志
logger = logging.getLogger(__name__)

class PhysicsBasedModel:
    """
    基础物理模型类，实现了简化的CO2气窜物理模型
    """
    def __init__(self):
        """初始化物理模型参数"""
        self.params = PHYSICS_CONFIG
        
    def predict(self, X):
        """
        使用物理模型进行预测
        
        Args:
            X: 特征DataFrame
            
        Returns:
            numpy.ndarray: 预测结果
        """
        # 检查必要的特征是否存在
        required_features = ['permeability', 'oil_viscosity', 'well_spacing', 
                            'effective_thickness', 'formation_pressure']
        
        for feature in required_features:
            if feature not in X.columns:
                logger.warning(f"物理模型缺少必要特征: {feature}")
                return np.zeros(len(X))
        
        # 简化的物理模型计算
        # 基于达西定律和CO2流动方程的简化模型
        
        # 计算流动系数
        k = X['permeability'].values  # 渗透率 (md)
        h = X['effective_thickness'].values  # 有效厚度 (m)
        mu = X['oil_viscosity'].values  # 油粘度 (mPa·s)
        L = X['well_spacing'].values  # 井距 (m)
        p = X['formation_pressure'].values  # 地层压力 (MPa)
        
        # 转换单位
        k_SI = k * 9.869e-16  # md 转换为 m²
        mu_SI = mu * 1e-3  # mPa·s 转换为 Pa·s
        
        # 计算流动系数
        flow_coefficient = (k_SI * h) / (mu_SI * L)
        
        # 计算压力梯度影响
        pressure_factor = p / self.params['reference_pressure']
        
        # 计算基础PV数预测值
        pv_pred = flow_coefficient * pressure_factor
        
        # 归一化预测结果
        pv_pred = pv_pred / np.max(pv_pred) * 5  # 假设最大PV数约为5
        
        return pv_pred

class ResidualModel:
    """
    残差建模类，结合物理模型和机器学习模型
    """
    def __init__(self, ml_model_type='random_forest'):
        """
        初始化残差模型
        
        Args:
            ml_model_type: 机器学习模型类型，可选'random_forest', 'gradient_boosting', 'gaussian_process'
        """
        self.physics_model = PhysicsBasedModel()
        self.ml_model_type = ml_model_type
        self.ml_model = self._create_ml_model()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_ml_model(self):
        """创建机器学习模型"""
        if self.ml_model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.ml_model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.ml_model_type == 'gaussian_process':
            kernel = ConstantKernel() * RBF() + WhiteKernel()
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=10,
                random_state=42
            )
        else:
            logger.warning(f"不支持的模型类型: {self.ml_model_type}，使用默认的随机森林")
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def fit(self, X, y):
        """
        训练残差模型
        
        Args:
            X: 特征DataFrame
            y: 目标变量
            
        Returns:
            self
        """
        # 使用物理模型进行预测
        physics_pred = self.physics_model.predict(X)
        
        # 计算残差
        residuals = y - physics_pred
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练机器学习模型预测残差
        self.ml_model.fit(X_scaled, residuals)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        使用残差模型进行预测
        
        Args:
            X: 特征DataFrame
            
        Returns:
            numpy.ndarray: 预测结果
        """
        if not self.is_fitted:
            logger.warning("模型尚未训练，无法进行预测")
            return np.zeros(len(X))
        
        # 使用物理模型进行预测
        physics_pred = self.physics_model.predict(X)
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 使用机器学习模型预测残差
        residual_pred = self.ml_model.predict(X_scaled)
        
        # 组合物理模型和残差预测
        final_pred = physics_pred + residual_pred
        
        return final_pred
    
    def evaluate(self, X, y):
        """
        评估模型性能
        
        Args:
            X: 特征DataFrame
            y: 目标变量
            
        Returns:
            dict: 评估指标
        """
        # 物理模型预测
        physics_pred = self.physics_model.predict(X)
        physics_r2 = r2_score(y, physics_pred)
        physics_rmse = np.sqrt(mean_squared_error(y, physics_pred))
        physics_mae = mean_absolute_error(y, physics_pred)
        
        # 残差模型预测
        if self.is_fitted:
            final_pred = self.predict(X)
            final_r2 = r2_score(y, final_pred)
            final_rmse = np.sqrt(mean_squared_error(y, final_pred))
            final_mae = mean_absolute_error(y, final_pred)
        else:
            final_r2, final_rmse, final_mae = 0, 0, 0
        
        # 计算改进百分比
        if physics_r2 > 0:
            r2_improvement = (final_r2 - physics_r2) / abs(physics_r2) * 100
        else:
            r2_improvement = np.inf if final_r2 > 0 else 0
            
        rmse_improvement = (physics_rmse - final_rmse) / physics_rmse * 100
        mae_improvement = (physics_mae - final_mae) / physics_mae * 100
        
        metrics = {
            'physics_r2': physics_r2,
            'physics_rmse': physics_rmse,
            'physics_mae': physics_mae,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'r2_improvement': r2_improvement,
            'rmse_improvement': rmse_improvement,
            'mae_improvement': mae_improvement
        }
        
        logger.info(f"物理模型 R²: {physics_r2:.4f}, RMSE: {physics_rmse:.4f}, MAE: {physics_mae:.4f}")
        logger.info(f"残差模型 R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
        logger.info(f"R²改进: {r2_improvement:.2f}%, RMSE改进: {rmse_improvement:.2f}%, MAE改进: {mae_improvement:.2f}%")
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        交叉验证评估模型性能
        
        Args:
            X: 特征DataFrame
            y: 目标变量
            cv: 交叉验证折数
            
        Returns:
            dict: 评估指标
        """
        # 初始化指标列表
        physics_r2_list = []
        final_r2_list = []
        
        # 创建交叉验证折
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X):
            # 分割数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 物理模型预测
            physics_pred = self.physics_model.predict(X_test)
            physics_r2 = r2_score(y_test, physics_pred)
            physics_r2_list.append(physics_r2)
            
            # 训练残差模型
            self.fit(X_train, y_train)
            
            # 残差模型预测
            final_pred = self.predict(X_test)
            final_r2 = r2_score(y_test, final_pred)
            final_r2_list.append(final_r2)
        
        # 计算平均指标
        avg_physics_r2 = np.mean(physics_r2_list)
        avg_final_r2 = np.mean(final_r2_list)
        
        # 计算改进百分比
        if avg_physics_r2 > 0:
            r2_improvement = (avg_final_r2 - avg_physics_r2) / abs(avg_physics_r2) * 100
        else:
            r2_improvement = np.inf if avg_final_r2 > 0 else 0
        
        metrics = {
            'avg_physics_r2': avg_physics_r2,
            'avg_final_r2': avg_final_r2,
            'r2_improvement': r2_improvement,
            'physics_r2_list': physics_r2_list,
            'final_r2_list': final_r2_list
        }
        
        logger.info(f"交叉验证 - 物理模型平均R²: {avg_physics_r2:.4f}")
        logger.info(f"交叉验证 - 残差模型平均R²: {avg_final_r2:.4f}")
        logger.info(f"交叉验证 - R²改进: {r2_improvement:.2f}%")
        
        return metrics
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            
        Returns:
            None
        """
        model_data = {
            'ml_model': self.ml_model,
            'scaler': self.scaler,
            'ml_model_type': self.ml_model_type,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存至 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            ResidualModel: 加载的模型
        """
        model_data = joblib.load(filepath)
        
        # 创建新模型实例
        model = cls(ml_model_type=model_data['ml_model_type'])
        
        # 加载保存的组件
        model.ml_model = model_data['ml_model']
        model.scaler = model_data['scaler']
        model.is_fitted = model_data['is_fitted']
        
        logger.info(f"模型已从 {filepath} 加载")
        return model
    
    def plot_predictions(self, X, y, output_path=None):
        """
        绘制预测结果对比图
        
        Args:
            X: 特征DataFrame
            y: 目标变量
            output_path: 输出文件路径
            
        Returns:
            None
        """
        # 物理模型预测
        physics_pred = self.physics_model.predict(X)
        
        # 残差模型预测
        if self.is_fitted:
            final_pred = self.predict(X)
        else:
            final_pred = physics_pred
        
        # 创建数据框用于绘图
        plot_df = pd.DataFrame({
            'Actual': y,
            'Physics Model': physics_pred,
            'Residual Model': final_pred
        })
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        # 物理模型
        plt.subplot(2, 2, 1)
        plt.scatter(y, physics_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel('实际值')
        plt.ylabel('物理模型预测值')
        plt.title(f'物理模型 (R² = {r2_score(y, physics_pred):.4f})')
        
        # 残差模型
        plt.subplot(2, 2, 2)
        plt.scatter(y, final_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel('实际值')
        plt.ylabel('残差模型预测值')
        plt.title(f'残差模型 (R² = {r2_score(y, final_pred):.4f})')
        
        # 残差分布
        plt.subplot(2, 2, 3)
        physics_residuals = y - physics_pred
        sns.histplot(physics_residuals, kde=True)
        plt.xlabel('物理模型残差')
        plt.ylabel('频率')
        plt.title('物理模型残差分布')
        
        # 残差模型残差分布
        plt.subplot(2, 2, 4)
        final_residuals = y - final_pred
        sns.histplot(final_residuals, kde=True)
        plt.xlabel('残差模型残差')
        plt.ylabel('频率')
        plt.title('残差模型残差分布')
        
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"预测结果对比图已保存至 {output_path}")
        
        plt.close()

def train_and_evaluate_residual_model(X, y, model_type='random_forest', test_size=0.2, cv=5):
    """
    训练和评估残差模型
    
    Args:
        X: 特征DataFrame
        y: 目标变量
        model_type: 机器学习模型类型
        test_size: 测试集比例
        cv: 交叉验证折数
        
    Returns:
        tuple: (模型, 评估指标)
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 创建并训练残差模型
    model = ResidualModel(ml_model_type=model_type)
    model.fit(X_train, y_train)
    
    # 评估模型
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    cv_metrics = model.cross_validate(X, y, cv=cv)
    
    # 绘制预测结果
    output_path = os.path.join(PATHS['results_dir'], f'residual_model_{model_type}_predictions.png')
    model.plot_predictions(X_test, y_test, output_path)
    
    # 保存模型
    model_path = os.path.join(PATHS['model_dir'], f'residual_model_{model_type}.pkl')
    model.save(model_path)
    
    # 汇总评估指标
    metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'cv': cv_metrics
    }
    
    return model, metrics

if __name__ == "__main__":
    # 测试代码
    from data_processor import load_data, preprocess_data, engineer_features
    from enhanced_features import optimize_features_for_small_sample
    
    # 加载数据
    df = load_data()
    
    # 预处理数据
    df = preprocess_data(df)
    
    # 基础特征工程
    df = engineer_features(df)
    
    # 增强特征工程
    target_col = DATA_CONFIG['target_column']
    df_optimized = optimize_features_for_small_sample(df, target_col)
    
    # 分离特征和目标
    X = df_optimized.drop(columns=[target_col])
    y = df_optimized[target_col]
    
    # 训练和评估残差模型
    model_types = ['random_forest', 'gradient_boosting', 'gaussian_process']
    best_model = None
    best_r2 = -np.inf
    
    for model_type in model_types:
        logger.info(f"训练和评估 {model_type} 残差模型")
        model, metrics = train_and_evaluate_residual_model(X, y, model_type=model_type)
        
        # 记录最佳模型
        if metrics['test']['final_r2'] > best_r2:
            best_r2 = metrics['test']['final_r2']
            best_model = model
            best_model_type = model_type
    
    logger.info(f"最佳模型: {best_model_type}, 测试集 R²: {best_r2:.4f}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统主程序

本程序是CCUS CO2气窜预测系统的入口点，负责协调数据处理、模型训练、不确定性量化和可视化等模块。
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# 导入自定义模块
from config import PATHS, DATA_CONFIG, MODEL_CONFIG, VIZ_CONFIG, VERSION, VERSION_STRING
from data_processor import load_data, preprocess_data, engineer_features, scale_features, split_dataset, feature_selection
from physics_ml import PhysicsGuidedXGBoost, PhysicsEnsembleModel, BayesianPhysicsModel, GaussianProcessPhysicsModel, train_model, evaluate_model, cross_validate
from uncertainty import UncertaintyQuantifier, monte_carlo_simulation, evaluate_uncertainty
from visualization import Visualizer, plot_feature_distribution, plot_training_history

# 设置日志
def setup_logging():
    """设置日志系统"""
    # 创建日志目录
    if not os.path.exists(PATHS['log_dir']):
        os.makedirs(PATHS['log_dir'])
    
    # 设置日志文件名
    log_filename = f"ccus_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(PATHS['log_dir'], log_filename)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 返回日志路径
    return log_filepath

# 主要功能类
class CCUSMigrationPredictor:
    """
    CCUS CO2气窜预测系统
    
    整合数据处理、模型训练、不确定性量化和可视化等功能，提供完整的气窜预测流程。
    """
    
# 在 main.py 的 CCUSMigrationPredictor 类初始化方法中
    def __init__(self, data_path=None, model_type=None, verbose=True):
        """
        初始化CCUS CO2气窜预测系统
        
        Args:
            data_path: 数据文件路径，若为None则使用配置中的默认路径
            model_type: 模型类型，若为None则使用配置中的默认模型
            verbose: 是否显示详细信息
        """
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        # 设置数据路径
        self.data_path = data_path if data_path else os.path.join(PATHS['data_dir'], DATA_CONFIG['file_path'])
        
        # 设置模型类型 - 默认使用高级集成模型
        self.model_type = model_type if model_type else MODEL_CONFIG.get('default_model', 'advanced_ensemble')
        
        # 创建可视化器
        self.visualizer = Visualizer()
        
        # 初始化模型和特征缩放器
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        
        # 初始化其他属性
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.uncertainty_quantifier = None
        
        # 记录特征工程信息
        self.feature_importance = None
        self.engineered_features = None
        
        self.logger.info(f"CCUS CO2气窜预测系统初始化完成 - {VERSION_STRING}")
        self.logger.info(f"数据文件: {self.data_path}")
        self.logger.info(f"模型类型: {self.model_type}")
    def load_and_process_data(self):
        """
        加载并处理数据
        
        Returns:
            pandas.DataFrame: 处理后的数据
        """
        self.logger.info("开始加载和处理数据")
        
        # 加载原始数据
        self.df = load_data(self.data_path, DATA_CONFIG['encoding'])
        
        # 预处理数据
        self.df = preprocess_data(self.df)
        
        # 特征工程
        self.df = engineer_features(self.df)
        
        # 记录特征信息
        self.engineered_features = self.df.columns.tolist()
        self.logger.info(f"特征工程后的特征数量: {len(self.engineered_features)}")
        
        # 目标变量分析和转换
        target_col = DATA_CONFIG['target_column']
        original_target_col = None  # 跟踪原始目标变量列名
        
        if target_col in self.df.columns:
            skewness = self.df[target_col].skew()
            self.logger.info(f"目标变量'{target_col}'的偏度: {skewness:.4f}")
            
            # 如果偏度大于1，考虑对数变换
            if skewness > 1.0:
                self.logger.info(f"目标变量分布偏斜，应用对数变换")
                # 保存原始目标变量，但使用一个不会被作为特征的变量名
                original_target_col = f"{target_col}_original"
                self.original_target_values = self.df[target_col].copy()  # 保存到类属性而不是DataFrame
                
                # 确保数据为正
                min_val = self.df[target_col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1e-3
                    self.df[target_col] = np.log1p(self.df[target_col] + shift)
                    self.target_transform = lambda x: np.expm1(x) - shift
                    self.inverse_transform = True
                    self.logger.info(f"应用对数变换 (log1p)，添加了{shift}的偏移量")
                else:
                    self.df[target_col] = np.log1p(self.df[target_col])
                    self.target_transform = lambda x: np.expm1(x)
                    self.inverse_transform = True
                    self.logger.info(f"应用对数变换 (log1p)")
                    
                self.logger.info(f"原始目标变量已保存到类属性，不会被用作特征")
            else:
                self.inverse_transform = False
                self.target_transform = lambda x: x
        
        # 分析数据分层分布
        bins = [0, 0.1, 0.5, float('inf')]  # 定义PV数分层边界
        labels = [0, 1, 2]  # 对应的标签
        
        if target_col in self.df.columns:
            y_strat = pd.cut(self.df[target_col], bins=bins, labels=labels)
            self.logger.info(f"样本分层分布: {y_strat.value_counts().to_dict()}")
            
            # 使用可能的分层抽样进行数据划分
            try_stratify = y_strat.value_counts().min() >= 3  # 至少需要3个样本
            
            # 检查并确保没有目标变量的原始值被包含在特征中
            if f"{target_col}_original" in self.df.columns:
                self.logger.warning(f"检测到目标变量的原始值在特征中，将其移除: {target_col}_original")
                self.df = self.df.drop(columns=[f"{target_col}_original"])
                
                # 更新特征列表
                if f"{target_col}_original" in self.engineered_features:
                    self.engineered_features.remove(f"{target_col}_original")
            
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_dataset(
                self.df, 
                target_column=target_col,
                test_size=DATA_CONFIG['test_size'],
                validation_size=DATA_CONFIG['validation_size'],
                random_state=DATA_CONFIG['random_state'],
                stratify=try_stratify  # 尝试分层抽样
            )
        else:
            # 常规数据划分
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_dataset(
                self.df, 
                target_column=target_col,
                test_size=DATA_CONFIG['test_size'],
                validation_size=DATA_CONFIG['validation_size'],
                random_state=DATA_CONFIG['random_state']
            )
        
        # 检查是否有目标变量相关的列在特征中
        if isinstance(self.X_train, pd.DataFrame):
            target_related_cols = [col for col in self.X_train.columns if target_col in col]
            if target_related_cols:
                self.logger.warning(f"特征中存在与目标变量相关的列，这可能导致数据泄露: {target_related_cols}")
                self.X_train = self.X_train.drop(columns=target_related_cols)
                self.X_val = self.X_val.drop(columns=target_related_cols)
                self.X_test = self.X_test.drop(columns=target_related_cols)
                self.logger.info(f"已从特征中移除与目标变量相关的列")
                
                # 更新特征列表
                for col in target_related_cols:
                    if col in self.engineered_features:
                        self.engineered_features.remove(col)
        
        # 特征选择，减少维度
        n_features = min(30, self.X_train.shape[1] // 2)  # 选择适当的特征数量
        if self.X_train.shape[1] > 50:  # 当特征过多时进行特征选择
            self.logger.info(f"原始特征数量({self.X_train.shape[1]})过多，执行特征选择")
            self.X_train, self.X_val, self.X_test, self.feature_selector = feature_selection(
                self.X_train, self.y_train, self.X_val, self.X_test, 
                method='importance', n_features=n_features
            )
            self.logger.info(f"特征选择后: 训练集 {self.X_train.shape}，验证集 {self.X_val.shape}，测试集 {self.X_test.shape}")
            
            # 记录保留的特征
            if hasattr(self.feature_selector, 'get_support') and isinstance(self.X_train, pd.DataFrame):
                selected_features = self.X_train.columns.tolist()
                self.logger.info(f"保留的特征: {', '.join(selected_features[:10])}...")
                
                # 更新engineered_features以反映特征选择后的特征集
                self.engineered_features = selected_features
        
        # 特征缩放
        self.X_train_scaled, self.feature_scaler = scale_features(self.X_train)
        self.X_val_scaled = self.feature_scaler.transform(self.X_val)
        self.X_test_scaled = self.feature_scaler.transform(self.X_test)
        
        # 检查特征缩放后是否有NaN值
        if isinstance(self.X_train_scaled, pd.DataFrame):
            if self.X_train_scaled.isnull().any().any():
                self.logger.warning("特征缩放后训练集中存在NaN值，将进行中位数填充")
                self.X_train_scaled = self.X_train_scaled.fillna(self.X_train_scaled.median())
                
        elif isinstance(self.X_train_scaled, np.ndarray):
            if np.isnan(self.X_train_scaled).any():
                self.logger.warning("特征缩放后训练集中存在NaN值，将进行中位数填充")
                # 对每列应用中位数填充
                for col in range(self.X_train_scaled.shape[1]):
                    mask = np.isnan(self.X_train_scaled[:, col])
                    if np.any(mask):
                        median_val = np.nanmedian(self.X_train_scaled[:, col])
                        self.X_train_scaled[mask, col] = median_val
        
        # 分析并记录训练数据分布情况
        if isinstance(self.y_train, pd.Series):
            # 计算高值样本比例
            high_value_ratio = (self.y_train > 0.1).mean()
            self.logger.info(f"训练集中高值(>0.1)样本比例: {high_value_ratio:.2%}")
            
            # 记录训练数据统计信息
            self.y_mean = self.y_train.mean()
            self.y_std = self.y_train.std()
            self.y_median = self.y_train.median()
            
            self.logger.info(f"训练集目标变量统计: 均值={self.y_mean:.4f}, 标准差={self.y_std:.4f}, 中位数={self.y_median:.4f}")
        
        self.logger.info(f"数据处理完成: {len(self.df)}行, {len(self.df.columns)}列")
        self.logger.info(f"训练集: {len(self.X_train)}样本, 验证集: {len(self.X_val)}样本, 测试集: {len(self.X_test)}样本")
        
        return self.df
    
    def train_and_evaluate_model(self):
        """
        训练并评估模型，使用优化的超参数
        
        Returns:
            dict: 评估指标字典
        """
        self.logger.info(f"开始训练{self.model_type}模型")
        
        # 训练模型，启用超参数优化
        self.model = train_model(
            self.X_train_scaled, 
            self.y_train, 
            self.X_val_scaled, 
            self.y_val, 
            model_type=self.model_type,
            optimize_params=True  # 启用超参数优化
        )
        
        # 评估模型
        metrics = evaluate_model(self.model, self.X_test_scaled, self.y_test)
        
        # 获取特征重要性
        if hasattr(self.model, 'get_feature_importance'):
            self.feature_importance = self.model.get_feature_importance()
            
            # 绘制特征重要性图
            if self.feature_importance:
                # 只显示前20个最重要的特征，避免图表过于拥挤
                sorted_features = sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20]
                
                features = [item[0] for item in sorted_features]
                importance = [item[1] for item in sorted_features]
                
                fig = self.visualizer.plot_feature_importance(
                    features, 
                    importance, 
                    title="特征重要性(TOP 20)"
                )
                
                # 保存图形
                self.visualizer.save_figure(fig, "feature_importance")
                
                # 输出特征重要性排名
                self.logger.info("特征重要性排名(TOP 10):")
                for i, (feat, imp) in enumerate(sorted_features[:10]):
                    self.logger.info(f"  {i+1}. {feat}: {imp:.4f}")
        
        self.logger.info(f"模型训练和评估完成: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        return metrics
    
    def cross_validate_model(self, n_splits=None):
        """
        交叉验证模型
        
        Args:
            n_splits: 交叉验证折数，若为None则使用配置中的设置
            
        Returns:
            dict: 交叉验证结果
        """
        self.logger.info(f"开始{self.model_type}模型的交叉验证")
        
        # 执行交叉验证
        cv_results = cross_validate(
            pd.concat([self.X_train, self.X_val]), 
            pd.concat([self.y_train, self.y_val]),
            model_type=self.model_type,
            n_splits=n_splits,
            loocv=MODEL_CONFIG.get('loocv', True)
        )
        
        # 绘制交叉验证结果图
        fig = self.visualizer.plot_cross_validation_results(
            cv_results,
            title=f"{self.model_type}模型交叉验证结果"
        )
        
        # 保存图形
        self.visualizer.save_figure(fig, "cross_validation_results")
        
        self.logger.info(f"交叉验证完成: RMSE={cv_results['rmse_mean']:.4f}±{cv_results['rmse_std']:.4f}")
        
        return cv_results
    
    def quantify_prediction_uncertainty(self, X=None, method='monte_carlo'):
        """
        量化预测不确定性
        
        Args:
            X: 预测特征，若为None则使用测试集
            method: 不确定性量化方法，可选 'model_based', 'bootstrap', 'monte_carlo'
            
        Returns:
            dict: 不确定性量化结果
        """
        self.logger.info(f"开始使用{method}方法量化预测不确定性")
        
        # 如果未指定特征矩阵，则使用测试集
        if X is None:
            X = self.X_test_scaled
        
        # 创建不确定性量化器（如果尚未创建）
        if self.uncertainty_quantifier is None:
            self.uncertainty_quantifier = UncertaintyQuantifier(self.model)
        
        # 量化不确定性
        uncertainty_results = self.uncertainty_quantifier.quantify_uncertainty(
            X, 
            method=method, 
            y=self.y_test if X is self.X_test_scaled else None,
            X_train=self.X_train_scaled,
            y_train=self.y_train
        )
        
        # 绘制不确定性可视化图
        fig = self.uncertainty_quantifier.plot_uncertainty(
            X,
            y=self.y_test if X is self.X_test_scaled else None,
            uncertainty_method=method
        )
        
        # 保存图形
        self.visualizer.save_figure(fig, f"uncertainty_{method}")
        
        # 创建风险图
        risk_map = self.uncertainty_quantifier.create_risk_map(
            X,
            y=self.y_test if X is self.X_test_scaled else None
        )
        
        # 保存风险数据
        risk_data = risk_map['risk_data']
        risk_data.to_csv(os.path.join(PATHS['results_dir'], 'risk_assessment.csv'), index=False)
        
        self.logger.info(f"预测不确定性量化完成，置信水平: {uncertainty_results['confidence_level']:.2%}")
        
        return uncertainty_results
    
    def analyze_feature_sensitivity(self, X=None, top_n=10):
        """
        分析特征敏感性，关注最重要的特征
        
        Args:
            X: 基准特征数据，若为None则使用测试集的第一个样本
            top_n: 显示前N个最敏感特征
            
        Returns:
            dict: 敏感性分析结果
        """
        self.logger.info("开始特征敏感性分析")
        
        # 如果未指定基准特征，则使用测试集的第一个样本
        if X is None:
            X = self.X_test_scaled[:1]
        
        # 使用不确定性量化器进行敏感性分析
        if self.uncertainty_quantifier is None:
            self.uncertainty_quantifier = UncertaintyQuantifier(self.model)
        
        # 获取特征重要性，优先分析重要特征
        important_features = None
        if hasattr(self.model, 'get_feature_importance'):
            feature_importance = self.model.get_feature_importance()
            if feature_importance:
                important_features = [f[0] for f in sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]]
                
                self.logger.info(f"将优先分析以下重要特征: {', '.join(important_features[:5])}...")
        

        # 只分析模型知道的特征
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            features_to_analyze = [f for f in self.X_train.columns if f in self.model.feature_names]
            if len(features_to_analyze) == 0:
                # 如果没有匹配的特征，使用所有特征
                features_to_analyze = self.X_train.columns.tolist()
        else:
            features_to_analyze = self.X_train.columns.tolist()

        self.logger.info(f"将分析{len(features_to_analyze)}个特征的敏感性")

        # 分析特征敏感性
        sensitivity_results = self.uncertainty_quantifier.analyze_sensitivities(
            X,
            feature_names=features_to_analyze
        )
        
        # 绘制特征敏感性图
        fig = self.uncertainty_quantifier.plot_sensitivity(
            sensitivity_results,
            top_n=top_n
        )
        
        # 保存图形
        self.visualizer.save_figure(fig, "feature_sensitivity")
        
        # 输出敏感性排名
        self.logger.info(f"特征敏感性分析完成，前5个最敏感特征:")
        for i, feature in enumerate(sensitivity_results['sorted_features'][:5]):
            sensitivity = sensitivity_results['sensitivities'][feature]['sensitivity']
            self.logger.info(f"  {i+1}. {feature}: {sensitivity:.4f}")
        
        return sensitivity_results
    
    def visualize_predictions(self, X=None, y=None):
        """
        可视化预测结果
        
        Args:
            X: 预测特征，若为None则使用测试集
            y: 真实标签，若为None则使用测试集标签
            
        Returns:
            list: 生成的图形对象列表
        """
        self.logger.info("开始可视化预测结果")
        
        # 如果未指定特征和标签，则使用测试集
        if X is None:
            X = self.X_test_scaled
        if y is None:
            y = self.y_test
        
        # 获取预测值
        y_pred = self.model.predict(X)
        
        # 创建图形列表
        figures = []
        
        # 绘制预测值与实际值对比图
        fig = self.visualizer.plot_prediction_vs_actual(
            y, 
            y_pred, 
            title="预测值与实际值对比"
        )
        self.visualizer.save_figure(fig, "prediction_vs_actual")
        figures.append(fig)
        
        # 绘制残差分析图
        fig = self.visualizer.plot_residuals(
            y, 
            y_pred, 
            title="残差分析"
        )
        self.visualizer.save_figure(fig, "residuals_analysis")
        figures.append(fig)
        
        # 如果有不确定性量化结果，则创建综合仪表盘
        if self.uncertainty_quantifier is not None:
            # 获取不确定性量化结果
            uncertainty_results = self.uncertainty_quantifier.quantify_uncertainty(
                X, 
                method='monte_carlo', 
                y=y
            )
            
            # 创建预测结果字典
            predictions = {
                'mean': uncertainty_results['mean'],
                'std': uncertainty_results['std'],
                'lower_bound': uncertainty_results['lower_bound'],
                'upper_bound': uncertainty_results['upper_bound'],
                'actual': y
            }
            
            # 绘制综合仪表盘
            fig = self.visualizer.plot_dashboard(
                X if isinstance(X, pd.DataFrame) else self.X_test,
                predictions,
                feature_importance=self.feature_importance,
                uncertainty=uncertainty_results,
                title="CO2气窜预测仪表盘"
            )
            self.visualizer.save_figure(fig, "prediction_dashboard")
            figures.append(fig)
        
        self.logger.info(f"预测结果可视化完成，生成了{len(figures)}个图形")
        
        return figures
    
    def save_model(self, filepath=None):
        """
        保存模型
        
        Args:
            filepath: 模型保存路径，若为None则使用配置中的默认路径
        """
        if self.model is None:
            self.logger.warning("模型尚未训练，无法保存")
            return
        
        # 设置保存路径
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], PATHS['model_filename'])
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        if hasattr(self.model, 'save_model'):
            self.model.save_model(filepath)
        else:
            joblib.dump(self.model, filepath)
        
        # 保存特征缩放器
        scaler_path = os.path.join(PATHS['model_dir'], 'feature_scaler.pkl')
        joblib.dump(self.feature_scaler, scaler_path)
        
        # 保存特征名称
        feature_names_path = os.path.join(PATHS['model_dir'], 'feature_names.pkl')
        joblib.dump(self.X_train.columns.tolist(), feature_names_path)
        
        self.logger.info(f"模型已保存至 {filepath}")
        self.logger.info(f"特征缩放器已保存至 {scaler_path}")
    
    def load_model(self, filepath=None):
        """
        加载模型
        
        Args:
            filepath: 模型加载路径，若为None则使用配置中的默认路径
        """
        # 设置加载路径
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], PATHS['model_filename'])
        
        # 加载模型
        if not os.path.exists(filepath):
            self.logger.warning(f"模型文件 {filepath} 不存在，无法加载")
            return False
        
        try:
            # 尝试使用模型类自带的加载方法
            if self.model_type == 'physics_guided_xgboost':
                self.model = PhysicsGuidedXGBoost.load_model(filepath)
            elif self.model_type == 'physics_informed_nn':
                # PhysicsInformedNN暂不可用，使用PhysicsGuidedXGBoost替代
                self.logger.warning("PhysicsInformedNN不可用，使用PhysicsGuidedXGBoost替代")
                self.model = PhysicsGuidedXGBoost.load_model(filepath)
            elif self.model_type == 'ensemble':
                self.model = PhysicsEnsembleModel.load_model(filepath)
            else:
                # 使用joblib加载
                self.model = joblib.load(filepath)
            
            # 加载特征缩放器
            scaler_path = os.path.join(PATHS['model_dir'], 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
            
            # 加载特征名称
            feature_names_path = os.path.join(PATHS['model_dir'], 'feature_names.pkl')
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
            
            self.logger.info(f"模型已从 {filepath} 加载")
            return True
        
        except Exception as e:
            self.logger.error(f"加载模型时出错: {str(e)}")
            return False
    
    def predict(self, X, return_uncertainty=False):
        """
        使用模型进行预测
        
        Args:
            X: 预测特征
            return_uncertainty: 是否返回不确定性估计
            
        Returns:
            numpy.ndarray 或 dict: 预测结果，如果return_uncertainty=True则返回包含均值和不确定性的字典
        """
        if self.model is None:
            self.logger.warning("模型尚未训练或加载，无法进行预测")
            return None
        
        # 预处理输入特征
        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.transform(X)
        else:
            X_scaled = X
        
        # 使用模型预测
        if return_uncertainty:
            # 量化不确定性
            if self.uncertainty_quantifier is None:
                self.uncertainty_quantifier = UncertaintyQuantifier(self.model)
            
            uncertainty_results = self.uncertainty_quantifier.quantify_uncertainty(
                X_scaled, 
                method='monte_carlo'
            )
            
            return uncertainty_results
        else:
            # 仅返回预测值
            return self.model.predict(X_scaled)
    
    def generate_report(self, output_dir=None):
        """
        生成综合报告
        
        Args:
            output_dir: 报告输出目录，若为None则使用配置中的默认路径
            
        Returns:
            str: 报告路径
        """
        self.logger.info("开始生成综合报告")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = PATHS['results_dir']
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 报告文件路径
        report_path = os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # 创建HTML报告
        with open(report_path, 'w', encoding='utf-8') as f:
            # 写入HTML头部
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>CCUS CO2气窜预测系统报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metrics {{ display: flex; flex-wrap: wrap; justify-content: space-between; margin: 20px 0; }}
                    .metric-card {{ background: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; width: 30%; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; margin: 10px 0; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 15px 0; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>CCUS CO2气窜预测系统报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>系统版本: {VERSION_STRING}</p>
                    
                    <h2>数据概览</h2>
            """)
            
            # 写入数据概览
            f.write(f"""
                    <p>数据文件: {self.data_path}</p>
                    <p>总样本数: {len(self.df) if self.df is not None else 'N/A'}</p>
                    <p>特征数量: {len(self.df.columns) - 1 if self.df is not None else 'N/A'}</p>
                    <p>训练集: {len(self.X_train) if self.X_train is not None else 'N/A'}样本</p>
                    <p>验证集: {len(self.X_val) if self.X_val is not None else 'N/A'}样本</p>
                    <p>测试集: {len(self.X_test) if self.X_test is not None else 'N/A'}样本</p>
                    
                    <h2>模型信息</h2>
                    <p>模型类型: {self.model_type}</p>
            """)
            
            # 获取模型评估指标并写入
            if self.model is not None:
                metrics = evaluate_model(self.model, self.X_test_scaled, self.y_test)
                
                f.write("""
                    <div class="metrics">
                """)
                
                for metric_name, metric_value in metrics.items():
                    f.write(f"""
                        <div class="metric-card">
                            <h3>{metric_name.upper()}</h3>
                            <div class="metric-value">{metric_value:.4f}</div>
                        </div>
                    """)
                
                f.write("""
                    </div>
                """)
            
            # 写入特征重要性
            if self.feature_importance:
                f.write("""
                    <h2>特征重要性</h2>
                    <table>
                        <tr>
                            <th>特征</th>
                            <th>重要性</th>
                        </tr>
                """)
                
                # 按重要性排序
                sorted_features = sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for feature, importance in sorted_features:
                    f.write(f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{importance:.4f}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                """)
            
            # 写入图片
            f.write("""
                <h2>可视化结果</h2>
                <div class="visuals">
            """)
            
            # 获取结果目录中的所有图片
            image_files = [f for f in os.listdir(output_dir) if f.endswith(f'.{VIZ_CONFIG["save_format"]}')]
            
            for image_file in image_files:
                image_path = os.path.join(output_dir, image_file)
                image_name = os.path.splitext(image_file)[0].replace('_', ' ').title()
                
                f.write(f"""
                    <div class="image-container">
                        <h3>{image_name}</h3>
                        <img src="{image_file}" alt="{image_name}">
                    </div>
                """)
            
            # 写入HTML尾部
            # 写入更详细的结论与建议
            if self.feature_importance:
                # 提取前5个最重要的特征
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                top_features_str = ', '.join([f"<strong>{feat}</strong> ({imp:.3f})" for feat, imp in top_features])
                
                # 获取模型性能指标
                model_metrics = evaluate_model(self.model, self.X_test_scaled, self.y_test)
                r2_value = model_metrics['r2']
                rmse_value = model_metrics['rmse']
                
                # 根据R²值评估模型质量
                model_quality = "优秀" if r2_value >= 0.85 else "良好" if r2_value >= 0.7 else "一般" if r2_value >= 0.5 else "较弱"
                
                # 写入更详细的HTML结论
                f.write(f"""
                </div>
                
                <h2>结论与建议</h2>
                
                <h3>模型性能评估</h3>
                <p>本次建模的决定系数(R²)为 <strong>{r2_value:.4f}</strong>，均方根误差(RMSE)为 <strong>{rmse_value:.4f}</strong>，
                模型预测能力 <strong>{model_quality}</strong>。</p>
                
                <h3>关键因素分析</h3>
                <p>对PV数影响最大的五个因素依次为: {top_features_str}。</p>
                
                <h3>工程建议</h3>
                <ul>
                    <li>建议优先关注和调控这些关键参数，以控制注入过程中的气窜现象</li>
                    <li>特别需要关注储层能量和注入强度比的平衡，合理控制注入速率</li>
                    <li>根据敏感性分析结果，在不同区块采用差异化的注入策略</li>
                    <li>定期监测关键参数变化，及时调整注入方案</li>
                </ul>
                
                <h3>预测的应用价值</h3>
                <p>当前模型已能较好地预测PV数，可用于指导实际生产中的注入参数优化，
                降低气窜风险，提高驱油效率。</p>
                
                </div>
            </body>
            </html>
                """)
            else:
                # 如果没有特征重要性信息，使用简化版结论
                f.write("""
                </div>
                
                <h2>结论与建议</h2>
                <p>基于以上分析，系统提供以下结论与建议：</p>
                <ul>
                    <li>建议关注特征重要性排名靠前的关键参数</li>
                    <li>预测结果的不确定性分析表明需要重点关注高风险区域</li>
                    <li>建议根据敏感性分析结果优化注入参数</li>
                </ul>
                
                </div>
            </body>
            </html>
                """)
        
        self.logger.info(f"综合报告已生成至 {report_path}")
        
        return report_path
    
    def run_full_workflow(self):
        self.logger.info("【进度1/8】开始执行完整工作流程")
        
        self.logger.info("【进度2/8】Step 1：数据加载与预处理开始")
        self.load_and_process_data()
        self.logger.info("【进度2/8】Step 1：数据加载与预处理完成")
        
        self.logger.info("【进度3/8】Step 2：模型训练与评估开始")
        metrics = self.train_and_evaluate_model()
        self.logger.info("【进度3/8】Step 2：模型训练与评估完成")
        
        self.logger.info("【进度4/8】Step 3：交叉验证开始")
        cv_results = self.cross_validate_model()
        self.logger.info("【进度4/8】Step 3：交叉验证完成")
        
        self.logger.info("【进度5/8】Step 4：预测不确定性量化开始")
        uncertainty_results = self.quantify_prediction_uncertainty()
        self.logger.info("【进度5/8】Step 4：预测不确定性量化完成")
        
        self.logger.info("【进度6/8】Step 5：特征敏感性分析开始")
        sensitivity_results = self.analyze_feature_sensitivity()
        self.logger.info("【进度6/8】Step 5：特征敏感性分析完成")
        
        self.logger.info("【进度7/8】Step 6：结果可视化开始")
        self.visualize_predictions()
        self.logger.info("【进度7/8】Step 6：结果可视化完成")
        
        self.logger.info("【进度8/8】Step 7：模型保存与报告生成开始")
        self.save_model()
        report_path = self.generate_report()
        self.logger.info("【进度8/8】Step 7：模型保存与报告生成完成")
        
        results = {
            'metrics': metrics,
            'cv_results': cv_results,
            'top_features': list(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))[:5] if self.feature_importance else None,
            'report_path': report_path
        }
        
        self.logger.info("【完成】完整工作流程执行完成")
        return results


# 命令行界面
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CCUS CO2气窜预测系统')
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 训练模型命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data', type=str, help='数据文件路径')
    train_parser.add_argument('--model', type=str, choices=['physics_guided_xgboost', 'physics_informed_nn', 'ensemble'],
                              help='模型类型')
    train_parser.add_argument('--output', type=str, help='模型输出路径')
    train_parser.add_argument('--cv', action='store_true', help='执行交叉验证')
    train_parser.add_argument('--report', action='store_true', help='生成报告')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='使用模型进行预测')
    predict_parser.add_argument('--model', type=str, help='模型文件路径')
    predict_parser.add_argument('--input', type=str, required=True, help='输入数据文件路径')
    predict_parser.add_argument('--output', type=str, help='预测结果输出路径')
    predict_parser.add_argument('--uncertainty', action='store_true', help='量化预测不确定性')
    
    # 评估命令
    evaluate_parser = subparsers.add_parser('evaluate', help='评估模型性能')
    evaluate_parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    evaluate_parser.add_argument('--data', type=str, required=True, help='评估数据文件路径')
    evaluate_parser.add_argument('--output', type=str, help='评估结果输出路径')
    
    # 可视化命令
    visualize_parser = subparsers.add_parser('visualize', help='可视化预测结果')
    visualize_parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    visualize_parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    visualize_parser.add_argument('--output', type=str, help='可视化结果输出路径')
    visualize_parser.add_argument('--type', type=str, choices=['map', 'comparison', 'sensitivity', 'dashboard'],
                                help='可视化类型')
    
    # 完整工作流命令
    workflow_parser = subparsers.add_parser('workflow', help='执行完整工作流程')
    workflow_parser.add_argument('--data', type=str, help='数据文件路径')
    workflow_parser.add_argument('--model', type=str, help='模型类型')
    workflow_parser.add_argument('--output', type=str, help='输出目录')
    
    # 版本命令
    version_parser = subparsers.add_parser('version', help='显示版本信息')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 设置日志
    log_filepath = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"启动CCUS CO2气窜预测系统 - {VERSION_STRING}")
    
    # 解析命令行参数
    args = parse_args()
    
    # 根据命令执行相应功能
    if args.command == 'train':
        # 训练模型
        predictor = CCUSMigrationPredictor(
            data_path=args.data,
            model_type=args.model
        )
        
        # 加载和处理数据
        predictor.load_and_process_data()
        
        # 训练和评估模型
        metrics = predictor.train_and_evaluate_model()
        
        # 如果需要交叉验证
        if args.cv:
            cv_results = predictor.cross_validate_model()
            logger.info(f"交叉验证结果: RMSE={cv_results['rmse_mean']:.4f}±{cv_results['rmse_std']:.4f}")
        
        # 保存模型
        if args.output:
            predictor.save_model(args.output)
        else:
            predictor.save_model()
        
        # 如果需要生成报告
        if args.report:
            report_path = predictor.generate_report()
            logger.info(f"报告已生成至 {report_path}")
        
        logger.info(f"模型训练完成: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    elif args.command == 'predict':
        # 使用模型进行预测
        predictor = CCUSMigrationPredictor()
        
        # 加载模型
        if not predictor.load_model(args.model):
            logger.error("加载模型失败，无法进行预测")
            return
        
        # 加载输入数据
        try:
            input_data = pd.read_csv(args.input)
            logger.info(f"已加载输入数据: {args.input}, {len(input_data)}行")
        except Exception as e:
            logger.error(f"加载输入数据失败: {str(e)}")
            return
        
        # 进行预测
        if args.uncertainty:
            predictions = predictor.predict(input_data, return_uncertainty=True)
            
            # 保存预测结果
            if args.output:
                output_path = args.output
            else:
                output_path = os.path.join(PATHS['results_dir'], 'predictions_with_uncertainty.csv')
            
            # 创建预测结果DataFrame
            pred_df = input_data.copy()
            pred_df['predicted_value'] = predictions['mean']
            pred_df['prediction_std'] = predictions['std']
            pred_df['lower_bound'] = predictions['lower_bound']
            pred_df['upper_bound'] = predictions['upper_bound']
            
            # 保存结果
            pred_df.to_csv(output_path, index=False)
            logger.info(f"带不确定性的预测结果已保存至 {output_path}")
        else:
            predictions = predictor.predict(input_data)
            
            # 保存预测结果
            if args.output:
                output_path = args.output
            else:
                output_path = os.path.join(PATHS['results_dir'], 'predictions.csv')
            
            # 创建预测结果DataFrame
            pred_df = input_data.copy()
            pred_df['predicted_value'] = predictions
            
            # 保存结果
            pred_df.to_csv(output_path, index=False)
            logger.info(f"预测结果已保存至 {output_path}")
    
    elif args.command == 'evaluate':
        # 评估模型性能
        predictor = CCUSMigrationPredictor()
        
        # 加载模型
        if not predictor.load_model(args.model):
            logger.error("加载模型失败，无法进行评估")
            return
        
        # 加载评估数据
        try:
            eval_data = pd.read_csv(args.data)
            logger.info(f"已加载评估数据: {args.data}, {len(eval_data)}行")
        except Exception as e:
            logger.error(f"加载评估数据失败: {str(e)}")
            return
        
        # 分离特征和标签
        if DATA_CONFIG['target_column'] in eval_data.columns:
            X_eval = eval_data.drop(columns=[DATA_CONFIG['target_column']])
            y_eval = eval_data[DATA_CONFIG['target_column']]
            
            # 进行预测
            if predictor.feature_scaler is not None:
                X_eval_scaled = predictor.feature_scaler.transform(X_eval)
            else:
                X_eval_scaled = X_eval
            
            # 评估模型
            metrics = evaluate_model(predictor.model, X_eval_scaled, y_eval)
            
            # 输出评估结果
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
            
            # 保存评估结果
            if args.output:
                output_path = args.output
            else:
                output_path = os.path.join(PATHS['results_dir'], 'evaluation_results.json')
            
            # 保存结果
            import json
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"评估结果已保存至 {output_path}")
            
            # 创建可视化器
            visualizer = Visualizer()
            
            # 绘制预测vs实际图
            y_pred = predictor.model.predict(X_eval_scaled)
            fig = visualizer.plot_prediction_vs_actual(y_eval, y_pred)
            visualizer.save_figure(fig, "eval_prediction_vs_actual")
            
            # 绘制残差分析图
            fig = visualizer.plot_residuals(y_eval, y_pred)
            visualizer.save_figure(fig, "eval_residuals")
        else:
            logger.error(f"评估数据中未找到目标列: {DATA_CONFIG['target_column']}")
    
    elif args.command == 'visualize':
        # 可视化预测结果
        predictor = CCUSMigrationPredictor()
        
        # 加载模型
        if not predictor.load_model(args.model):
            logger.error("加载模型失败，无法进行可视化")
            return
        
        # 加载数据
        try:
            data = pd.read_csv(args.data)
            logger.info(f"已加载数据: {args.data}, {len(data)}行")
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return
        
        # 创建可视化器
        visualizer = Visualizer()
        
        # 设置输出目录
        output_dir = args.output if args.output else PATHS['results_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据可视化类型创建不同图形
        if args.type == 'map':
            # 需要空间坐标列
            if 'x' in data.columns and 'y' in data.columns:
                # 进行预测
                predictions = predictor.predict(data)
                
                # 绘制气窜分布图
                fig = visualizer.plot_co2_migration_map(
                    data['x'], 
                    data['y'], 
                    predictions,
                    title="CO2气窜分布预测图"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "co2_migration_map")
                logger.info(f"气窜分布图已保存至 {filepath}")
                
                # 如果支持3D绘图，则创建3D图
                if VIZ_CONFIG['3d_plot']:
                    fig = visualizer.plot_3d_migration(
                        data['x'],
                        data['y'],
                        predictions,
                        title="CO2气窜3D分布预测图"
                    )
                    
                    if fig:
                        filepath = visualizer.save_figure(fig, "co2_migration_3d")
                        logger.info(f"气窜3D分布图已保存至 {filepath}")
            else:
                logger.error("数据中未找到空间坐标列(x, y)，无法创建分布图")
        
        elif args.type == 'comparison':
            # 需要目标列进行比较
            if DATA_CONFIG['target_column'] in data.columns:
                # 分离特征和标签
                X = data.drop(columns=[DATA_CONFIG['target_column']])
                y = data[DATA_CONFIG['target_column']]
                
                # 进行预测
                if predictor.feature_scaler is not None:
                    X_scaled = predictor.feature_scaler.transform(X)
                else:
                    X_scaled = X
                
                y_pred = predictor.model.predict(X_scaled)
                
                # 绘制预测vs实际图
                fig = visualizer.plot_prediction_vs_actual(
                    y, 
                    y_pred,
                    title="预测值与实际值对比"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "prediction_vs_actual")
                logger.info(f"预测对比图已保存至 {filepath}")
                
                # 绘制残差分析图
                fig = visualizer.plot_residuals(
                    y, 
                    y_pred,
                    title="残差分析"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "residuals_analysis")
                logger.info(f"残差分析图已保存至 {filepath}")
            else:
                logger.error(f"数据中未找到目标列: {DATA_CONFIG['target_column']}")
        
        elif args.type == 'sensitivity':
            # 分析特征敏感性
            # 使用不确定性量化器
            uncertainty_quantifier = UncertaintyQuantifier(predictor.model)
            
            # 使用第一个样本进行敏感性分析
            sensitivity_results = uncertainty_quantifier.analyze_sensitivities(
                data.iloc[0:1],
                feature_names=data.columns.tolist()
            )
            
            # 绘制特征敏感性图
            fig = uncertainty_quantifier.plot_sensitivity(
                sensitivity_results,
                top_n=10
            )
            
            # 保存图形
            filepath = visualizer.save_figure(fig, "feature_sensitivity")
            logger.info(f"特征敏感性图已保存至 {filepath}")
        
        elif args.type == 'dashboard':
            # 创建综合仪表盘
            # 需要目标列
            if DATA_CONFIG['target_column'] in data.columns:
                # 分离特征和标签
                X = data.drop(columns=[DATA_CONFIG['target_column']])
                y = data[DATA_CONFIG['target_column']]
                
                # 进行预测并量化不确定性
                uncertainty_quantifier = UncertaintyQuantifier(predictor.model)
                
                # 如果有特征缩放器，则应用
                if predictor.feature_scaler is not None:
                    X_scaled = predictor.feature_scaler.transform(X)
                else:
                    X_scaled = X
                
                # 量化不确定性
                uncertainty_results = uncertainty_quantifier.quantify_uncertainty(
                    X_scaled, 
                    method='monte_carlo', 
                    y=y
                )
                
                # 获取特征重要性
                feature_importance = None
                if hasattr(predictor.model, 'get_feature_importance'):
                    feature_importance = predictor.model.get_feature_importance()
                
                # 创建预测结果字典
                predictions = {
                    'mean': uncertainty_results['mean'],
                    'std': uncertainty_results['std'],
                    'lower_bound': uncertainty_results['lower_bound'],
                    'upper_bound': uncertainty_results['upper_bound'],
                    'actual': y
                }
                
                # 绘制仪表盘
                fig = visualizer.plot_dashboard(
                    X,
                    predictions,
                    feature_importance=feature_importance,
                    uncertainty=uncertainty_results,
                    title="CO2气窜预测仪表盘"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "prediction_dashboard")
                logger.info(f"预测仪表盘已保存至 {filepath}")
            else:
                logger.error(f"数据中未找到目标列: {DATA_CONFIG['target_column']}")
        
        else:
            # 默认创建所有可能的可视化
            # 进行预测
            predictions = predictor.predict(data, return_uncertainty=True)
            
            # 如果有目标列，则创建对比图
            if DATA_CONFIG['target_column'] in data.columns:
                y = data[DATA_CONFIG['target_column']]
                
                # 绘制预测vs实际图
                fig = visualizer.plot_prediction_vs_actual(
                    y, 
                    predictions['mean'],
                    title="预测值与实际值对比"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "prediction_vs_actual")
                logger.info(f"预测对比图已保存至 {filepath}")
                
                # 绘制残差分析图
                fig = visualizer.plot_residuals(
                    y, 
                    predictions['mean'],
                    title="残差分析"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "residuals_analysis")
                logger.info(f"残差分析图已保存至 {filepath}")
            
            # 如果有空间坐标，则创建分布图
            if 'x' in data.columns and 'y' in data.columns:
                # 绘制气窜分布图
                fig = visualizer.plot_co2_migration_map(
                    data['x'], 
                    data['y'], 
                    predictions['mean'],
                    title="CO2气窜分布预测图"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "co2_migration_map")
                logger.info(f"气窜分布图已保存至 {filepath}")
            
            # 绘制不确定性图
            if 'x' in data.columns and 'y' in data.columns:
                # 创建不确定性分布图
                fig = visualizer.plot_uncertainty_map(
                    data['x'].values.reshape(-1, 1), 
                    data['y'].values.reshape(-1, 1),
                    predictions['mean'].reshape(-1, 1),
                    predictions['std'].reshape(-1, 1),
                    title="预测不确定性分布图"
                )
                
                # 保存图形
                filepath = visualizer.save_figure(fig, "uncertainty_map")
                logger.info(f"不确定性分布图已保存至 {filepath}")
    
    elif args.command == 'workflow':
        # 执行完整工作流程
        predictor = CCUSMigrationPredictor(
            data_path=args.data,
            model_type=args.model
        )
        
        # 执行工作流
        results = predictor.run_full_workflow()
        
        # 输出结果摘要
        logger.info("工作流执行完成，结果摘要:")
        logger.info(f"RMSE: {results['metrics']['rmse']:.4f}")
        logger.info(f"R²: {results['metrics']['r2']:.4f}")
        logger.info(f"报告路径: {results['report_path']}")
        
        if results['top_features']:
            logger.info("重要特征:")
            for feature, importance in results['top_features']:
                logger.info(f"  {feature}: {importance:.4f}")
    
    elif args.command == 'version':
        # 显示版本信息
        print(f"CCUS CO2气窜预测系统 - {VERSION_STRING}")
        print(f"发布日期: {VERSION['release_date']}")
    
    else:
        # 未指定命令，显示帮助信息
        print("请指定要执行的命令。使用 --help 查看帮助信息。")
        print(f"CCUS CO2气窜预测系统 - {VERSION_STRING}")


if __name__ == "__main__":
    main()
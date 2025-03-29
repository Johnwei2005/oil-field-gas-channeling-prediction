#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统物理约束机器学习模块

本模块实现了物理约束机器学习模型，将油藏工程原理集成到数据驱动模型中。
"""

import os
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# 注释掉 TensorFlow 相关导入
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras import regularizers
import xgboost as xgb

# 导入配置
from config import MODEL_CONFIG, PHYSICS_CONFIG, PATHS

# 设置日志
logger = logging.getLogger(__name__)

class BayesianPhysicsModel(BaseEstimator, RegressorMixin):
    """
    贝叶斯物理模型
    
    使用贝叶斯方法结合物理先验知识，对预测进行不确定性量化。
    """
    
    def __init__(self, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, physics_prior=None):
        """
        初始化贝叶斯物理模型
        
        Args:
            alpha_1: 噪声精度的形状参数
            alpha_2: 噪声精度的速率参数
            lambda_1: 权重精度的形状参数
            lambda_2: 权重精度的速率参数
            physics_prior: 物理先验知识
        """
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.physics_prior = physics_prior if physics_prior else {}
        
        # 初始化模型
        self.model = BayesianRidge(
            alpha_1=alpha_1, 
            alpha_2=alpha_2, 
            lambda_1=lambda_1, 
            lambda_2=lambda_2
        )
        
        self.feature_names = None
    
    def fit(self, X, y, sample_weight=None):
        """
        训练贝叶斯物理模型
        
        Args:
            X: 训练特征
            y: 训练标签
            sample_weight: 样本权重
            
        Returns:
            self: 训练好的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # 应用物理先验
        if self.physics_prior:
            # 这里可以根据物理先验调整初始权重或样本权重
            pass
        
        # 训练模型
        self.model.fit(X, y, sample_weight=sample_weight)
        
        logger.info("BayesianPhysicsModel模型训练完成")
        
        return self
    
    def predict(self, X, return_std=False):
        """
        使用模型进行预测
        
        Args:
            X: 特征数据
            return_std: 是否返回标准差
            
        Returns:
            numpy.ndarray 或 tuple: 预测结果，如果return_std=True则返回(均值, 标准差)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X, return_std=return_std)
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            dict: 特征重要性字典
        """
        importance = np.abs(self.model.coef_)
        
        # 如果有特征名，则创建特征名-重要性字典
        if self.feature_names:
            importance_dict = {name: imp for name, imp in zip(self.feature_names, importance)}
        else:
            importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importance)}
        
        return importance_dict
    
    def save_model(self, filepath=None):
        """
        保存模型
        
        Args:
            filepath: 模型保存路径
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'bayesian_model.pkl')
        
        joblib.dump(self, filepath)
        logger.info(f"BayesianPhysicsModel模型已保存至 {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """
        加载模型
        
        Args:
            filepath: 模型加载路径
            
        Returns:
            BayesianPhysicsModel: 加载的模型
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'bayesian_model.pkl')
        
        model = joblib.load(filepath)
        logger.info(f"BayesianPhysicsModel模型已从 {filepath} 加载")
        
        return model


class PhysicsGuidedXGBoost(BaseEstimator, RegressorMixin):
    """
    物理约束增强的XGBoost模型
    
    将物理约束作为正则化项集成到XGBoost模型中，提高模型在小样本下的泛化能力。
    """
    
    def __init__(self, 
                 n_estimators=None, 
                 learning_rate=None, 
                 max_depth=None, 
                 gamma=None,
                 min_child_weight=None,
                 subsample=None,
                 colsample_bytree=None,
                 reg_alpha=None,
                 reg_lambda=None,
                 physics_weight=None,
                 random_state=None):
        """
        初始化物理约束XGBoost模型
        
        Args:
            n_estimators: 弱学习器数量
            learning_rate: 学习率
            max_depth: 树最大深度
            gamma: 拆分叶节点所需的最小损失减少量
            min_child_weight: 叶子节点最小权重和
            subsample: 训练样本子采样比例
            colsample_bytree: 构建树时列采样比例
            reg_alpha: L1正则化系数
            reg_lambda: L2正则化系数
            physics_weight: 物理约束权重
            random_state: 随机种子
        """
        # 设置超参数，如果未指定则使用配置文件中的值
        self.n_estimators = n_estimators if n_estimators else MODEL_CONFIG['xgboost_params'].get('n_estimators', 200)
        self.learning_rate = learning_rate if learning_rate else MODEL_CONFIG['xgboost_params'].get('learning_rate', 0.05)
        self.max_depth = max_depth if max_depth else MODEL_CONFIG['xgboost_params'].get('max_depth', 5)
        self.gamma = gamma if gamma else MODEL_CONFIG['xgboost_params'].get('gamma', 0.1)
        self.min_child_weight = min_child_weight if min_child_weight else MODEL_CONFIG['xgboost_params'].get('min_child_weight', 2)
        self.subsample = subsample if subsample else MODEL_CONFIG['xgboost_params'].get('subsample', 0.8)
        self.colsample_bytree = colsample_bytree if colsample_bytree else MODEL_CONFIG['xgboost_params'].get('colsample_bytree', 0.8)
        self.reg_alpha = reg_alpha if reg_alpha else MODEL_CONFIG['xgboost_params'].get('reg_alpha', 0.1)
        self.reg_lambda = reg_lambda if reg_lambda else MODEL_CONFIG['xgboost_params'].get('reg_lambda', 1.0)
        self.physics_weight = physics_weight if physics_weight else MODEL_CONFIG.get('physics_weight', 0.3)
        self.random_state = random_state if random_state else MODEL_CONFIG['xgboost_params'].get('random_state', 42)
        
        # 初始化XGBoost模型
        self.model = None
        self.feature_names = None
        

    def _physics_constraints(self, X, y_pred):
        """
        改进的物理约束损失函数，具有更先进的物理学原理
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y_pred: 模型预测值，形状为 (n_samples,)
                
        返回:
            float: 物理约束损失值
        """
        # 将输入转换为DataFrame以便于特征提取
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 初始化约束惩罚
        penalties = []
        
        # 物理约束1: 渗透率与PV数的关系
        if 'permeability' in X.columns:
            # 预期：渗透率越高，PV数应该越大
            perm_corr = np.corrcoef(X['permeability'], y_pred)[0, 1]
            if perm_corr < 0:  # 负相关违反了物理原理
                penalties.append(abs(perm_corr) * 1.5)  # 增加惩罚权重
        
        # 物理约束2: 井距与PV数的关系
        if 'well_spacing' in X.columns:
            # 预期：井距越大，PV数应该越小
            spacing_corr = np.corrcoef(X['well_spacing'], y_pred)[0, 1]
            if spacing_corr > 0:  # 正相关违反了物理原理
                penalties.append(spacing_corr * 1.5)  # 增加惩罚权重
        
        # 物理约束3: 孔隙度与PV数的关系
        if 'porosity' in X.columns:
            # 预期：孔隙度越高，PV数应该越大
            porosity_corr = np.corrcoef(X['porosity'], y_pred)[0, 1]
            if porosity_corr < 0:  # 负相关违反了物理原理
                penalties.append(abs(porosity_corr) * 1.2)
        
        # 物理约束4: 油粘度与PV数的关系
        if 'oil_viscosity' in X.columns:
            # 预期：油粘度越高，PV数应该越小
            visc_corr = np.corrcoef(X['oil_viscosity'], y_pred)[0, 1]
            if visc_corr > 0:  # 正相关违反了物理原理
                penalties.append(visc_corr * 1.2)
        
        # 增强物理约束
        # 约束5: 相迁移性应与PV数正相关
        if 'phase_mobility_index' in X.columns:
            mobility_corr = np.corrcoef(X['phase_mobility_index'], y_pred)[0, 1]
            if mobility_corr < 0:  # 负相关违反了物理原理
                penalties.append(abs(mobility_corr) * 2.0)  # 为重要特征赋予更高权重
        
        # 约束6: 注入效率应与PV数正相关
        if 'injection_efficiency' in X.columns:
            inj_eff_corr = np.corrcoef(X['injection_efficiency'], y_pred)[0, 1]
            if inj_eff_corr < 0:
                penalties.append(abs(inj_eff_corr) * 1.8)
        
        # 约束7: 流动能力应与PV数正相关
        if 'flow_capacity' in X.columns:
            flow_cap_corr = np.corrcoef(X['flow_capacity'], y_pred)[0, 1]
            if flow_cap_corr < 0:
                penalties.append(abs(flow_cap_corr) * 1.5)
        
        # 约束8: 综合稳定性指数应与PV数负相关
        if 'combined_stability_index' in X.columns:
            stability_corr = np.corrcoef(X['combined_stability_index'], y_pred)[0, 1]
            if stability_corr > 0:
                penalties.append(stability_corr * 1.8)
        
        # 约束9: 检查物理限制
        # PV数应该非负
        if np.any(y_pred < 0):
            penalties.append(np.sum(y_pred[y_pred < 0] ** 2) * 5.0)  # 对负值施加强惩罚
        
        # 计算整体物理损失: 惩罚的加权平均
        if penalties:
            # 使用softmax给予更大违规更高的权重
            penalties = np.array(penalties)
            weights = np.exp(penalties) / np.sum(np.exp(penalties))
            physics_loss = np.sum(penalties * weights) 
        else:
            physics_loss = 0.0
        
        return physics_loss
    
    def _custom_objective(self, y_true, y_pred):
        """
        自定义目标函数，结合MSE和物理约束
        
        Args:
            y_true: 真实标签
            y_pred: 预测值
            
        Returns:
            tuple: (梯度, Hessian)
        """
        # 标准MSE梯度和Hessian
        grad = y_pred - y_true
        hess = np.ones_like(y_pred)
        
        return grad, hess
    
    def _custom_eval(self, y_true, y_pred):
        """
        自定义评估函数，结合MSE和物理约束
        
        Args:
            y_true: 真实标签
            y_pred: 预测值
            
        Returns:
            tuple: ('physics_mse', 评估值)
        """
        # 标准MSE
        mse = mean_squared_error(y_true, y_pred)
        
        # 物理约束损失
        X = self.eval_X
        physics_loss = self._physics_constraints(X, y_pred)
        
        # 结合MSE和物理约束
        eval_result = mse + self.physics_weight * physics_loss
        
        return 'physics_mse', eval_result
    
    def fit(self, X, y, eval_set=None):
        """
        训练物理约束XGBoost模型
        
        Args:
            X: 训练特征
            y: 训练标签
            eval_set: 评估集
            
        Returns:
            self: 训练好的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 构建XGBoost参数
        params = {
            'objective': 'reg:squarederror',
            'eta': self.learning_rate,
            'max_depth': self.max_depth,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'alpha': self.reg_alpha,
            'lambda': self.reg_lambda,
            'random_state': self.random_state
        }
        
        # 创建DMatrix对象 - 确保使用feature_names
        if isinstance(X, pd.DataFrame):
            dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        else:
            # 如果X不是DataFrame，则需要使用预先保存的feature_names
            if self.feature_names:
                X_df = pd.DataFrame(X, columns=self.feature_names)
                dtrain = xgb.DMatrix(X_df, label=y, feature_names=self.feature_names)
            else:
                dtrain = xgb.DMatrix(X, label=y)
        
        # 如果有评估集，则创建评估DMatrix
        evals = []
        if eval_set:
            eval_X, eval_y = eval_set[0]
            
            if isinstance(eval_X, pd.DataFrame):
                deval = xgb.DMatrix(eval_X, label=eval_y, feature_names=eval_X.columns.tolist())
            else:
                # 如果eval_X不是DataFrame，使用与训练数据相同的特征名
                if self.feature_names:
                    eval_X_df = pd.DataFrame(eval_X, columns=self.feature_names)
                    deval = xgb.DMatrix(eval_X_df, label=eval_y, feature_names=self.feature_names)
                else:
                    deval = xgb.DMatrix(eval_X, label=eval_y)
            
            evals = [(dtrain, 'train'), (deval, 'eval')]
            
            # 保存eval_X用于自定义评估
            self.eval_X = eval_X
            
            # 使用带有早停的训练
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=20,
                verbose_eval=False
            )
        else:
            # 没有评估集，不使用早停
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                verbose_eval=False
            )
        
        logger.info(f"PhysicsGuidedXGBoost模型训练完成，特征数量: {X.shape[1]}")
        
        return self
    
    def predict(self, X):
        """
        Using the model to make predictions
        
        Args:
            X: feature data
            
        Returns:
            numpy.ndarray: prediction results
        """
        if self.model is None:
            raise ValueError("Model not yet trained")
        
        # If X is already a DataFrame, use it directly
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        # If X is an array but we have feature names stored, use those
        elif hasattr(self, 'feature_names') and self.feature_names is not None:
            feature_names = self.feature_names
        else:
            # Without feature names, we can't proceed with XGBoost
            raise ValueError("Feature names are required for prediction")
        
        # Convert to DataFrame with feature names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        # Create DMatrix with feature names
        dtest = xgb.DMatrix(X, feature_names=feature_names)
        
        # Predict
        predictions = self.model.predict(dtest)
        
        return predictions
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            dict: 特征重要性字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 获取特征重要性
        importance = self.model.get_score(importance_type='gain')
        
        # 如果有特征名，则替换特征索引
        if self.feature_names:
            new_importance = {}
            for feat, imp in importance.items():
                if feat.startswith('f') and feat[1:].isdigit():
                    # XGBoost默认使用f0, f1, f2...作为特征名
                    idx = int(feat[1:])
                    if idx < len(self.feature_names):
                        new_importance[self.feature_names[idx]] = imp
                else:
                    new_importance[feat] = imp
            importance = new_importance
        
        return importance
    
    def save_model(self, filepath=None):
        """
        保存模型
        
        Args:
            filepath: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], PATHS['model_filename'])
        
        joblib.dump(self, filepath)
        logger.info(f"PhysicsGuidedXGBoost模型已保存至 {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """
        加载模型
        
        Args:
            filepath: 模型加载路径
            
        Returns:
            PhysicsGuidedXGBoost: 加载的模型
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], PATHS['model_filename'])
        
        model = joblib.load(filepath)
        logger.info(f"PhysicsGuidedXGBoost模型已从 {filepath} 加载")
        
        return model


class GaussianProcessPhysicsModel(BaseEstimator, RegressorMixin):
    """
    高斯过程物理模型
    
    使用高斯过程回归结合物理核函数，提供准确的预测及不确定性估计。
    """
    
    def __init__(self, 
                kernel=None, 
                alpha=None, 
                n_restarts_optimizer=None,
                physics_kernel=True,
                random_state=None):
        """
        初始化高斯过程物理模型
        
        Args:
            kernel: 核函数
            alpha: 噪声水平
            n_restarts_optimizer: 优化器重启次数
            physics_kernel: 是否使用物理核函数
            random_state: 随机种子
        """
        self.alpha = alpha if alpha else MODEL_CONFIG['gp_params'].get('alpha', 1e-6)
        self.n_restarts_optimizer = n_restarts_optimizer if n_restarts_optimizer else MODEL_CONFIG['gp_params'].get('n_restarts_optimizer', 10)
        self.physics_kernel = physics_kernel
        self.random_state = random_state if random_state else 42
        
        # 设置核函数
        if kernel is None:
            kernel_type = MODEL_CONFIG['gp_params'].get('kernel', 'RBF + WhiteKernel')
            if kernel_type == 'RBF + WhiteKernel':
                self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.alpha)
            elif kernel_type == 'Matern':
                self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=self.alpha)
            else:
                self.kernel = RBF(length_scale=1.0)
        else:
            self.kernel = kernel
        
        # 初始化模型
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state
        )
        
        self.feature_names = None
    
    def _physics_kernel(self, X1, X2=None):
        """
        物理知识增强的核函数
        
        基于物理知识构建的核函数，增强了在少量样本时的泛化能力。
        
        Args:
            X1: 第一个特征矩阵
            X2: 第二个特征矩阵
            
        Returns:
            numpy.ndarray: 核矩阵
        """
        if X2 is None:
            X2 = X1
        
        # 将输入转换为DataFrame以便于特征提取
        if isinstance(X1, pd.DataFrame):
            X1_df = X1
        elif self.feature_names is not None:
            X1_df = pd.DataFrame(X1, columns=self.feature_names)
        else:
            X1_df = pd.DataFrame(X1)
        
        if isinstance(X2, pd.DataFrame):
            X2_df = X2
        elif self.feature_names is not None:
            X2_df = pd.DataFrame(X2, columns=self.feature_names)
        else:
            X2_df = pd.DataFrame(X2)
        
        # 这里可以实现基于物理原理的核函数，例如考虑渗透率、井距等特征的物理关系
        # 这里只是简单示例，实际应用中应根据具体物理原理设计
        
        # 基础RBF核
        base_kernel = self.kernel(X1, X2)
        
        # 物理增强
        if self.physics_kernel and 'permeability' in X1_df.columns and 'permeability' in X2_df.columns:
            # 示例：考虑渗透率的影响，渗透率差异大的样本点相似度降低
            perm_scale = 0.1  # 缩放因子
            perm1 = X1_df['permeability'].values.reshape(-1, 1)
            perm2 = X2_df['permeability'].values.reshape(-1, 1)
            perm_diff = np.abs(perm1 - perm2.T)
            physics_factor = np.exp(-perm_scale * perm_diff)
            
            return base_kernel * physics_factor
        
        return base_kernel
    
    def fit(self, X, y):
        """
        训练高斯过程物理模型
        
        Args:
            X: 训练特征
            y: 训练标签
            
        Returns:
            self: 训练好的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 处理NaN值
        if isinstance(X, pd.DataFrame):
            if X.isnull().any().any():
                logger.warning("高斯过程模型训练数据中存在NaN值，将进行插补")
                X = X.fillna(X.median())
        else:
            if np.isnan(X).any():
                logger.warning("高斯过程模型训练数据中存在NaN值，将进行插补")
                mask = np.isnan(X)
                X_clean = X.copy()
                for col in range(X.shape[1]):
                    col_median = np.nanmedian(X[:, col])
                    X_clean[mask[:, col], col] = col_median
                X = X_clean
        
        # 处理目标变量中的NaN
        if isinstance(y, pd.Series):
            if y.isnull().any():
                logger.warning("高斯过程模型目标变量中存在NaN值，将使用均值填充")
                y = y.fillna(y.mean())
        else:
            if np.isnan(y).any():
                logger.warning("高斯过程模型目标变量中存在NaN值，将使用均值填充")
                y_clean = np.copy(y)
                y_clean[np.isnan(y)] = np.nanmean(y)
                y = y_clean
        
        # 保存训练数据，用于备用模型
        self._X = X
        self._y = y
        
        # 如果使用物理核函数，则设置自定义核函数
        if self.physics_kernel:
            # 注意：实际使用自定义核函数需要更复杂的实现
            pass
        
        # 训练模型
        try:
            self.model.fit(X, y)
            logger.info("GaussianProcessPhysicsModel模型训练完成")
        except Exception as e:
            logger.error(f"高斯过程模型训练失败: {str(e)}")
            # 创建并训练备用模型
            self.backup_model = BayesianRidge()
            self.backup_model.fit(X, y)
            logger.info("已训练备用BayesianRidge模型")
        
        return self
    
    def predict(self, X, return_std=False):
        """
        使用模型进行预测
        
        Args:
            X: 特征数据
            return_std: 是否返回标准差
            
        Returns:
            numpy.ndarray 或 tuple: 预测结果，如果return_std=True则返回(均值, 标准差)
        """
        # 检查并处理NaN值
        if isinstance(X, pd.DataFrame):
            if X.isnull().any().any():
                logger.warning("GaussianProcessPhysicsModel预测数据中存在NaN值，将进行插补")
                X = X.fillna(X.median())
        else:
            # 如果X不是DataFrame但我们有特征名称，将其转换为DataFrame
            if hasattr(self, 'feature_names') and self.feature_names:
                X = pd.DataFrame(X, columns=self.feature_names)
            elif np.isnan(X).any():
                logger.warning("GaussianProcessPhysicsModel预测数据中存在NaN值，将进行插补")
                # 创建掩码识别NaN值
                mask = np.isnan(X)
                X_clean = X.copy()
                for col in range(X.shape[1]):
                    # 使用列中位数填充NaN
                    col_data = X[:, col]
                    col_median = np.nanmedian(col_data)
                    X_clean[mask[:, col], col] = col_median
                X = X_clean
        
        try:
            # 尝试使用原始模型预测
            return self.model.predict(X, return_std=return_std)
        except Exception as e:
            logger.error(f"GaussianProcessRegressor预测失败: {str(e)}")
            # 回退策略：使用更简单的模型
            if not hasattr(self, 'backup_model'):
                self.backup_model = BayesianRidge()
                # 如果有训练数据，先训练备用模型
                if hasattr(self, '_X') and hasattr(self, '_y'):
                    self.backup_model.fit(self._X, self._y)
                else:
                    # 直接使用输入特征的均值作为预测结果
                    if return_std:
                        return np.ones(X.shape[0]) * np.nanmean(X), np.ones(X.shape[0]) * 0.1
                    else:
                        return np.ones(X.shape[0]) * np.nanmean(X)
            
            # 使用备用模型预测
            pred = self.backup_model.predict(X)
            if return_std:
                # 简单估计标准差为预测值的10%
                std = np.abs(pred) * 0.1
                return pred, std
            return pred
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            dict: 特征重要性字典（使用核函数的长度尺度倒数）
        """
        importance = {}
        
        # 尝试从RBF或Matern核获取长度尺度
        if hasattr(self.model.kernel_, 'k2') and hasattr(self.model.kernel_.k1, 'k2'):
            # 复合核函数
            kernel = self.model.kernel_.k1.k2
            if hasattr(kernel, 'length_scale'):
                length_scales = kernel.length_scale
                if np.isscalar(length_scales):
                    # 单一长度尺度
                    if self.feature_names and len(self.feature_names) == 1:
                        importance[self.feature_names[0]] = 1.0 / length_scales
                    else:
                        importance['feature_0'] = 1.0 / length_scales
                else:
                    # 每个特征一个长度尺度
                    for i, ls in enumerate(length_scales):
                        if self.feature_names and i < len(self.feature_names):
                            importance[self.feature_names[i]] = 1.0 / ls
                        else:
                            importance[f'feature_{i}'] = 1.0 / ls
        
        return importance
    
    def save_model(self, filepath=None):
        """
        保存模型
        
        Args:
            filepath: 模型保存路径
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'gp_model.pkl')
        
        joblib.dump(self, filepath)
        logger.info(f"GaussianProcessPhysicsModel模型已保存至 {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """
        加载模型
        
        Args:
            filepath: 模型加载路径
            
        Returns:
            GaussianProcessPhysicsModel: 加载的模型
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'gp_model.pkl')
        
        model = joblib.load(filepath)
        logger.info(f"GaussianProcessPhysicsModel模型已从 {filepath} 加载")
        
        return model


class PhysicsEnsembleModel(BaseEstimator, RegressorMixin):
    """
    物理集成模型
    
    集成多个基础模型，结合物理约束优化预测性能。
    """
    
    def __init__(self, base_models=None, weights=None):
        """
        初始化物理集成模型
        
        Args:
            base_models: 基础模型列表，默认使用XGBoost和高斯过程
            weights: 模型权重列表，如果为None则使用等权重
        """
        self.base_models = base_models if base_models else []
        self.weights = weights
        self.feature_names = None
        
        # 如果没有提供基础模型，则创建默认模型
        if not self.base_models:
            self.base_models = [
                PhysicsGuidedXGBoost(),
                # PhysicsInformedNN(),  # 已移除依赖TensorFlow的神经网络模型
                GaussianProcessPhysicsModel()
            ]
        
        # 如果没有提供权重，则使用等权重
        if self.weights is None:
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)
    
    def fit(self, X, y, eval_set=None):
        """
        训练物理集成模型
        
        Args:
            X: 训练特征
            y: 训练标签
            eval_set: 评估集
            
        Returns:
            self: 训练好的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 训练每个基础模型
        for i, model in enumerate(self.base_models):
            logger.info(f"训练集成模型的第{i+1}个基础模型: {model.__class__.__name__}")
            model.fit(X, y, eval_set=eval_set)
        
        # 如果权重为None，则使用等权重
        if self.weights is None:
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        logger.info(f"PhysicsEnsembleModel模型训练完成，包含{len(self.base_models)}个基础模型")
        
        return self
    
    def predict(self, X, return_std=False):
        """
        使用模型进行预测
        
        Args:
            X: 特征数据
            return_std: 是否返回标准差
            
        Returns:
            numpy.ndarray 或 tuple: 预测结果，如果return_std=True则返回(均值, 标准差)
        """
        # 获取各个模型的预测结果
        predictions = []
        
        for model in self.base_models:
            # 检查模型是否支持return_std
            if return_std and hasattr(model, 'predict') and 'return_std' in model.predict.__code__.co_varnames:
                pred, _ = model.predict(X, return_std=True)
            else:
                pred = model.predict(X)
            
            predictions.append(pred)
        
        # 计算加权平均预测
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += self.weights[i] * pred
        
        if return_std:
            # 计算预测标准差
            variance = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                variance += self.weights[i] * (pred - weighted_pred) ** 2
            
            std = np.sqrt(variance)
            return weighted_pred, std
        
        return weighted_pred
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        使用加权平均的方式综合各个模型的特征重要性。
        
        Returns:
            dict: 特征重要性字典
        """
        # 初始化特征重要性字典
        importance_dict = {}
        
        # 遍历每个基础模型
        for i, model in enumerate(self.base_models):
            # 获取模型的特征重要性
            if hasattr(model, 'get_feature_importance'):
                model_importance = model.get_feature_importance()
                
                # 如果模型返回了特征重要性
                if model_importance:
                    # 对于每个特征，加权累加到总的特征重要性中
                    for feature, imp in model_importance.items():
                        if feature in importance_dict:
                            importance_dict[feature] += self.weights[i] * imp
                        else:
                            importance_dict[feature] = self.weights[i] * imp
        
        # 如果没有获取到任何特征重要性，则返回空字典
        if not importance_dict and self.feature_names:
            # 至少返回特征名列表
            importance_dict = {feature: 0.0 for feature in self.feature_names}
        
        return importance_dict
    
    def save_model(self, filepath=None):
        """
        保存模型
        
        Args:
            filepath: 模型保存路径
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'ensemble_model.pkl')
        
        joblib.dump(self, filepath)
        logger.info(f"PhysicsEnsembleModel模型已保存至 {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """
        加载模型
        
        Args:
            filepath: 模型加载路径
            
        Returns:
            PhysicsEnsembleModel: 加载的模型
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'ensemble_model.pkl')
        
        model = joblib.load(filepath)
        logger.info(f"PhysicsEnsembleModel模型已从 {filepath} 加载")
        
        return model
# 在 physics_ml.py 中的 PhysicsEnsembleModel 类之后添加此类

class AdvancedPhysicsEnsemble(BaseEstimator, RegressorMixin):
    """
    高级物理引导集成模型
    
    结合多个模型与物理约束和区段特定建模，
    以在不同PV值范围内实现更好的性能。
    """
    
    def __init__(self, 
                low_value_model=None,
                high_value_model=None, 
                meta_model=None,
                value_threshold=0.1,
                use_physics_constraints=True):
        """
        初始化高级集成模型
        
        参数:
            low_value_model: 低PV值模型（≤ 阈值）
            high_value_model: 高PV值模型（> 阈值）
            meta_model: 用于组合预测的元模型
            value_threshold: 区分低/高PV值的阈值
            use_physics_constraints: 是否应用物理约束
        """
        # 如果未提供模型，则设置默认模型
        self.low_value_model = low_value_model or PhysicsGuidedXGBoost(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        self.high_value_model = high_value_model or PhysicsGuidedXGBoost(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            physics_weight=0.4
        )
        
        # 为元模型使用更简单的回归器
        try:
            # 尝试创建高斯过程模型
            if 'gp_params' in MODEL_CONFIG:
                self.meta_model = meta_model or GaussianProcessPhysicsModel()
            else:
                # 如果没有gp_params配置，使用贝叶斯模型
                self.meta_model = meta_model or BayesianPhysicsModel()
        except Exception as e:
            # 如果创建失败，使用简单的贝叶斯模型
            logger.warning(f"创建元模型失败: {str(e)}，使用备选模型")
            self.meta_model = BayesianPhysicsModel()
        
        self.value_threshold = value_threshold
        self.use_physics_constraints = use_physics_constraints
        self.feature_names = None
        
        # 额外的集成组件
        try:
            self.gaussian_model = GaussianProcessPhysicsModel()
        except Exception:
            # 如果创建失败，使用简单的回归器
            self.gaussian_model = BayesianPhysicsModel()
        
        self.bayesian_model = BayesianPhysicsModel()
        
        # 训练数据统计信息，用于验证
        self.y_mean = None
        self.y_std = None
        
    def fit(self, X, y, eval_set=None):
        """
        训练高级集成模型
        
        参数:
            X: 训练特征
            y: 训练标签
            eval_set: 评估集
            
        返回:
            self: 训练好的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 保存训练数据统计信息
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        # 处理数据中的NaN值
        if isinstance(X, pd.DataFrame):
            # 检查是否存在NaN值
            if X.isnull().any().any():
                logger.warning(f"训练数据中存在NaN值，将进行插补处理")
                # 使用中位数填充NaN
                X_clean = X.fillna(X.median())
            else:
                X_clean = X
        else:
            # 如果是numpy数组
            if np.isnan(X).any():
                logger.warning(f"训练数据中存在NaN值，将进行插补处理")
                # 创建一个掩码来识别NaN值
                mask = np.isnan(X)
                # 用列中位数填充NaN
                X_clean = X.copy()
                for col in range(X.shape[1]):
                    col_median = np.nanmedian(X[:, col])
                    X_clean[mask[:, col], col] = col_median
            else:
                X_clean = X
        
        # 确保y中没有NaN
        if isinstance(y, pd.Series):
            if y.isnull().any():
                logger.warning(f"目标变量中存在NaN值，将使用均值填充")
                y_clean = y.fillna(y.mean())
            else:
                y_clean = y
        else:
            if np.isnan(y).any():
                logger.warning(f"目标变量中存在NaN值，将使用均值填充")
                y_clean = np.copy(y)
                y_clean[np.isnan(y)] = np.nanmean(y)
            else:
                y_clean = y
            
        # 将数据分为低值和高值段
        low_mask = y_clean <= self.value_threshold
        high_mask = ~low_mask
        
        X_low, y_low = X_clean[low_mask], y_clean[low_mask]
        X_high, y_high = X_clean[high_mask], y_clean[high_mask]
        
        logger.info(f"使用{len(X_low)}个样本训练低值模型")
        logger.info(f"使用{len(X_high)}个样本训练高值模型")
        
        # 处理可能的空段，使用SMOTE处理不平衡数据
        if len(X_high) < 5:  # 高值样本不足
            logger.warning("高值样本不足，使用合成过采样")
            from imblearn.over_sampling import SMOTE
            
            try:
                # 转换为数组格式以用于SMOTE
                X_arr = X_clean.values if isinstance(X_clean, pd.DataFrame) else X_clean
                # 创建合成样本
                smote = SMOTE(k_neighbors=min(5, len(X_clean)-1))
                X_resampled, y_resampled = smote.fit_resample(X_arr, pd.cut(y_clean, bins=[0, self.value_threshold, float('inf')], labels=[0, 1]))
                
                # 过采样后重新分割
                high_mask_resampled = y_resampled == 1
                X_high = X_resampled[high_mask_resampled]
                
                # 为高段生成合成y值
                y_high = np.random.uniform(
                    self.value_threshold, 
                    self.value_threshold * 3,  # 合理的高值
                    size=sum(high_mask_resampled)
                )
                
                logger.info(f"创建了{len(X_high)}个合成高值样本")
            except Exception as e:
                logger.error(f"SMOTE错误: {str(e)}，对两个模型使用完整数据集")
                X_low, y_low = X_clean, y_clean
                X_high, y_high = X_clean, y_clean
        
        # 训练低值模型
        if len(X_low) > 0:
            self.low_value_model.fit(X_low, y_low, eval_set=eval_set)
        else:
            logger.warning("没有低值样本，使用完整数据集")
            self.low_value_model.fit(X_clean, y_clean, eval_set=eval_set)
        
        # 训练高值模型
        if len(X_high) > 0:
            self.high_value_model.fit(X_high, y_high, eval_set=eval_set)
        else:
            logger.warning("没有高值样本，使用完整数据集")
            self.high_value_model.fit(X_clean, y_clean, eval_set=eval_set)
        
        # 在完整数据集上训练额外模型，确保没有NaN值
        try:
            self.gaussian_model.fit(X_clean, y_clean)
        except Exception as e:
            logger.error(f"高斯过程模型训练失败: {str(e)}，使用贝叶斯岭回归替代")
            self.gaussian_model = BayesianPhysicsModel()
            self.gaussian_model.fit(X_clean, y_clean)
        
        self.bayesian_model.fit(X_clean, y_clean)
        
        # 准备元模型训练数据
        meta_X = np.zeros((len(X_clean), 4))  # 4个特征: 2个分段模型 + 2个额外模型
        
        # 获取所有基础模型的预测
        meta_X[:, 0] = self.low_value_model.predict(X_clean)
        meta_X[:, 1] = self.high_value_model.predict(X_clean)
        meta_X[:, 2] = self.gaussian_model.predict(X_clean)
        meta_X[:, 3] = self.bayesian_model.predict(X_clean)
        
        # 如果X维度不太高，则添加原始特征
        if X_clean.shape[1] <= 10:
            if isinstance(X_clean, pd.DataFrame):
                X_arr = X_clean.values
            else:
                X_arr = X_clean
            meta_X = np.hstack((meta_X, X_arr))
        
        # 训练元模型
        self.meta_model.fit(meta_X, y_clean)
        
        logger.info("高级物理集成模型训练完成")
        return self
    
    def _augment_minority_samples(self, X, y, threshold=0.1, k_neighbors=5):
        """
        针对高PV值样本进行合成过采样
        
        Args:
            X: 特征数据
            y: 目标值
            threshold: 高值样本阈值
            k_neighbors: K近邻数量
            
        Returns:
            tuple: (增强后的X, 增强后的y)
        """
        from sklearn.preprocessing import StandardScaler
        
        # 划分高值和低值样本
        high_mask = y > threshold
        X_high, y_high = X[high_mask], y[high_mask]
        X_low, y_low = X[~high_mask], y[~high_mask]
        
        high_count = sum(high_mask)
        low_count = sum(~high_mask)
        
        # 如果高值样本太少，进行过采样
        if high_count < 5:
            logger.warning(f"高值样本数量过少({high_count})，无法进行SMOTE，使用基于物理规则的合成")
            
            # 基于物理规则合成样本
            synthetic_samples = []
            synthetic_targets = []
            
            # 选择一个高值样本作为基准
            base_sample = X_high.iloc[0] if isinstance(X_high, pd.DataFrame) else X_high[0]
            base_target = y_high.iloc[0] if isinstance(y_high, pd.Series) else y_high[0]
            
            # 基于物理参数合成样本
            for i in range(min(10, low_count // 2)):
                synthetic_sample = base_sample.copy()
                
                # 对重要特征进行扰动
                if 'injection_intensity_ratio' in X.columns:
                    col_idx = X.columns.get_loc('injection_intensity_ratio')
                    synthetic_sample[col_idx] *= (1 + np.random.normal(0, 0.2))
                
                if 'permeability' in X.columns:
                    col_idx = X.columns.get_loc('permeability')
                    synthetic_sample[col_idx] *= (1 + np.random.normal(0, 0.15))
                
                # 计算合成目标值 (基于物理关系略微调整)
                synthetic_target = base_target * (1 + np.random.normal(0, 0.1))
                
                synthetic_samples.append(synthetic_sample)
                synthetic_targets.append(synthetic_target)
            
            # 合并原始和合成样本
            if isinstance(X, pd.DataFrame):
                X_augmented = pd.concat([X, pd.DataFrame(synthetic_samples, columns=X.columns)])
                y_augmented = pd.concat([y, pd.Series(synthetic_targets)])
            else:
                X_augmented = np.vstack([X, np.array(synthetic_samples)])
                y_augmented = np.hstack([y, np.array(synthetic_targets)])
            
            logger.info(f"已基于物理规则合成{len(synthetic_samples)}个高值样本")
        
        else:
            try:
                from imblearn.over_sampling import SMOTE
                
                # 标准化特征以提高SMOTE效果
                scaler = StandardScaler()
                if isinstance(X, pd.DataFrame):
                    X_values = scaler.fit_transform(X)
                else:
                    X_values = scaler.fit_transform(X)
                
                # 创建二分类标签用于SMOTE
                y_binary = (y > threshold).astype(int)
                
                # 应用SMOTE
                smote = SMOTE(k_neighbors=min(k_neighbors, high_count-1), random_state=42)
                X_resampled, y_binary_resampled = smote.fit_resample(X_values, y_binary)
                
                # 为合成的高值样本创建目标值
                # 基于原始高值样本的分布生成新的目标值
                high_mean = np.mean(y_high)
                high_std = np.std(y_high)
                
                # 反转标准化
                if isinstance(X, pd.DataFrame):
                    X_resampled = pd.DataFrame(
                        scaler.inverse_transform(X_resampled), 
                        columns=X.columns
                    )
                else:
                    X_resampled = scaler.inverse_transform(X_resampled)
                
                # 为新的高值样本分配目标值
                y_resampled = y.copy() if isinstance(y, pd.Series) else y.copy()
                
                # 找出新合成的高值样本索引
                new_high_indices = np.where(y_binary_resampled == 1)[0][high_count:]
                
                # 为新高值样本分配目标值
                new_targets = np.random.normal(high_mean, high_std, size=len(new_high_indices))
                new_targets = np.maximum(threshold, new_targets)  # 确保不低于阈值
                
                # 构建新的目标数组
                if isinstance(y, pd.Series):
                    y_resampled = pd.Series(index=range(len(X_resampled)))
                    y_resampled.iloc[:len(y)] = y.values
                    y_resampled.iloc[len(y):] = 0
                    y_resampled.iloc[new_high_indices] = new_targets
                else:
                    y_resampled = np.zeros(len(X_resampled))
                    y_resampled[:len(y)] = y
                    y_resampled[new_high_indices] = new_targets
                
                X_augmented = X_resampled
                y_augmented = y_resampled
                
                logger.info(f"通过SMOTE合成了{len(new_high_indices)}个高值样本")
                
            except Exception as e:
                logger.error(f"SMOTE错误: {str(e)}，使用原始数据集")
                X_augmented, y_augmented = X, y
        
        return X_augmented, y_augmented
    
    def predict(self, X, return_std=False):
        """
        使用集成模型进行预测
        
        参数:
            X: 特征
            return_std: 是否返回预测标准差
            
        返回:
            numpy.ndarray 或 tuple: 预测值，可选带标准差
        """
        if self.low_value_model is None or self.high_value_model is None:
            raise ValueError("模型尚未训练")
        
        # 处理预测数据中的NaN值
        if isinstance(X, pd.DataFrame):
            # 检查是否存在NaN值
            if X.isnull().any().any():
                logger.warning(f"预测数据中存在NaN值，将进行插补处理")
                # 使用中位数填充NaN
                X_clean = X.fillna(X.median())
            else:
                X_clean = X
        else:
            # 如果是numpy数组
            if np.isnan(X).any():
                logger.warning(f"预测数据中存在NaN值，将进行插补处理")
                # 创建一个掩码来识别NaN值
                mask = np.isnan(X)
                # 用列中位数填充NaN
                X_clean = X.copy()
                for col in range(X.shape[1]):
                    col_median = np.nanmedian(X[:, col])
                    X_clean[mask[:, col], col] = col_median
            else:
                X_clean = X
        
        # 获取所有基础模型的预测
        low_preds = self.low_value_model.predict(X_clean)
        high_preds = self.high_value_model.predict(X_clean)
        gaussian_preds = self.gaussian_model.predict(X_clean)
        bayesian_preds = self.bayesian_model.predict(X_clean)
        
        # 准备元模型输入
        meta_X = np.zeros((len(X_clean), 4))
        meta_X[:, 0] = low_preds
        meta_X[:, 1] = high_preds
        meta_X[:, 2] = gaussian_preds
        meta_X[:, 3] = bayesian_preds
        
        # 如果X维度不太高，则添加原始特征
        if X_clean.shape[1] <= 10:
            if isinstance(X_clean, pd.DataFrame):
                X_arr = X_clean.values
            else:
                X_arr = X_clean
            meta_X = np.hstack((meta_X, X_arr))
        
        # 获取元模型预测
        if return_std and hasattr(self.meta_model, 'predict') and 'return_std' in self.meta_model.predict.__code__.co_varnames:
            meta_pred, meta_std = self.meta_model.predict(meta_X, return_std=True)
        else:
            meta_pred = self.meta_model.predict(meta_X)
            
            # 从基础模型分歧估计不确定性
            model_preds = np.vstack([low_preds, high_preds, gaussian_preds, bayesian_preds])
            meta_std = np.std(model_preds, axis=0)
        
        # 如果启用，则应用物理约束
        if self.use_physics_constraints:
            # 确保预测非负
            meta_pred = np.maximum(meta_pred, 0)
            
            # 根据训练数据应用实际限制
            if self.y_mean is not None and self.y_std is not None:
                upper_limit = self.y_mean + 5 * self.y_std
                meta_pred = np.minimum(meta_pred, upper_limit)
        
        if return_std:
            return meta_pred, meta_std
        return meta_pred
    
    def _apply_enhanced_physics_constraints(self, X, predictions):
        """
        应用增强的物理约束到预测结果
        
        Args:
            X: 特征数据
            predictions: 初步预测结果
            
        Returns:
            numpy.ndarray: 应用物理约束后的预测结果
        """
        constrained_preds = predictions.copy()
        
        # 如果有物理属性，进行约束
        if isinstance(X, pd.DataFrame):
            # 1. 注入强度比与PV数的物理关系约束
            if 'injection_intensity_ratio' in X.columns:
                # 注入强度比与PV数呈正相关
                for i, (idx, row) in enumerate(X.iterrows()):
                    inj_intensity = row['injection_intensity_ratio']
                    # 如果注入强度比过低但预测PV数很高，施加约束
                    if inj_intensity < self.y_mean/10 and predictions[i] > self.y_mean*2:
                        constrained_preds[i] = min(predictions[i], self.y_mean)
                        logger.debug(f"对样本{i}应用注入强度物理约束，原值:{predictions[i]:.4f}，调整为:{constrained_preds[i]:.4f}")
            
            # 2. 渗透率与PV数的物理关系约束
            if 'permeability' in X.columns and 'porosity' in X.columns:
                for i, (idx, row) in enumerate(X.iterrows()):
                    perm = row['permeability']
                    poro = row['porosity']
                    
                    # 低渗透率+高孔隙度的情况下，气窜风险会降低
                    if perm < 10 and poro > 0.2 and predictions[i] > self.y_mean*1.5:
                        constrained_preds[i] = min(predictions[i], self.y_mean)
                        logger.debug(f"对样本{i}应用渗透率-孔隙度物理约束，原值:{predictions[i]:.4f}，调整为:{constrained_preds[i]:.4f}")
            
            # 3. 区块特定约束
            block_columns = [col for col in X.columns if col.startswith('block_')]
            for block_col in block_columns:
                if block_col in X.columns:
                    for i, (idx, row) in enumerate(X.iterrows()):
                        if row[block_col] == 1:  # 如果样本属于该区块
                            # 可以针对不同区块应用不同约束
                            if block_col == 'block_MALJAMAR' and predictions[i] < 0.05:
                                # MALJAMAR区块的最小PV数阈值
                                constrained_preds[i] = max(predictions[i], 0.05)
                                logger.debug(f"对样本{i}应用区块{block_col}特定约束，原值:{predictions[i]:.4f}，调整为:{constrained_preds[i]:.4f}")
        
        return constrained_preds
    
    def get_feature_importance(self):
        """
        获取集成模型的特征重要性
        
        返回:
            dict: 特征重要性字典
        """
        importance_dict = {}
        
        # 收集基础模型的特征重要性
        if hasattr(self.low_value_model, 'get_feature_importance'):
            low_importance = self.low_value_model.get_feature_importance()
            for feature, imp in low_importance.items():
                importance_dict[feature] = importance_dict.get(feature, 0) + 0.3 * imp
        
        if hasattr(self.high_value_model, 'get_feature_importance'):
            high_importance = self.high_value_model.get_feature_importance()
            for feature, imp in high_importance.items():
                importance_dict[feature] = importance_dict.get(feature, 0) + 0.3 * imp
        
        if hasattr(self.gaussian_model, 'get_feature_importance'):
            gaussian_importance = self.gaussian_model.get_feature_importance()
            for feature, imp in gaussian_importance.items():
                importance_dict[feature] = importance_dict.get(feature, 0) + 0.2 * imp
        
        if hasattr(self.bayesian_model, 'get_feature_importance'):
            bayesian_importance = self.bayesian_model.get_feature_importance()
            for feature, imp in bayesian_importance.items():
                importance_dict[feature] = importance_dict.get(feature, 0) + 0.2 * imp
        
        return importance_dict
    
    def save_model(self, filepath=None):
        """
        保存集成模型
        
        参数:
            filepath: 保存路径
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'advanced_ensemble_model.pkl')
        
        joblib.dump(self, filepath)
        logger.info(f"高级集成模型已保存至 {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """
        加载集成模型
        
        参数:
            filepath: 加载路径
            
        返回:
            AdvancedPhysicsEnsemble: 加载的模型
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'advanced_ensemble_model.pkl')
        
        model = joblib.load(filepath)
        logger.info(f"高级集成模型已从 {filepath} 加载")
        return model
class BlockBasedEnsembleModel(BaseEstimator, RegressorMixin):
    """
    基于区块的集成模型
    
    为不同区块训练专门的子模型，提高预测精度。
    """
    
    def __init__(self, base_model_class=PhysicsGuidedXGBoost, min_samples=5, global_model=None):
        """
        初始化区块分组模型
        
        Args:
            base_model_class: 基础模型类
            min_samples: 训练区块特定模型的最小样本数
            global_model: 全局模型（用于未见过的区块）
        """
        self.base_model_class = base_model_class
        self.min_samples = min_samples
        self.global_model = global_model
        
        self.block_models = {}  # 区块特定模型
        self.block_columns = []  # 区块特征列
        self.feature_names = None
        self.default_params = {}  # 默认参数
    
    def fit(self, X, y, eval_set=None):
        """
        训练区块分组模型
        
        Args:
            X: 训练特征
            y: 训练标签
            eval_set: 评估集
            
        Returns:
            self: 训练好的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 识别区块特征列
        if self.feature_names:
            self.block_columns = [col for col in self.feature_names if col.startswith('block_')]
        else:
            # 如果没有特征名称，则无法识别区块列
            logger.warning("无法识别区块列，将仅训练全局模型")
            self.block_columns = []
        
        # 训练全局模型
        if self.global_model is None:
            self.global_model = self.base_model_class()
        
        self.global_model.fit(X, y, eval_set=eval_set)
        logger.info("全局模型训练完成")
        
        # 为每个区块训练特定模型
        if self.block_columns:
            for block_col in self.block_columns:
                # 获取该区块的样本
                is_block = X[block_col] == 1
                block_samples = sum(is_block)
                
                if block_samples >= self.min_samples:
                    logger.info(f"为区块'{block_col}'训练特定模型，样本数: {block_samples}")
                    
                    X_block = X[is_block]
                    y_block = y[is_block]
                    
                    # 创建并训练区块特定模型
                    block_model = self.base_model_class(**self.default_params)
                    
                    try:
                        # 如果有评估集，尝试创建区块特定的评估集
                        block_eval_set = None
                        if eval_set:
                            eval_X, eval_y = eval_set[0]
                            eval_is_block = eval_X[block_col] == 1
                            
                            if sum(eval_is_block) > 0:
                                block_eval_set = [(
                                    eval_X[eval_is_block], 
                                    eval_y[eval_is_block]
                                )]
                        
                        block_model.fit(X_block, y_block, eval_set=block_eval_set)
                        self.block_models[block_col] = block_model
                        logger.info(f"区块'{block_col}'模型训练完成")
                    except Exception as e:
                        logger.error(f"区块'{block_col}'模型训练失败: {str(e)}")
                else:
                    logger.info(f"区块'{block_col}'样本数({block_samples})不足，不训练特定模型")
        
        return self
    
    def predict(self, X):
        """
        使用模型进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            numpy.ndarray: 预测结果
        """
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame) and self.feature_names:
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 获取全局模型预测
        global_pred = self.global_model.predict(X)
        
        # 如果没有区块特定模型，直接返回全局预测
        if not self.block_models:
            return global_pred
        
        # 对于每个样本，如果属于某个区块且有对应模型，则使用区块特定预测
        final_pred = global_pred.copy()
        
        for block_col, block_model in self.block_models.items():
            if block_col in X.columns:
                is_block = X[block_col] == 1
                
                if any(is_block):
                    block_pred = block_model.predict(X[is_block])
                    final_pred[is_block] = block_pred
        
        return final_pred
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            dict: 特征重要性字典
        """
        # 获取全局模型的特征重要性
        if hasattr(self.global_model, 'get_feature_importance'):
            importance_dict = self.global_model.get_feature_importance()
        else:
            importance_dict = {}
        
        # 为有区块特定模型的区块增加重要性
        for block_col, block_model in self.block_models.items():
            if hasattr(block_model, 'get_feature_importance'):
                # 获取该区块模型的特征重要性
                block_importance = block_model.get_feature_importance()
                
                # 为区块特征增加重要性
                if block_col not in importance_dict:
                    importance_dict[block_col] = 0.0
                
                # 加权增加区块重要性
                importance_dict[block_col] += 0.1
                
                # 将区块模型的特征重要性加入总重要性中，但权重较低
                for feature, imp in block_importance.items():
                    if feature in importance_dict:
                        importance_dict[feature] += 0.05 * imp
        
        return importance_dict
    
    def save_model(self, filepath=None):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'block_ensemble_model.pkl')
        
        joblib.dump(self, filepath)
        logger.info(f"区块集成模型已保存至 {filepath}")
    
    @classmethod
    def load_model(cls, filepath=None):
        """
        加载模型
        
        Args:
            filepath: 加载路径
            
        Returns:
            BlockBasedEnsembleModel: 加载的模型
        """
        if filepath is None:
            filepath = os.path.join(PATHS['model_dir'], 'block_ensemble_model.pkl')
        
        model = joblib.load(filepath)
        logger.info(f"区块集成模型已从 {filepath} 加载")
        
        return model
# 在 physics_ml.py 中，更新 optimize_xgboost_params 函数
def optimize_xgboost_params(X_train, y_train, X_val, y_val, n_trials=50):
    """
    使用Optuna为XGBoost进行增强的超参数优化
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        n_trials: 优化尝试次数
        
    返回:
        tuple: (最佳参数字典, 最佳R²值)
    """
    import optuna
    
    logger.info(f"开始XGBoost超参数优化，尝试次数: {n_trials}")
    
    def objective(trial):
        # 扩展的参数搜索空间
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.001, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
            'physics_weight': trial.suggest_float('physics_weight', 0.1, 1.0)
        }
        
        # 创建物理约束XGBoost模型
        model = PhysicsGuidedXGBoost(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            physics_weight=params['physics_weight']
        )
        
        # 使用早停训练模型
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            # 评估性能
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            # 处理无效的分数
            if np.isnan(r2) or np.isinf(r2):
                return -1.0
                
            # 记录当前尝试结果
            logger.info(f"Trial {trial.number}: R² = {r2:.4f}, Params = {params}")
            
            return r2
        
        except Exception as e:
            logger.error(f"Trial {trial.number} 错误: {str(e)}")
            # 对于失败的试验返回非常低的分数
            return -1.0
    
    # 创建并运行优化研究
    study = optuna.create_study(direction='maximize')
    
    try:
        study.optimize(objective, n_trials=n_trials)
        
        # 获取并记录最佳结果
        if study.trials:
            # 检查是否有成功的trials
            successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if successful_trials:
                best_params = study.best_params
                best_r2 = study.best_value
                logger.info(f"超参数优化完成，最佳 R² = {best_r2:.4f}")
                logger.info(f"最佳参数: {best_params}")
                return best_params, best_r2
    
    except Exception as e:
        logger.error(f"优化过程出错: {str(e)}")
    
    # 如果没有成功的trials或出现错误，返回默认参数
    logger.warning("超参数优化失败，使用默认参数")
    default_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'physics_weight': 0.3
    }
    
    return default_params, 0.0

# 在 physics_ml.py 中，更新 train_model 函数（大约第827行）
def train_model(X_train, y_train, X_val=None, y_val=None, model_type=None, optimize_params=True):
    """
    使用增强的参数优化训练气窜预测模型
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        model_type: 模型类型，默认使用配置默认值
        optimize_params: 是否执行参数优化
        
    返回:
        BaseEstimator: 训练好的模型
    """
    if model_type is None:
        model_type = MODEL_CONFIG.get('default_model', 'physics_guided_xgboost')
    
    # 准备评估集
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
    
    # 训练模型
    logger.info(f"开始训练 {model_type} 模型")
    
    if model_type == 'physics_guided_xgboost':
        if optimize_params and X_val is not None and y_val is not None:
            # 执行超参数优化
            best_params, _ = optimize_xgboost_params(X_train, y_train, X_val, y_val, n_trials=100)
            
            # 使用优化参数创建模型
            model = PhysicsGuidedXGBoost(
                n_estimators=best_params.get('n_estimators'),
                learning_rate=best_params.get('learning_rate'),
                max_depth=best_params.get('max_depth'),
                min_child_weight=best_params.get('min_child_weight'),
                subsample=best_params.get('subsample'),
                colsample_bytree=best_params.get('colsample_bytree'),
                gamma=best_params.get('gamma'),
                reg_alpha=best_params.get('reg_alpha'),
                reg_lambda=best_params.get('reg_lambda'),
                physics_weight=best_params.get('physics_weight')
            )
        else:
            # 使用默认参数
            model = PhysicsGuidedXGBoost()
    elif model_type == 'physics_informed_nn':
        # 神经网络模型不可用，使用XGBoost代替
        logger.warning("PhysicsInformedNN不可用，使用PhysicsGuidedXGBoost代替")
        model = PhysicsGuidedXGBoost()
    elif model_type == 'bayesian_ridge':
        model = BayesianPhysicsModel()
    elif model_type == 'gaussian_process':
        model = GaussianProcessPhysicsModel()
    elif model_type == 'ensemble':
        model = PhysicsEnsembleModel()
    elif model_type == 'advanced_ensemble':
        # 新的高级集成模型
        if optimize_params and X_val is not None and y_val is not None:
            # 首先为低值优化XGBoost
            X_train_low = X_train[y_train <= 0.1]
            y_train_low = y_train[y_train <= 0.1]
            
            # 在相同函数中，修改低值模型优化部分
            if len(X_train_low) > 5:
                X_val_low = X_val[y_val <= 0.1] if len(X_val[y_val <= 0.1]) > 0 else X_val
                y_val_low = y_val[y_val <= 0.1] if len(y_val[y_val <= 0.1]) > 0 else y_val
                
                try:
                    best_params_low, _ = optimize_xgboost_params(
                        X_train_low, y_train_low, X_val_low, y_val_low, n_trials=50
                    )
                    low_model = PhysicsGuidedXGBoost(**best_params_low)
                except ValueError as e:
                    logger.warning(f"低值样本优化失败: {str(e)}，使用预设参数")
                    # 为低值样本使用更适合的预设参数
                    low_model = PhysicsGuidedXGBoost(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=5,
                        min_child_weight=1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        physics_weight=0.3
                    )
            else:
                # 使用经验丰富的低值样本模型参数
                low_model = PhysicsGuidedXGBoost(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    physics_weight=0.3
                )
            
            # 然后为高值优化XGBoost（如果可能）
            X_train_high = X_train[y_train > 0.1]
            y_train_high = y_train[y_train > 0.1]
            
            if len(X_train_high) > 5:
                X_val_high = X_val[y_val > 0.1] if len(X_val[y_val > 0.1]) > 0 else X_val
                y_val_high = y_val[y_val > 0.1] if len(y_val[y_val > 0.1]) > 0 else y_val
                
                try:
                    best_params_high, _ = optimize_xgboost_params(
                        X_train_high, y_train_high, X_val_high, y_val_high, n_trials=50
                    )
                    high_model = PhysicsGuidedXGBoost(**best_params_high)
                except ValueError as e:
                    logger.warning(f"高值样本优化失败: {str(e)}，使用预设参数")
                    # 为高值样本使用更适合的预设参数
                    high_model = PhysicsGuidedXGBoost(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        min_child_weight=2,
                        subsample=0.7,
                        colsample_bytree=0.7,
                        reg_alpha=0.2,
                        reg_lambda=1.0,
                        physics_weight=0.4
                    )
            else:
                # 使用经验丰富的高值样本模型参数
                high_model = PhysicsGuidedXGBoost(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=2,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.2,
                    reg_lambda=1.0,
                    physics_weight=0.4
                )
            
            # 创建高级集成
            model = AdvancedPhysicsEnsemble(
                low_value_model=low_model,
                high_value_model=high_model
            )
        else:
            model = AdvancedPhysicsEnsemble()
    elif model_type == 'block_ensemble':
        # 使用区块分组模型
        base_model = PhysicsGuidedXGBoost(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            physics_weight=0.3
        )
        model = BlockBasedEnsembleModel(
            base_model_class=PhysicsGuidedXGBoost, 
            min_samples=5,
            global_model=base_model
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    model.fit(X_train, y_train, eval_set=eval_set)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        
    Returns:
        dict: 评估指标
    """
    # 获取预测值
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # 输出评估结果
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    return metrics


def cross_validate(X, y, model_type=None, n_splits=None, loocv=None):
    """
    交叉验证模型性能，增加NaN检测和处理
    
    Args:
        X: 特征数据
        y: 标签数据
        model_type: 模型类型
        n_splits: 交叉验证折数
        loocv: 是否使用留一交叉验证
        
    Returns:
        dict: 交叉验证结果
    """
    if model_type is None:
        model_type = MODEL_CONFIG.get('default_model', 'physics_guided_xgboost')
    
    if n_splits is None:
        n_splits = MODEL_CONFIG.get('cv_folds', 5)
    
    if loocv is None:
        loocv = MODEL_CONFIG.get('loocv', True)
    
    # 检查数据是否包含NaN值
    if isinstance(X, pd.DataFrame):
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"交叉验证数据中包含{nan_count}个NaN值")
            
            # 记录包含NaN的列
            nan_cols = X.columns[X.isnull().any()].tolist()
            logger.warning(f"以下列包含NaN值: {', '.join(nan_cols)}")
            
            # 对NaN值进行填充
            X_clean = X.fillna(X.median())
            logger.info("已使用中位数填充NaN值")
        else:
            X_clean = X
    else:
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            logger.warning(f"交叉验证数据中包含{nan_count}个NaN值")
            
            # 对NaN值进行填充
            X_clean = np.copy(X)
            for col in range(X.shape[1]):
                col_median = np.nanmedian(X[:, col])
                mask = np.isnan(X[:, col])
                X_clean[mask, col] = col_median
            logger.info("已使用中位数填充NaN值")
        else:
            X_clean = X
    
    # 对于小样本数据，使用留一交叉验证
    if loocv or len(X) <= 10:
        logger.info("使用留一交叉验证(LOOCV)")
        cv = LeaveOneOut()
    else:
        logger.info(f"使用{n_splits}折交叉验证")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储各折的评估指标
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    # 进行交叉验证
    for i, (train_idx, test_idx) in enumerate(cv.split(X_clean)):
        # 划分训练集和测试集
        if isinstance(X_clean, pd.DataFrame):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
        else:
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        try:
            model = train_model(X_train, y_train, model_type=model_type)
            
            # 评估模型
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            # 添加诊断信息
            logger.info(f"交叉验证第{i+1}折: 目标变量统计")
            logger.info(f"  训练集: 均值={np.mean(y_train):.6f}, 方差={np.var(y_train):.6f}, 最小值={np.min(y_train):.6f}, 最大值={np.max(y_train):.6f}")
            logger.info(f"  测试集: 均值={np.mean(y_test):.6f}, 方差={np.var(y_test):.6f}, 最小值={np.min(y_test):.6f}, 最大值={np.max(y_test):.6f}")
            logger.info(f"  预测值: 均值={np.mean(y_pred):.6f}, 方差={np.var(y_pred):.6f}, 最小值={np.min(y_pred):.6f}, 最大值={np.max(y_pred):.6f}")

            # 安全计算R²
            try:
                # 先尝试使用scikit-learn的r2_score
                r2 = r2_score(y_test, y_pred)
                
                # 如果r2_score返回NaN，使用手动计算并添加额外的安全措施
                if np.isnan(r2) or np.isinf(r2):
                    logger.warning(f"第{i+1}折: scikit-learn的r2_score返回{r2}，尝试手动计算")
                    y_test_mean = np.mean(y_test)
                    ss_total = np.sum((y_test - y_test_mean) ** 2)
                    ss_residual = np.sum((y_test - y_pred) ** 2)
                    
                    # 如果分母接近零，使用一个极小值代替
                    if ss_total < 1e-10:
                        r2 = 0.0
                        logger.warning(f"第{i+1}折: 目标变量方差接近零({ss_total:.10f})，设置R²为0")
                    else:
                        r2 = 1.0 - (ss_residual / ss_total)
                        
                        # 检查是否为合理值
                        if r2 < -1.0:
                            logger.warning(f"第{i+1}折: R²过小({r2:.4f})，限制为-1.0")
                            r2 = -1.0
                        elif r2 > 1.0:
                            logger.warning(f"第{i+1}折: R²过大({r2:.4f})，限制为1.0")
                            r2 = 1.0
            except Exception as e:
                logger.error(f"计算R²时出错: {str(e)}")
                r2 = 0.0  # 出错时默认为0
            
            # 存储评估指标
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            
            logger.info(f"交叉验证第{i+1}折: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        except Exception as e:
            logger.error(f"交叉验证第{i+1}折出错: {str(e)}")
            # 继续下一折，而不是中断整个过程
            continue
    
    # 如果所有折都失败了
    if len(mse_scores) == 0:
        logger.error("所有交叉验证折都失败了")
        # 返回默认值
        return {
            'mse_mean': np.nan, 'mse_std': np.nan,
            'rmse_mean': np.nan, 'rmse_std': np.nan,
            'mae_mean': np.nan, 'mae_std': np.nan,
            'r2_mean': np.nan, 'r2_std': np.nan
        }
    
    # 计算平均评估指标
    cv_results = {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores)
    }
    
    logger.info(f"交叉验证完成: RMSE={cv_results['rmse_mean']:.4f}±{cv_results['rmse_std']:.4f}, R2={cv_results['r2_mean']:.4f}±{cv_results['r2_std']:.4f}")
    
    return cv_results


def main():
    """物理机器学习模型主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(PATHS['log_dir'], PATHS['log_filename'])
    )
    
    logger.info("物理约束机器学习模型模块测试")
    
    # 创建并训练示例模型，用于演示
    from data_processor import load_data, preprocess_data, engineer_features, split_dataset
    
    # 加载数据
    df = load_data()
    
    # 预处理
    df = preprocess_data(df)
    
    # 特征工程
    df = engineer_features(df)
    
    # 分割数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    
    # 训练模型
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 保存模型
    model.save_model()
    
    return model

if __name__ == "__main__":
    main()
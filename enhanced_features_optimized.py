#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统特征优化模块

本模块实现了针对小样本量数据的特征工程方法，
包括物理约束特征、特征选择和特征优化。
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging

# 导入配置
from config import FEATURE_CONFIG, PHYSICS_CONFIG

# 设置日志
logger = logging.getLogger(__name__)

def create_physics_informed_features(df):
    """
    创建基于物理原理的特征
    
    Args:
        df: 输入DataFrame
        
    Returns:
        DataFrame: 添加物理特征后的DataFrame
    """
    # 创建DataFrame的副本，避免修改原始数据
    df_physics = df.copy()
    
    # 检查必要的特征是否存在
    required_features = ['permeability', 'oil_viscosity', 'well_spacing', 
                         'effective_thickness', 'formation_pressure', 
                         'porosity', 'temperature', 'oil_density']
    
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        logger.warning(f"缺少创建物理特征所需的列: {missing_features}")
        # 如果缺少必要特征，尝试创建替代特征
        for feature in missing_features:
            if feature == 'permeability':
                df_physics['permeability'] = 50.0  # 使用典型值
            elif feature == 'oil_viscosity':
                df_physics['oil_viscosity'] = 5.0  # 使用典型值
            elif feature == 'oil_density':
                df_physics['oil_density'] = 850.0  # 使用典型值
            elif feature == 'porosity':
                df_physics['porosity'] = 0.2  # 使用典型值
            elif feature == 'temperature':
                df_physics['temperature'] = 60.0  # 使用典型值
    
    # 1. 迁移性比(Mobility Ratio)
    # 假设CO2粘度为0.05 mPa·s
    co2_viscosity = PHYSICS_CONFIG.get('co2_viscosity', 0.05)
    df_physics['mobility_ratio'] = df_physics['oil_viscosity'] / co2_viscosity
    
    # 2. 重力数(Gravity Number)
    # 假设CO2密度为800 kg/m³
    co2_density = PHYSICS_CONFIG.get('co2_density', 800)
    gravity = PHYSICS_CONFIG.get('gravity', 9.8)
    
    if 'oil_density' in df_physics.columns:
        df_physics['gravity_number'] = (df_physics['oil_density'] - co2_density) * gravity * \
                                      df_physics['permeability'] * df_physics['effective_thickness'] / \
                                      df_physics['oil_viscosity']
    
    # 3. 指进系数(Fingering Index)
    # 假设不可动水饱和度Swi=0.2，剩余油饱和度Sor=0.3
    swi = PHYSICS_CONFIG.get('swi', 0.2)
    sor = PHYSICS_CONFIG.get('sor', 0.3)
    
    df_physics['fingering_index'] = df_physics['mobility_ratio'] * (1 - swi - sor)
    
    # 4. 渗透率-孔隙度比(Permeability-Porosity Ratio)
    if 'porosity' in df_physics.columns:
        df_physics['perm_porosity_ratio'] = df_physics['permeability'] / df_physics['porosity']
    
    # 5. 井距-厚度比(Spacing-Thickness Ratio)
    df_physics['spacing_thickness_ratio'] = df_physics['well_spacing'] / df_physics['effective_thickness']
    
    # 6. 储层能量(Reservoir Energy)
    if 'temperature' in df_physics.columns:
        df_physics['reservoir_energy'] = df_physics['formation_pressure'] * df_physics['temperature'] / 100
    
    # 7. 相迁移性指数(Phase Mobility Index)
    df_physics['phase_mobility_index'] = (df_physics['permeability'] / df_physics['oil_viscosity']) * \
                                        (co2_viscosity / df_physics['oil_viscosity'])
    
    # 8. 驱替效率(Displacement Efficiency)
    # 假设初始油饱和度Soi=0.8
    soi = PHYSICS_CONFIG.get('soi', 0.8)
    
    df_physics['displacement_efficiency'] = (soi - sor) / soi
    
    # 9. 压力-粘度比(Pressure-Viscosity Ratio)
    df_physics['pressure_viscosity_ratio'] = df_physics['formation_pressure'] / df_physics['oil_viscosity']
    
    # 10. 流动能力指数(Flow Capacity Index)
    df_physics['flow_capacity_index'] = df_physics['permeability'] * df_physics['effective_thickness'] / \
                                       (df_physics['oil_viscosity'] * df_physics['well_spacing'])
    
    logger.info(f"创建了{len(df_physics.columns) - len(df.columns)}个物理约束特征")
    
    return df_physics

def select_optimal_features_limited(df, target_column, max_features=10, method='hybrid'):
    """
    选择最优特征子集，限制最大特征数量
    
    Args:
        df: 特征DataFrame
        target_column: 目标变量列名
        max_features: 最大特征数量
        method: 特征选择方法，可选'mutual_info', 'random_forest', 'lasso', 'hybrid'
        
    Returns:
        list: 选择的特征列表
    """
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    selected_features = []
    
    if method == 'mutual_info':
        # 使用互信息选择特征
        mi_scores = mutual_info_regression(X_scaled, y)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        selected_features = mi_scores.sort_values(ascending=False).index[:max_features].tolist()
        
    elif method == 'random_forest':
        # 使用随机森林特征重要性选择特征
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        selected_features = importances.sort_values(ascending=False).index[:max_features].tolist()
        
    elif method == 'lasso':
        # 使用LASSO选择特征
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, y)
        importances = pd.Series(np.abs(lasso.coef_), index=X.columns)
        selected_features = importances.sort_values(ascending=False).index[:max_features].tolist()
        
    elif method == 'hybrid':
        # 结合多种方法选择特征
        # 1. 互信息
        mi_scores = mutual_info_regression(X_scaled, y)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_ranks = mi_scores.rank(ascending=False)
        
        # 2. 随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_scores = pd.Series(rf.feature_importances_, index=X.columns)
        rf_ranks = rf_scores.rank(ascending=False)
        
        # 3. LASSO
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, y)
        lasso_scores = pd.Series(np.abs(lasso.coef_), index=X.columns)
        lasso_ranks = lasso_scores.rank(ascending=False)
        
        # 结合排名
        combined_ranks = mi_ranks + rf_ranks + lasso_ranks
        selected_features = combined_ranks.sort_values().index[:max_features].tolist()
    
    # 确保核心物理特征被包含
    core_features = [
        'permeability',           # 渗透率
        'oil_viscosity',          # 油相粘度
        'well_spacing',           # 井距
        'effective_thickness',    # 有效厚度
        'formation_pressure',     # 地层压力
        'mobility_ratio',         # 迁移性比
        'fingering_index',        # 指进系数
        'flow_capacity_index',    # 流动能力指数
        'gravity_number',         # 重力数
        'pressure_viscosity_ratio' # 压力-粘度比
    ]
    
    # 过滤掉不在数据集中的特征
    core_features = [f for f in core_features if f in df.columns]
    
    # 确保核心特征被包含，同时保持总数不超过max_features
    for feature in core_features:
        if feature not in selected_features and len(selected_features) < max_features:
            selected_features.append(feature)
        elif feature not in selected_features:
            # 替换最不重要的特征
            selected_features[-1] = feature
    
    logger.info(f"选择了{len(selected_features)}个特征: {selected_features}")
    
    return selected_features

def evaluate_feature_set(df, target_column, features, model=None):
    """
    评估特征集的性能
    
    Args:
        df: 数据DataFrame
        target_column: 目标变量列名
        features: 特征列表
        model: 评估模型，默认为随机森林
        
    Returns:
        float: 交叉验证R²得分
    """
    if target_column not in features:
        features = features + [target_column]
        
    X = df[features].drop(columns=[target_column])
    y = df[features][target_column]
    
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    mean_r2 = np.mean(cv_scores)
    
    logger.info(f"特征集 {features} 的平均R²: {mean_r2:.4f}")
    
    return mean_r2

def optimize_features_for_small_sample(df, target_column, max_features=10):
    """
    针对小样本量数据优化特征
    
    Args:
        df: 输入DataFrame
        target_column: 目标变量列名
        max_features: 最大特征数量
        
    Returns:
        DataFrame: 优化后的DataFrame
    """
    logger.info("开始针对小样本量数据优化特征")
    
    # 1. 创建物理约束特征
    df_physics = create_physics_informed_features(df)
    
    # 2. 使用不同方法选择特征并评估
    methods = ['mutual_info', 'random_forest', 'lasso', 'hybrid']
    best_method = None
    best_features = None
    best_score = -np.inf
    
    for method in methods:
        logger.info(f"使用 {method} 方法选择特征")
        features = select_optimal_features_limited(df_physics, target_column, max_features, method)
        score = evaluate_feature_set(df_physics, target_column, features)
        
        if score > best_score:
            best_score = score
            best_method = method
            best_features = features
    
    logger.info(f"最佳特征选择方法: {best_method}, R²: {best_score:.4f}")
    logger.info(f"最佳特征集 (共{len(best_features)}个): {best_features}")
    
    # 3. 返回包含最佳特征的DataFrame
    if target_column not in best_features:
        best_features.append(target_column)
        
    return df_physics[best_features]

if __name__ == "__main__":
    # 测试代码
    from data_processor import load_data, preprocess_data
    
    # 加载数据
    df = load_data()
    
    # 预处理数据
    df = preprocess_data(df)
    
    # 设置目标变量
    target_column = 'PV_number'
    
    # 创建物理约束特征
    df_physics = create_physics_informed_features(df)
    
    # 选择最优特征
    selected_features = select_optimal_features_limited(df_physics, target_column, max_features=10)
    
    # 评估特征集
    score = evaluate_feature_set(df_physics, target_column, selected_features)
    
    print(f"选择的特征: {selected_features}")
    print(f"特征集评估R²: {score:.4f}")

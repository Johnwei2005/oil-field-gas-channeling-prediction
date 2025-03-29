#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统增强特征工程模块

本模块实现了基于物理约束的特征工程方法，针对小样本量数据集进行优化。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import logging

# 导入配置
from config import PHYSICS_CONFIG, FEATURE_CONFIG, PATHS, DATA_CONFIG

# 设置日志
logger = logging.getLogger(__name__)

def create_physics_informed_features(df):
    """
    创建基于物理约束的特征
    
    Args:
        df: 输入数据DataFrame
        
    Returns:
        pandas.DataFrame: 增强后的DataFrame
    """
    logger.info("开始创建物理约束特征")
    
    # 创建原始数据副本
    df_enhanced = df.copy()
    
    # 1. 压力差 (Pressure Difference)
    if all(col in df.columns for col in ['formation_pressure', 'pre_injection_pressure']):
        df_enhanced['pressure_difference'] = df['formation_pressure'] - df['pre_injection_pressure']
        logger.info("创建特征: 压力差(pressure_difference)")
    
    # 2. 渗透率-孔隙度比 (Permeability-Porosity Ratio)
    if all(col in df.columns for col in ['permeability', 'porosity']):
        # 添加小常数避免除零错误
        df_enhanced['perm_porosity_ratio'] = df['permeability'] / (df['porosity'] + 1e-10)
        logger.info("创建特征: 渗透率-孔隙度比(perm_porosity_ratio)")
    
    # 3. 井距-厚度比 (Spacing-Thickness Ratio)
    if all(col in df.columns for col in ['well_spacing', 'effective_thickness']):
        # 添加小常数避免除零错误
        df_enhanced['spacing_thickness_ratio'] = df['well_spacing'] / (df['effective_thickness'] + 1e-10)
        logger.info("创建特征: 井距-厚度比(spacing_thickness_ratio)")
    
    # 4. 储层能量 (Reservoir Energy)
    if all(col in df.columns for col in ['formation_pressure', 'formation_temperature']):
        # 简化的储层能量计算
        df_enhanced['reservoir_energy'] = df['formation_pressure'] * df['formation_temperature'] / 100
        logger.info("创建特征: 储层能量(reservoir_energy)")
    
    # 5. 相迁移性指数 (Phase Mobility Index)
    if all(col in df.columns for col in ['oil_viscosity', 'permeability']):
        co2_visc = PHYSICS_CONFIG['co2_viscosity']
        df_enhanced['phase_mobility_index'] = (df['permeability'] / (df['oil_viscosity'] + 1e-10)) * (co2_visc / (df['oil_viscosity'] + 1e-10))
        logger.info("创建特征: 相迁移性指数(phase_mobility_index)")
    
    # 6. 驱替效率 (Displacement Efficiency)
    if all(col in df.columns for col in ['pre_injection_oil_saturation']):
        # 简化的驱替效率计算
        Sor = 0.3  # 假设剩余油饱和度
        df_enhanced['displacement_efficiency'] = (df['pre_injection_oil_saturation'] - Sor) / (df['pre_injection_oil_saturation'] + 1e-10)
        logger.info("创建特征: 驱替效率(displacement_efficiency)")
    
    # 7. 注入效率 (Injection Efficiency)
    if all(col in df.columns for col in ['permeability', 'oil_viscosity', 'well_spacing']):
        # 简化的注入效率计算
        df_enhanced['injection_efficiency'] = df['permeability'] / ((df['oil_viscosity'] + 1e-10) * (df['well_spacing'] + 1e-10))
        logger.info("创建特征: 注入效率(injection_efficiency)")
    
    # 8. 流动能力 (Flow Capacity)
    if all(col in df.columns for col in ['permeability', 'effective_thickness']):
        df_enhanced['flow_capacity'] = df['permeability'] * df['effective_thickness']
        logger.info("创建特征: 流动能力(flow_capacity)")
    
    # 9. 综合稳定性指数 (Combined Stability Index)
    if all(col in df.columns for col in ['mobility_ratio', 'gravity_number']):
        df_enhanced['combined_stability_index'] = df['mobility_ratio'] / (df['gravity_number'] + 1e-10)
        logger.info("创建特征: 综合稳定性指数(combined_stability_index)")
    
    # 10. CO2溶解度因子 (CO2 Solubility Factor)
    if all(col in df.columns for col in ['formation_temperature', 'formation_pressure']):
        # 简化的CO2溶解度计算
        T_ref = 20  # 参考温度
        P_ref = 5   # 参考压力
        df_enhanced['co2_solubility_factor'] = (df['formation_pressure'] / P_ref) * np.exp(-0.05 * (df['formation_temperature'] - T_ref))
        logger.info("创建特征: CO2溶解度因子(co2_solubility_factor)")
    
    return df_enhanced

def create_interaction_features(df, target_col=None, top_n=10):
    """
    创建特征交互项
    
    Args:
        df: 输入数据DataFrame
        target_col: 目标变量列名
        top_n: 选择的顶部特征数量
        
    Returns:
        pandas.DataFrame: 增强后的DataFrame
    """
    logger.info("开始创建特征交互项")
    
    # 创建原始数据副本
    df_enhanced = df.copy()
    
    # 如果指定了目标列，则排除目标列
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None
    
    # 选择数值型特征
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # 如果有目标变量，选择与目标相关性最高的特征
    if y is not None:
        # 使用互信息选择特征（适用于非线性关系）
        selector = SelectKBest(mutual_info_regression, k=min(top_n, len(numeric_cols)))
        selector.fit(X[numeric_cols], y)
        
        # 获取特征得分和索引
        scores = selector.scores_
        indices = np.argsort(scores)[::-1][:top_n]
        
        # 选择顶部特征
        selected_features = [numeric_cols[i] for i in indices if i < len(numeric_cols)]
        logger.info(f"选择了{len(selected_features)}个顶部特征用于交互项创建")
    else:
        # 如果没有目标变量，使用所有数值特征
        selected_features = numeric_cols
    
    # 创建交互特征
    n_features = len(selected_features)
    for i in range(n_features):
        for j in range(i+1, n_features):
            feat1 = selected_features[i]
            feat2 = selected_features[j]
            
            # 乘积交互
            interaction_name = f"{feat1}_x_{feat2}"
            df_enhanced[interaction_name] = df[feat1] * df[feat2]
            
            # 比率交互（避免除零）
            ratio_name = f"{feat1}_div_{feat2}"
            df_enhanced[ratio_name] = df[feat1] / (df[feat2] + 1e-10)
    
    logger.info(f"创建了{df_enhanced.shape[1] - df.shape[1]}个交互特征")
    return df_enhanced

def create_nonlinear_transformations(df, target_col=None):
    """
    创建非线性变换特征
    
    Args:
        df: 输入数据DataFrame
        target_col: 目标变量列名
        
    Returns:
        pandas.DataFrame: 增强后的DataFrame
    """
    logger.info("开始创建非线性变换特征")
    
    # 创建原始数据副本
    df_enhanced = df.copy()
    
    # 如果指定了目标列，则排除目标列
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    
    # 选择数值型特征
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # 对每个数值特征创建非线性变换
    for col in numeric_cols:
        # 确保数据为正（对于对数变换）
        min_val = df[col].min()
        
        # 平方变换
        df_enhanced[f"{col}_squared"] = df[col] ** 2
        
        # 平方根变换（确保非负）
        if min_val >= 0:
            df_enhanced[f"{col}_sqrt"] = np.sqrt(df[col])
        else:
            shift = abs(min_val) + 1e-3
            df_enhanced[f"{col}_sqrt"] = np.sqrt(df[col] + shift)
        
        # 对数变换（确保正值）
        if min_val <= 0:
            shift = abs(min_val) + 1e-3
            df_enhanced[f"{col}_log"] = np.log1p(df[col] + shift)
        else:
            df_enhanced[f"{col}_log"] = np.log1p(df[col])
    
    logger.info(f"创建了{df_enhanced.shape[1] - df.shape[1]}个非线性变换特征")
    return df_enhanced

def select_optimal_features(df, target_col, method='hybrid', n_features=None):
    """
    选择最优特征子集
    
    Args:
        df: 输入数据DataFrame
        target_col: 目标变量列名
        method: 特征选择方法，可选'mutual_info', 'f_regression', 'random_forest', 'hybrid'
        n_features: 选择的特征数量，如果为None则自动确定
        
    Returns:
        list: 选择的特征列表
    """
    logger.info(f"使用{method}方法选择最优特征")
    
    # 分离特征和目标
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 选择数值型特征
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    # 如果未指定特征数量，则自动确定（样本量的平方根）
    if n_features is None:
        n_features = min(int(np.sqrt(len(df))), len(numeric_cols))
    else:
        n_features = min(n_features, len(numeric_cols))
    
    logger.info(f"将选择{n_features}个特征")
    
    # 根据方法选择特征
    if method == 'mutual_info':
        # 互信息法（适用于非线性关系）
        selector = SelectKBest(mutual_info_regression, k=n_features)
        selector.fit(X_numeric, y)
        
        # 获取特征得分和索引
        scores = selector.scores_
        indices = np.argsort(scores)[::-1][:n_features]
        
        # 选择特征
        selected_features = [numeric_cols[i] for i in indices if i < len(numeric_cols)]
        
    elif method == 'f_regression':
        # F检验法（适用于线性关系）
        selector = SelectKBest(f_regression, k=n_features)
        selector.fit(X_numeric, y)
        
        # 获取特征得分和索引
        scores = selector.scores_
        indices = np.argsort(scores)[::-1][:n_features]
        
        # 选择特征
        selected_features = [numeric_cols[i] for i in indices if i < len(numeric_cols)]
        
    elif method == 'random_forest':
        # 随机森林特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_numeric, y)
        
        # 获取特征重要性和索引
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        # 选择特征
        selected_features = [numeric_cols[i] for i in indices if i < len(numeric_cols)]
        
    elif method == 'hybrid':
        # 混合方法：结合互信息和随机森林
        # 互信息
        mi_selector = SelectKBest(mutual_info_regression, k=n_features)
        mi_selector.fit(X_numeric, y)
        mi_scores = mi_selector.scores_
        
        # 随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_numeric, y)
        rf_importances = rf.feature_importances_
        
        # 归一化得分
        mi_scores = mi_scores / np.sum(mi_scores)
        rf_importances = rf_importances / np.sum(rf_importances)
        
        # 组合得分
        combined_scores = (mi_scores + rf_importances) / 2
        indices = np.argsort(combined_scores)[::-1][:n_features]
        
        # 选择特征
        selected_features = [numeric_cols[i] for i in indices if i < len(numeric_cols)]
    
    else:
        raise ValueError(f"不支持的特征选择方法: {method}")
    
    # 确保包含物理约束特征
    physics_features = [
        'mobility_ratio', 'gravity_number', 'fingering_index', 
        'pressure_diffusivity', 'phase_transition_index',
        'perm_porosity_ratio', 'spacing_thickness_ratio',
        'reservoir_energy', 'displacement_efficiency',
        'phase_mobility_index', 'injection_efficiency',
        'flow_capacity', 'combined_stability_index',
        'co2_solubility_factor', 'pressure_difference'
    ]
    
    # 找出已存在于数据中的物理特征
    existing_physics_features = [f for f in physics_features if f in df.columns]
    
    # 将物理特征添加到选择的特征中（如果尚未包含）
    for feature in existing_physics_features:
        if feature not in selected_features:
            selected_features.append(feature)
    
    logger.info(f"最终选择了{len(selected_features)}个特征")
    return selected_features

def optimize_features_for_small_sample(df, target_col):
    """
    针对小样本量数据集优化特征
    
    Args:
        df: 输入数据DataFrame
        target_col: 目标变量列名
        
    Returns:
        pandas.DataFrame: 优化后的特征DataFrame
    """
    logger.info("开始针对小样本量数据集优化特征")
    
    # 1. 创建物理约束特征
    df_enhanced = create_physics_informed_features(df)
    
    # 2. 创建特征交互项
    df_enhanced = create_interaction_features(df_enhanced, target_col, top_n=5)
    
    # 3. 创建非线性变换特征
    df_enhanced = create_nonlinear_transformations(df_enhanced, target_col)
    
    # 4. 选择最优特征子集
    # 对于小样本量，特征数量应该限制在样本量的平方根左右
    n_features = min(int(np.sqrt(len(df))), 10)
    selected_features = select_optimal_features(df_enhanced, target_col, method='hybrid', n_features=n_features)
    
    # 确保包含目标列
    if target_col not in selected_features:
        selected_features.append(target_col)
    
    # 返回选择的特征子集
    return df_enhanced[selected_features]

def evaluate_feature_set(df, target_col, cv=5):
    """
    评估特征集的性能
    
    Args:
        df: 输入数据DataFrame
        target_col: 目标变量列名
        cv: 交叉验证折数
        
    Returns:
        dict: 评估指标
    """
    # 分离特征和目标
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 使用随机森林进行评估
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    # 计算评估指标
    metrics = {
        'r2_mean': np.mean(cv_scores),
        'r2_std': np.std(cv_scores),
        'feature_count': X.shape[1]
    }
    
    logger.info(f"特征集评估结果: R² = {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}, 特征数量 = {metrics['feature_count']}")
    return metrics

def plot_feature_importance(df, target_col, output_path=None):
    """
    绘制特征重要性图
    
    Args:
        df: 输入数据DataFrame
        target_col: 目标变量列名
        output_path: 输出文件路径
        
    Returns:
        None
    """
    # 分离特征和目标
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 使用随机森林计算特征重要性
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 获取特征重要性
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('特征重要性排名', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"特征重要性图已保存至 {output_path}")
    
    plt.close()
    
    return feature_importance

if __name__ == "__main__":
    # 测试代码
    from data_processor import load_data, preprocess_data, engineer_features
    
    # 加载数据
    df = load_data()
    
    # 预处理数据
    df = preprocess_data(df)
    
    # 基础特征工程
    df = engineer_features(df)
    
    # 增强特征工程
    target_col = DATA_CONFIG['target_column']
    df_optimized = optimize_features_for_small_sample(df, target_col)
    
    # 评估特征集
    metrics = evaluate_feature_set(df_optimized, target_col)
    
    # 绘制特征重要性
    output_path = os.path.join(PATHS['results_dir'], 'enhanced_feature_importance.png')
    feature_importance = plot_feature_importance(df_optimized, target_col, output_path)
    
    # 保存优化后的特征
    output_file = os.path.join(PATHS['results_dir'], 'optimized_features.csv')
    df_optimized.to_csv(output_file, index=False)
    logger.info(f"优化后的特征已保存至 {output_file}")

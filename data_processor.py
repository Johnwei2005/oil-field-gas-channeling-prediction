#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统数据处理模块

本模块负责数据加载、清洗、特征工程等处理，将原始数据转换为模型可用的特征。
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer
import logging
from datetime import datetime
# 导入配置
from config import DATA_CONFIG, COLUMN_MAPPING, PHYSICS_CONFIG, FEATURE_CONFIG, PATHS

# 设置日志
logger = logging.getLogger(__name__)

def load_data(file_path=None, encoding=None):
    """
    加载原始数据
    
    Args:
        file_path: 数据文件路径，若为None则使用配置文件中的路径
        encoding: 文件编码，若为None则使用配置文件中的编码
        
    Returns:
        pandas.DataFrame: 加载的原始数据
    """
    if file_path is None:
        file_path = os.path.join(PATHS['data_dir'], DATA_CONFIG['file_path'])
    if encoding is None:
        encoding = DATA_CONFIG['encoding']
    
    logger.info(f"正在加载数据文件: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"成功加载数据，共{len(df)}行，{len(df.columns)}列")
        return df
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise

def preprocess_data(df):
    """
    预处理原始数据，包括列名转换、缺失值处理、异常值处理和特征变换
    
    Args:
        df: 原始数据DataFrame
        
    Returns:
        pandas.DataFrame: 预处理后的数据
    """
    logger.info("开始数据预处理")
    
    # 保存原始数据副本，用于后续对比
    df_original = df.copy()
    
    # 重命名列（中文转英文）
    df = df.rename(columns=COLUMN_MAPPING)
    logger.info(f"列名已转换为英文，当前列名: {', '.join(df.columns)}")
    
    # 检测并处理缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"检测到缺失值: \n{missing_values[missing_values > 0]}")
        
        # 细分缺失值处理策略：按列类型和缺失比例
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            
            # 检查列的类型
            if col in df.select_dtypes(include=[np.number]).columns:
                if missing_pct > 0.3:  # 缺失率超过30%
                    logger.warning(f"列'{col}'缺失率较高({missing_pct:.2%})，考虑移除该特征")
                    # 不移除，而是用平均值代替，便于后续分析
                    df[col].fillna(df[col].mean(), inplace=True)
                    logger.info(f"列'{col}'使用平均值填充")
                elif missing_pct > 0.1:  # 缺失率10%-30%
                    # 使用更高级的方法填充，如MICE或迭代填充
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    logger.info(f"列'{col}'使用迭代填充方法处理缺失值")
                else:  # 缺失率低于10%
                    # 使用KNN填充
                    numeric_data = df.select_dtypes(include=[np.number])
                    try:
                        imputer = KNNImputer(n_neighbors=min(5, len(numeric_data) - 1))
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        logger.info(f"列'{col}'使用KNN方法填充缺失值")
                    except Exception as e:
                        logger.error(f"KNN填充失败: {str(e)}，改用中位数填充")
                        df[col].fillna(df[col].median(), inplace=True)
            else:
                # 非数值列使用众数填充
                try:
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)
                    logger.info(f"非数值列'{col}'使用众数填充")
                except:
                    logger.warning(f"列'{col}'无法填充缺失值")
    
    # 处理分类特征
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in COLUMN_MAPPING.values():  # 只处理已知的分类特征
            if FEATURE_CONFIG['categorical_encoding'] == 'one_hot':
                # 一热编码
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                )
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
                logger.info(f"列'{col}'已进行一热编码")
            elif FEATURE_CONFIG['categorical_encoding'] == 'target':
                # 目标编码 - 考虑特征的预测价值
                if DATA_CONFIG['target_column'] in df.columns:
                    # 创建交叉验证fold以避免数据泄露
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    
                    # 初始化目标编码列
                    df[f"{col}_target_encoded"] = np.nan
                    
                    # 对每个fold进行目标编码
                    for train_idx, val_idx in kf.split(df):
                        # 计算训练集中每个类别的目标平均值
                        target_means = df.iloc[train_idx].groupby(col)[DATA_CONFIG['target_column']].mean()
                        
                        # 将编码应用到验证集
                        df.loc[val_idx, f"{col}_target_encoded"] = df.loc[val_idx, col].map(target_means)
                    
                    # 处理测试集中出现但训练集中不存在的类别
                    global_mean = df[DATA_CONFIG['target_column']].mean()
                    df[f"{col}_target_encoded"].fillna(global_mean, inplace=True)
                    
                    # 删除原始分类列
                    df.drop(col, axis=1, inplace=True)
                    logger.info(f"列'{col}'已进行目标编码")
                else:
                    # 如果没有目标列，则使用标签编码
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
                    logger.info(f"目标列不存在，列'{col}'改为标签编码")
            else:
                # 标签编码
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                logger.info(f"列'{col}'已进行标签编码")
    
    # 处理异常值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == DATA_CONFIG['target_column']:
            continue  # 不处理目标变量
            
        # 计算IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 确定异常值边界 (加强版处理，使用3倍IQR)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # 检测异常值
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            logger.info(f"列'{col}'检测到{outlier_count}个异常值")
            
            # 对数值型特征应用温和处理，而非直接截断
            # 使用Winsorization方法 (将异常值设为边界值)
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            logger.info(f"列'{col}'的异常值已使用Winsorization方法处理")
    
    # 类型转换与特征变换
    for col in df.columns:
        # 处理数值转换
        if col in FEATURE_CONFIG['base_features'] and df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                logger.info(f"列'{col}'已转换为数值类型")
            except:
                logger.warning(f"列'{col}'无法转换为数值类型")
                
        # 对数转换适合偏斜分布的特征
        if col in numeric_cols and col != DATA_CONFIG['target_column']:
            # 检查分布偏斜
            skewness = df[col].skew()
            
            # 对高度偏斜的特征应用对数变换
            if abs(skewness) > 1.5:
                # 确保数据为正
                min_val = df[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1e-3  # 添加小偏移量
                    df[f"{col}_log"] = np.log1p(df[col] + shift)
                else:
                    df[f"{col}_log"] = np.log1p(df[col])
                logger.info(f"列'{col}'偏斜度为{skewness:.2f}，已创建对数变换特征")
    
    # 目标变量变换 (仅在目标列存在且为训练数据时)
    if DATA_CONFIG['target_column'] in df.columns:
        target_col = DATA_CONFIG['target_column']
        # 检查目标变量分布
        skewness = df[target_col].skew()
        
        # 仅为严重偏斜的目标变量应用变换
        if abs(skewness) > 1.0:
            # 保存原始目标变量
            df[f"{target_col}_original"] = df[target_col].copy()
            
            # 确保数据为正
            min_val = df[target_col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1e-3
                df[target_col] = np.log1p(df[target_col] + shift)
                logger.info(f"目标变量'{target_col}'已应用对数变换 (log1p)，添加了{shift}的偏移量")
            else:
                df[target_col] = np.log1p(df[target_col])
                logger.info(f"目标变量'{target_col}'已应用对数变换 (log1p)")
                
            # 添加标记，便于后续逆变换
            df.attrs['target_transformed'] = True
            df.attrs['transform_type'] = 'log1p'
            df.attrs['transform_shift'] = shift if min_val <= 0 else 0
        else:
            logger.info(f"目标变量'{target_col}'分布不是严重偏斜的 (偏斜度={skewness:.2f})，不应用变换")
    
    # 特征归一化预处理 (不应用于二值特征和目标变量)
    if FEATURE_CONFIG.get('apply_normalization', True):
        # 识别需要归一化的列 (排除二值列和目标列)
        norm_cols = []
        for col in numeric_cols:
            if col == DATA_CONFIG['target_column']:
                continue
            # 检查是否为二值列
            if len(df[col].unique()) <= 2:
                continue
            norm_cols.append(col)
        
        if norm_cols:
            # 选择归一化方法
            if FEATURE_CONFIG['scaling_method'] == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df[norm_cols] = scaler.fit_transform(df[norm_cols])
                logger.info(f"使用MinMaxScaler对{len(norm_cols)}个特征进行归一化")
            elif FEATURE_CONFIG['scaling_method'] == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df[norm_cols] = scaler.fit_transform(df[norm_cols])
                logger.info(f"使用RobustScaler对{len(norm_cols)}个特征进行归一化")
            else:
                # 默认使用StandardScaler
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df[norm_cols] = scaler.fit_transform(df[norm_cols])
                logger.info(f"使用StandardScaler对{len(norm_cols)}个特征进行归一化")
            
            # 保存缩放器，以便应用于测试数据
            df.attrs['feature_scaler'] = scaler
            df.attrs['scaled_columns'] = norm_cols
    
    # 对比处理前后的数据统计
    numeric_cols_after = df.select_dtypes(include=[np.number]).columns
    stats_before = df_original.describe()
    stats_after = df.describe()
    
    logger.info(f"数据预处理完成。处理前：{df_original.shape}，处理后：{df.shape}")
    logger.info(f"新增特征: {set(df.columns) - set(df_original.columns)}")
    
    return df
    

def engineer_features(df):
    """
    特征工程，创建物理模型驱动的派生特征
    并进行增强的NaN监控
    
    Args:
        df: 预处理后的数据DataFrame
        
    Returns:
        pandas.DataFrame: 增加了派生特征的数据
    """
    logger.info("开始特征工程")
    
    # 创建原始数据副本，用于后续验证
    df_original = df.copy()
    
    # 用于跟踪出现NaN的特征
    nan_features = {}
    
    # 创建一个记录每个特征计算过程的列表，用于NaN报告
    feature_creation_log = []
    
    # 确保所有基本特征都存在
    for feature in FEATURE_CONFIG['base_features']:
        if feature not in df.columns:
            logger.warning(f"基础特征'{feature}'不存在，特征工程可能不完整")
    
    # 创建一个函数来检查NaN值并记录信息
    def check_nan_feature(df, feature_name, description=""):
        """检查特征中的NaN值并记录信息"""
        if feature_name not in df.columns:
            return 0
        
        nan_count = df[feature_name].isna().sum()
        if nan_count > 0:
            logger.warning(f"特征'{feature_name}'包含{nan_count}个NaN值")
            nan_features[feature_name] = nan_count
            
            # 记录特征创建过程
            feature_creation_log.append({
                'feature': feature_name,
                'nan_count': nan_count,
                'description': description,
                'nan_indices': df[df[feature_name].isna()].index.tolist()[:10]  # 最多记录10个索引
            })
            
            return nan_count
        return 0
    
    # 1. 迁移性比 (Mobility Ratio)
    if all(col in df.columns for col in ['oil_viscosity']):
        try:
            # 添加小常数避免除零错误
            co2_visc = PHYSICS_CONFIG['co2_viscosity']
            df['mobility_ratio'] = df['oil_viscosity'] / (co2_visc + 1e-10)
            check_nan_feature(df, 'mobility_ratio', "迁移性比 = oil_viscosity / co2_viscosity")
            logger.info("创建特征: 迁移性比(mobility_ratio)")
        except Exception as e:
            logger.error(f"计算'mobility_ratio'时出错: {str(e)}")
            nan_features['mobility_ratio'] = -1  # 使用-1表示计算错误
    
    # 2. 重力数 (Gravity Number)
    if all(col in df.columns for col in ['oil_density', 'permeability', 'effective_thickness', 'oil_viscosity']):
        try:
            g = PHYSICS_CONFIG['gravity']
            co2_density = PHYSICS_CONFIG['co2_density']
            # 添加小常数避免除零错误
            df['gravity_number'] = (df['oil_density'] - co2_density) * g * df['permeability'] * df['effective_thickness'] / (df['oil_viscosity'] + 1e-10)
            check_nan_feature(df, 'gravity_number', "重力数 = (oil_density - co2_density) * g * permeability * effective_thickness / oil_viscosity")
            logger.info("创建特征: 重力数(gravity_number)")
        except Exception as e:
            logger.error(f"计算'gravity_number'时出错: {str(e)}")
            nan_features['gravity_number'] = -1
    
    # 3. 指进系数 (Fingering Index)
    if all(col in df.columns for col in ['mobility_ratio']):
        try:
            # 假设不可动水饱和度为0.2，剩余油饱和度为0.3
            Swi = 0.2
            Sor = 0.3
            df['fingering_index'] = df['mobility_ratio'] * (1 - Swi - Sor)
            check_nan_feature(df, 'fingering_index', "指进系数 = mobility_ratio * (1 - Swi - Sor)")
            logger.info("创建特征: 指进系数(fingering_index)")
        except Exception as e:
            logger.error(f"计算'fingering_index'时出错: {str(e)}")
            nan_features['fingering_index'] = -1
    
    # 4. 压力扩散系数 (Pressure Diffusivity)
    if all(col in df.columns for col in ['permeability', 'porosity', 'oil_viscosity']):
        try:
            ct = PHYSICS_CONFIG['default_compressibility']
            k_factor = 9.869e-16  # md到m²的转换因子
            # 添加小常数避免除零错误
            df['pressure_diffusivity'] = (df['permeability'] * k_factor) / ((df['porosity'] + 1e-10) * (df['oil_viscosity'] * 1e-3 + 1e-10) * ct)
            check_nan_feature(df, 'pressure_diffusivity', "压力扩散系数 = (permeability * k_factor) / (porosity * oil_viscosity * ct)")
            logger.info("创建特征: 压力扩散系数(pressure_diffusivity)")
        except Exception as e:
            logger.error(f"计算'pressure_diffusivity'时出错: {str(e)}")
            nan_features['pressure_diffusivity'] = -1
    
    # 5. 相态转化指数 (Phase Transition Index)
    if all(col in df.columns for col in ['formation_temperature', 'formation_pressure']):
        try:
            T_crit = PHYSICS_CONFIG['critical_temperature']
            P_crit = PHYSICS_CONFIG['critical_pressure']
            df['phase_transition_index'] = (df['formation_temperature'] / (T_crit + 1e-10)) + (df['formation_pressure'] / (P_crit + 1e-10))
            check_nan_feature(df, 'phase_transition_index', "相态转化指数 = (formation_temperature / T_crit) + (formation_pressure / P_crit)")
            logger.info("创建特征: 相态转化指数(phase_transition_index)")
        except Exception as e:
            logger.error(f"计算'phase_transition_index'时出错: {str(e)}")
            nan_features['phase_transition_index'] = -1
    
    # 6. 注入强度比 (Injection Intensity Ratio)
    if all(col in df.columns for col in ['pv_number', 'porosity', 'effective_thickness']):
        try:
            # 添加小常数避免除零错误
            df['injection_intensity_ratio'] = df['pv_number'] / ((df['porosity'] + 1e-10) * (df['effective_thickness'] + 1e-10))
            check_nan_feature(df, 'injection_intensity_ratio', "注入强度比 = pv_number / (porosity * effective_thickness)")
            logger.info("创建特征: 注入强度比(injection_intensity_ratio)")
        except Exception as e:
            logger.error(f"计算'injection_intensity_ratio'时出错: {str(e)}")
            nan_features['injection_intensity_ratio'] = -1
    
    # 7. 驱替效率估计 (Displacement Efficiency)
    if all(col in df.columns for col in ['pre_injection_oil_saturation', 'oil_viscosity']):
        try:
            Swi = 0.2  # 不可动水饱和度
            Sor = 0.3  # 剩余油饱和度
            co2_visc = PHYSICS_CONFIG['co2_viscosity']
            
            # 计算驱替效率，添加小常数避免除零错误
            df['displacement_efficiency'] = 1 / (1 + (Sor * df['oil_viscosity']) / (Swi * (co2_visc + 1e-10) + 1e-10))
            check_nan_feature(df, 'displacement_efficiency', "驱替效率 = 1 / (1 + (Sor * oil_viscosity) / (Swi * co2_visc))")
            logger.info("创建特征: 驱替效率(displacement_efficiency)")
        except Exception as e:
            logger.error(f"计算'displacement_efficiency'时出错: {str(e)}")
            nan_features['displacement_efficiency'] = -1
    
    # 8. 储层能量 (Reservoir Energy)
    if all(col in df.columns for col in ['formation_pressure', 'pre_injection_oil_saturation']):
        try:
            df['reservoir_energy'] = df['formation_pressure'] * df['pre_injection_oil_saturation']
            check_nan_feature(df, 'reservoir_energy', "储层能量 = formation_pressure * pre_injection_oil_saturation")
            logger.info("创建特征: 储层能量(reservoir_energy)")
        except Exception as e:
            logger.error(f"计算'reservoir_energy'时出错: {str(e)}")
            nan_features['reservoir_energy'] = -1
    
    # 9. CO2溶解度因子 (CO2 Solubility Factor)
    if all(col in df.columns for col in ['formation_temperature', 'formation_pressure']):
        try:
            # 简化的CO2溶解度计算
            df['co2_solubility_factor'] = 0.01 * df['formation_pressure'] * np.exp(-0.05 * df['formation_temperature'])
            check_nan_feature(df, 'co2_solubility_factor', "CO2溶解度因子 = 0.01 * formation_pressure * np.exp(-0.05 * formation_temperature)")
            logger.info("创建特征: CO2溶解度因子(co2_solubility_factor)")
        except Exception as e:
            logger.error(f"计算'co2_solubility_factor'时出错: {str(e)}")
            nan_features['co2_solubility_factor'] = -1
    
    # 10. 压差 (Pressure Difference)
    if all(col in df.columns for col in ['formation_pressure', 'pre_injection_pressure']):
        try:
            df['pressure_difference'] = df['formation_pressure'] - df['pre_injection_pressure']
            check_nan_feature(df, 'pressure_difference', "压差 = formation_pressure - pre_injection_pressure")
            logger.info("创建特征: 压差(pressure_difference)")
        except Exception as e:
            logger.error(f"计算'pressure_difference'时出错: {str(e)}")
            nan_features['pressure_difference'] = -1

    # 11. CO2/原油流动性对比 (CO2-Oil Mobility Ratio)
    if all(col in df.columns for col in ['oil_viscosity']):
        try:
            co2_viscosity = PHYSICS_CONFIG['co2_viscosity']
            df['co2_oil_mobility_contrast'] = df['oil_viscosity'] / (co2_viscosity + 1e-10)
            check_nan_feature(df, 'co2_oil_mobility_contrast', "CO2/原油流动性对比 = oil_viscosity / co2_viscosity")
            logger.info("创建特征: CO2/原油流动性对比(co2_oil_mobility_contrast)")
        except Exception as e:
            logger.error(f"计算'co2_oil_mobility_contrast'时出错: {str(e)}")
            nan_features['co2_oil_mobility_contrast'] = -1

    # 12. 驱替前缘稳定性指数 (Displacement Front Stability Index)
    if all(col in df.columns for col in ['permeability', 'oil_viscosity', 'well_spacing']):
        try:
            df['displacement_front_stability'] = df['permeability'] / ((df['oil_viscosity'] + 1e-10) * (df['well_spacing'] + 1e-10))
            check_nan_feature(df, 'displacement_front_stability', "驱替前缘稳定性指数 = permeability / (oil_viscosity * well_spacing)")
            logger.info("创建特征: 驱替前缘稳定性指数(displacement_front_stability)")
        except Exception as e:
            logger.error(f"计算'displacement_front_stability'时出错: {str(e)}")
            nan_features['displacement_front_stability'] = -1

    # 13. 储层储能系数 (Reservoir Storage Coefficient)
    if all(col in df.columns for col in ['porosity', 'formation_pressure']):
        try:
            # 假设原油压缩系数为常数
            oil_compressibility = PHYSICS_CONFIG['default_compressibility']
            df['reservoir_storage_coeff'] = df['porosity'] * oil_compressibility * df['formation_pressure']
            check_nan_feature(df, 'reservoir_storage_coeff', "储层储能系数 = porosity * oil_compressibility * formation_pressure")
            logger.info("创建特征: 储层储能系数(reservoir_storage_coeff)")
        except Exception as e:
            logger.error(f"计算'reservoir_storage_coeff'时出错: {str(e)}")
            nan_features['reservoir_storage_coeff'] = -1

    # 14. 相态迁移性指数 (Phase Transition Mobility Index)
    # 移除此特征，因为在NaN报告中发现它会产生NaN值
    """
    if all(col in df.columns for col in ['formation_temperature', 'formation_pressure', 'mobility_ratio']):
        critical_temp = PHYSICS_CONFIG['critical_temperature']
        critical_press = PHYSICS_CONFIG['critical_pressure']
        temp_ratio = df['formation_temperature'] / critical_temp
        press_ratio = df['formation_pressure'] / critical_press
        df['phase_mobility_index'] = temp_ratio * press_ratio * df['mobility_ratio']
        check_nan_feature(df, 'phase_mobility_index')
        logger.info("创建特征: 相态迁移性指数(phase_mobility_index)")
    """
    logger.info("跳过创建相态迁移性指数(phase_mobility_index)，因为它在NaN报告中被识别为有问题的特征")

    # 15. 特征交互项 (Feature Interactions)
    # 找出重要特征之间的交互
    important_features = ['formation_temperature', 'formation_pressure', 'permeability', 
                        'porosity', 'well_spacing', 'oil_viscosity', 'effective_thickness']
    for i in range(len(important_features)):
        for j in range(i+1, len(important_features)):
            feat1 = important_features[i]
            feat2 = important_features[j]
            if all(col in df.columns for col in [feat1, feat2]):
                try:
                    interaction_name = f"{feat1}_{feat2}_interaction"
                    df[interaction_name] = df[feat1] * df[feat2]
                    check_nan_feature(df, interaction_name, f"特征交互项 = {feat1} * {feat2}")
                    logger.info(f"创建特征交互项: {interaction_name}")
                except Exception as e:
                    logger.error(f"计算'{interaction_name}'时出错: {str(e)}")
                    nan_features[interaction_name] = -1

    # 16. 非线性变换 (Non-linear Transformations)
    for feature in important_features:
        if feature in df.columns:
            try:
                # 平方变换
                df[f"{feature}_squared"] = df[feature] ** 2
                check_nan_feature(df, f"{feature}_squared", f"平方变换 = {feature}^2")
                logger.info(f"创建非线性特征: {feature}_squared")
            except Exception as e:
                logger.error(f"计算'{feature}_squared'时出错: {str(e)}")
                nan_features[f"{feature}_squared"] = -1
            
            # 对数变换 (对非零正值)
            try:
                if (df[feature] > 0).all():
                    df[f"{feature}_log"] = np.log(df[feature])
                    check_nan_feature(df, f"{feature}_log", f"对数变换 = log({feature})")
                    logger.info(f"创建非线性特征: {feature}_log")
                elif (df[feature] != 0).all():
                    df[f"{feature}_log_abs"] = np.log(np.abs(df[feature]))
                    check_nan_feature(df, f"{feature}_log_abs", f"绝对值对数变换 = log(|{feature}|)")
                    logger.info(f"创建非线性特征: {feature}_log_abs")
            except Exception as e:
                logger.error(f"计算'{feature}_log'或'{feature}_log_abs'时出错: {str(e)}")
                nan_features[f"{feature}_log"] = -1

    # 17. 重力/粘性力比值 (Gravity to Viscous Force Ratio)
    if all(col in df.columns for col in ['oil_density', 'permeability', 'oil_viscosity', 'effective_thickness']):
        try:
            g = PHYSICS_CONFIG['gravity']
            co2_density = PHYSICS_CONFIG['co2_density']
            density_diff = np.abs(df['oil_density'] - co2_density)
            # 添加小常数避免除零错误
            df['gravity_viscous_ratio'] = (density_diff * g * df['permeability'] * df['effective_thickness']) / (df['oil_viscosity'] + 1e-10)
            check_nan_feature(df, 'gravity_viscous_ratio', "重力/粘性力比值 = (density_diff * g * permeability * effective_thickness) / oil_viscosity")
            logger.info("创建特征: 重力/粘性力比值(gravity_viscous_ratio)")
        except Exception as e:
            logger.error(f"计算'gravity_viscous_ratio'时出错: {str(e)}")
            nan_features['gravity_viscous_ratio'] = -1

    # 18. 渗透率与孔隙度比值 (Permeability to Porosity Ratio)
    if all(col in df.columns for col in ['permeability', 'porosity']):
        try:
            # 添加小常数避免除零错误
            df['perm_porosity_ratio'] = df['permeability'] / (df['porosity'] + 1e-10)
            check_nan_feature(df, 'perm_porosity_ratio', "渗透率与孔隙度比值 = permeability / porosity")
            logger.info("创建特征: 渗透率与孔隙度比值(perm_porosity_ratio)")
        except Exception as e:
            logger.error(f"计算'perm_porosity_ratio'时出错: {str(e)}")
            nan_features['perm_porosity_ratio'] = -1

    # 19. 井距与有效厚度比值 (Well Spacing to Effective Thickness Ratio)
    if all(col in df.columns for col in ['well_spacing', 'effective_thickness']):
        try:
            # 添加小常数避免除零错误
            df['spacing_thickness_ratio'] = df['well_spacing'] / (df['effective_thickness'] + 1e-10)
            check_nan_feature(df, 'spacing_thickness_ratio', "井距与有效厚度比值 = well_spacing / effective_thickness")
            logger.info("创建特征: 井距与有效厚度比值(spacing_thickness_ratio)")
        except Exception as e:
            logger.error(f"计算'spacing_thickness_ratio'时出错: {str(e)}")
            nan_features['spacing_thickness_ratio'] = -1
    
    # 20. 增强相态转换建模
    # 移除此部分，因为在NaN报告中发现它包含会产生NaN的特征
    logger.info("跳过创建增强相态转换特征，因为它们在NaN报告中被识别为有问题的特征")
    
    # 21. 高级压力-粘度交互
    if all(col in df.columns for col in ['formation_pressure', 'oil_viscosity', 'formation_temperature']):
        try:
            # 三维交互项
            df['pressure_visc_temp_interaction'] = df['formation_pressure'] * df['oil_viscosity'] * df['formation_temperature']
            check_nan_feature(df, 'pressure_visc_temp_interaction', "压力-粘度-温度三交互 = formation_pressure * oil_viscosity * formation_temperature")
            logger.info("创建特征: 压力-粘度-温度三交互(pressure_visc_temp_interaction)")
        except Exception as e:
            logger.error(f"计算'pressure_visc_temp_interaction'时出错: {str(e)}")
            nan_features['pressure_visc_temp_interaction'] = -1
        
        # 粘弹性理论相关特征 - 移除这个特征，因为NaN报告中标识为问题特征
        logger.info("跳过创建粘弹性指数(viscoelastic_index)，因为它在NaN报告中被识别为有问题的特征")

    # 22. 增强注入特征
    if all(col in df.columns for col in ['injection_intensity_ratio', 'permeability', 'porosity']):
        try:
            # 注入效率
            df['injection_efficiency'] = df['injection_intensity_ratio'] * (df['permeability'] / (df['porosity'] + 1e-10))
            check_nan_feature(df, 'injection_efficiency', "注入效率 = injection_intensity_ratio * (permeability / porosity)")
            logger.info("创建特征: 注入效率(injection_efficiency)")
        except Exception as e:
            logger.error(f"计算'injection_efficiency'时出错: {str(e)}")
            nan_features['injection_efficiency'] = -1
        
        # 储层响应指标
        if 'formation_pressure' in df.columns and 'pre_injection_pressure' in df.columns:
            try:
                # 添加小常数避免除零错误
                df['pressure_response_ratio'] = (df['formation_pressure'] - df['pre_injection_pressure']) / (df['injection_intensity_ratio'] + 1e-10)
                check_nan_feature(df, 'pressure_response_ratio', "压力响应比 = (formation_pressure - pre_injection_pressure) / injection_intensity_ratio")
                logger.info("创建特征: 压力响应比(pressure_response_ratio)")
            except Exception as e:
                logger.error(f"计算'pressure_response_ratio'时出错: {str(e)}")
                nan_features['pressure_response_ratio'] = -1

    # 23. 基于达西定律的特征
    if all(col in df.columns for col in ['permeability', 'oil_viscosity', 'well_spacing']):
        try:
            # 流动能力指标
            # 添加小常数避免除零错误
            df['flow_capacity'] = df['permeability'] / ((df['well_spacing'] + 1e-10) * (df['oil_viscosity'] + 1e-10))
            check_nan_feature(df, 'flow_capacity', "流动能力 = permeability / (well_spacing * oil_viscosity)")
            logger.info("创建特征: 流动能力(flow_capacity)")
        except Exception as e:
            logger.error(f"计算'flow_capacity'时出错: {str(e)}")
            nan_features['flow_capacity'] = -1
        
        # 扩散时间尺度
        if 'porosity' in df.columns:
            try:
                # 添加小常数避免除零错误
                df['diffusion_time_scale'] = ((df['well_spacing']**2) * df['porosity'] * df['oil_viscosity']) / (df['permeability'] + 1e-10)
                check_nan_feature(df, 'diffusion_time_scale', "扩散时间尺度 = (well_spacing^2 * porosity * oil_viscosity) / permeability")
                logger.info("创建特征: 扩散时间尺度(diffusion_time_scale)")
            except Exception as e:
                logger.error(f"计算'diffusion_time_scale'时出错: {str(e)}")
                nan_features['diffusion_time_scale'] = -1

    # 24. 增强驱替前缘稳定性特征
    if all(col in df.columns for col in ['mobility_ratio', 'gravity_number']):
        try:
            # 综合稳定性指数，添加小常数避免除零错误
            df['combined_stability_index'] = df['mobility_ratio'] / (1 + df['gravity_number'])
            check_nan_feature(df, 'combined_stability_index', "综合稳定性指数 = mobility_ratio / (1 + gravity_number)")
            logger.info("创建特征: 综合稳定性指数(combined_stability_index)")
        except Exception as e:
            logger.error(f"计算'combined_stability_index'时出错: {str(e)}")
            nan_features['combined_stability_index'] = -1
        
        # 关键特征的非线性变换 - 移除这些特征，因为在NaN报告中被识别为问题特征
        logger.info("跳过创建指进指数非线性变换，因为它们在NaN报告中被识别为有问题的特征")

    # 检查特征中是否有NaN值
    if nan_features:
        # 生成详细的NaN报告
        report_path = generate_nan_report(nan_features, df, feature_creation_log)
        logger.error(f"特征工程过程中发现NaN值，已生成报告: {report_path}")
        
        # 移除包含NaN的特征
        for feature in nan_features.keys():
            if feature in df.columns:
                logger.warning(f"移除包含NaN值的特征: {feature}")
                df = df.drop(columns=[feature])
        
        # 再次检查是否仍有NaN值
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            logger.error(f"移除问题特征后仍有NaN值: {nan_cols}")
            report_path = generate_nan_report(
                {col: df[col].isnull().sum() for col in nan_cols}, 
                df, 
                feature_creation_log
            )
            raise ValueError(f"特征工程过程中发现无法解决的NaN值，请查看报告: {report_path}")
    
    # 检查无穷值
    inf_check = np.isinf(df.values)
    if inf_check.any():
        inf_count = inf_check.sum()
        logger.error(f"特征工程过程中发现{inf_count}个无穷值")
        
        # 获取包含无穷值的列
        inf_columns = {}
        for col in df.columns:
            col_inf_count = np.isinf(df[col].values).sum()
            if col_inf_count > 0:
                inf_columns[col] = col_inf_count
                # 将无穷值替换为NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 移除包含无穷值的特征
        for feature in inf_columns.keys():
            if feature in df.columns:
                logger.warning(f"移除包含无穷值的特征: {feature}")
                df = df.drop(columns=[feature])
        
        # 保存调试数据
        debug_path = os.path.join(PATHS['data_dir'], 'debug_features_with_inf.csv')
        df.to_csv(debug_path, index=False)
        logger.error(f"已将处理后的数据保存到 {debug_path}")
    
    # 特征存储
    cleaned_path = os.path.join(PATHS['data_dir'], 'cleaned_' + PATHS['features_filename'])
    df.to_csv(cleaned_path, index=False)
    logger.info(f"特征工程完成，数据已保存到 {cleaned_path}")
    
    return df

def scale_features(df, method=None):
    """
    特征标准化/归一化
    
    Args:
        df: 特征工程后的数据
        method: 标准化方法，可选 'standard', 'minmax', 'robust'
        
    Returns:
        tuple: (缩放后的数据, 缩放器)
    """
    if method is None:
        method = FEATURE_CONFIG['scaling_method']
    
    # 确定需要缩放的列（数值型特征）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除目标变量
    if DATA_CONFIG['target_column'] in numeric_cols:
        numeric_cols.remove(DATA_CONFIG['target_column'])
    
    logger.info(f"将对{len(numeric_cols)}个数值特征进行{method}缩放")
    
    # 选择缩放器
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:  # 默认使用标准化
        scaler = StandardScaler()
    
    # 应用缩放
    if numeric_cols:
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info("特征缩放完成")
        return df_scaled, scaler
    else:
        logger.warning("没有找到需要缩放的数值特征")
        return df, None

def split_dataset(df, target_column=None, test_size=None, validation_size=None, random_state=None, stratify=False):
    """
    划分数据集为训练集、验证集和测试集，增加了对极少类别的处理
    
    Args:
        df: 数据DataFrame
        target_column: 目标变量列名
        test_size: 测试集比例
        validation_size: 验证集比例
        random_state: 随机种子
        stratify: 是否使用分层抽样
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if target_column is None:
        target_column = DATA_CONFIG['target_column']
    if test_size is None:
        test_size = DATA_CONFIG['test_size']
    if validation_size is None:
        validation_size = DATA_CONFIG['validation_size']
    if random_state is None:
        random_state = DATA_CONFIG['random_state']
    
    if target_column not in df.columns:
        logger.error(f"目标列'{target_column}'不存在于数据集中")
        raise ValueError(f"目标列'{target_column}'不存在于数据集中")
    
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"数据集特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
    
    # 计算验证集实际比例（相对于非测试数据）
    val_size = validation_size / (1 - test_size)
    
    # 检查是否可以使用分层抽样
    use_stratify = False
    if stratify:
        # 创建分层标签
        bins = [0, 0.1, 0.5, float('inf')]
        labels = [0, 1, 2]
        y_strat = pd.cut(y, bins=bins, labels=labels)
        
        # 检查每个类别的样本数量
        class_counts = y_strat.value_counts()
        min_samples = class_counts.min()
        
        if min_samples >= 3:  # 每个类别至少需要3个样本(1个训练,1个验证,1个测试)
            use_stratify = True
            stratify_y = y_strat
            logger.info(f"使用分层抽样，最小类别样本数: {min_samples}")
        else:
            logger.warning(f"无法使用分层抽样，最小类别只有{min_samples}个样本，改用随机抽样")
    
    # 首先分割出测试集
    if use_stratify:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
        )
        
        # 更新分层标签
        temp_bins = [0, 0.1, 0.5, float('inf')]
        temp_labels = [0, 1, 2]
        y_temp_strat = pd.cut(y_temp, bins=temp_bins, labels=temp_labels)
        
        # 再次检查每个类别的样本数量
        class_counts = y_temp_strat.value_counts()
        min_samples = class_counts.min()
        
        if min_samples >= 2:  # 每个类别至少需要2个样本(分为训练和验证)
            # 从剩余数据中分割出验证集，同样使用分层抽样
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp_strat
            )
        else:
            logger.warning(f"剩余数据无法进行分层抽样，最小类别只有{min_samples}个样本，改用随机抽样")
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=random_state
            )
    else:
        # 使用随机抽样
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
    
    logger.info(f"数据集划分完成: 训练集 {len(X_train)}样本, 验证集 {len(X_val)}样本, 测试集 {len(X_test)}样本")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_feature_stats(df):
    """
    计算特征统计信息
    
    Args:
        df: 数据DataFrame
        
    Returns:
        pandas.DataFrame: 特征统计信息
    """
    # 基本统计量
    stats = df.describe().T
    
    # 添加额外统计信息
    stats['missing'] = df.isnull().sum()
    stats['missing_pct'] = df.isnull().mean() * 100
    
    # 计算偏度和峰度
    if len(df) > 3:  # 至少需要3个样本才能计算
        stats['skewness'] = df.skew()
        stats['kurtosis'] = df.kurtosis()
    
    logger.info("已计算特征统计信息")
    return stats

def data_quality_report(df, output_path=None):
    """
    生成数据质量报告
    
    Args:
        df: 数据DataFrame
        output_path: 报告输出路径
        
    Returns:
        dict: 数据质量指标
    """
    if output_path is None:
        output_path = os.path.join(PATHS['results_dir'], 'data_quality_report.csv')
    
    # 数据完整性
    missing_values = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100
    
    # 数据唯一性
    unique_values = df.nunique()
    unique_pct = (unique_values / len(df)) * 100
    
    # 异常值检测（使用IQR方法）
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    outliers_pct = outliers / len(df) * 100
    
    # 汇总报告
    report = pd.DataFrame({
        'missing_values': missing_values,
        'missing_pct': missing_pct,
        'unique_values': unique_values,
        'unique_pct': unique_pct,
        'outliers': outliers,
        'outliers_pct': outliers_pct
    })
    
    # 保存报告
    report.to_csv(output_path)
    logger.info(f"数据质量报告已保存至 {output_path}")
    
    # 返回质量指标摘要
    quality_metrics = {
        'completeness': (1 - missing_values.sum() / (len(df) * len(df.columns))) * 100,
        'outliers_ratio': outliers.sum() / (len(df) * len(df.columns)) * 100,
        'feature_count': len(df.columns),
        'sample_count': len(df)
    }
    
    return quality_metrics
def generate_nan_report(nan_features, df, output_path=None):
    """
    生成NaN值详细报告
    
    Args:
        nan_features: 包含NaN的特征字典 {特征名: NaN数量}
        df: 包含NaN的数据DataFrame
        output_path: 报告输出路径
        
    Returns:
        str: 报告文件路径
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PATHS['results_dir'], f'nan_report_{timestamp}.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CCUS CO2气窜预测系统 - NaN值检测报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体统计
        f.write(f"检测到{len(nan_features)}个特征包含NaN值\n")
        f.write(f"NaN值总数: {sum(nan_features.values())}\n\n")
        
        # 按NaN数量排序特征
        sorted_nan_features = sorted(nan_features.items(), key=lambda x: x[1], reverse=True)
        
        f.write("NaN值详细情况:\n")
        f.write("-" * 80 + "\n")
        
        # 分析每个包含NaN的特征
        for feature, count in sorted_nan_features:
            f.write(f"\n特征: {feature}\n")
            f.write(f"NaN数量: {count}\n")
            
            # 获取包含NaN的样本索引
            nan_indices = df[df[feature].isna()].index.tolist()
            f.write(f"NaN样本索引: {', '.join(map(str, nan_indices[:10]))}")
            if len(nan_indices) > 10:
                f.write(f" 等共{len(nan_indices)}个样本\n")
            else:
                f.write("\n")
            
            # 分析可能的原因
            f.write("可能原因分析:\n")
            
            # 除零错误检查
            if feature in ['injection_intensity_ratio', 'perm_porosity_ratio']:
                if 'porosity' in df.columns:
                    zero_porosity = (df.loc[nan_indices, 'porosity'] == 0).sum()
                    if zero_porosity > 0:
                        f.write(f"  - 有{zero_porosity}个样本的'porosity'为零\n")
            
            if feature in ['gravity_number', 'gravity_viscous_ratio', 'displacement_front_stability']:
                if 'oil_viscosity' in df.columns:
                    zero_viscosity = (df.loc[nan_indices, 'oil_viscosity'] == 0).sum()
                    if zero_viscosity > 0:
                        f.write(f"  - 有{zero_viscosity}个样本的'oil_viscosity'为零\n")
            
            if feature in ['spacing_thickness_ratio']:
                if 'effective_thickness' in df.columns:
                    zero_thickness = (df.loc[nan_indices, 'effective_thickness'] == 0).sum()
                    if zero_thickness > 0:
                        f.write(f"  - 有{zero_thickness}个样本的'effective_thickness'为零\n")
            
            if feature in ['pressure_response_ratio']:
                if 'injection_intensity_ratio' in df.columns:
                    zero_intensity = (df.loc[nan_indices, 'injection_intensity_ratio'] == 0).sum()
                    if zero_intensity > 0:
                        f.write(f"  - 有{zero_intensity}个样本的'injection_intensity_ratio'为零\n")
            
            # 对数变换检查
            if feature.endswith('_log') or feature.endswith('_log_abs'):
                base_feature = feature.replace('_log', '').replace('_log_abs', '')
                if base_feature in df.columns:
                    negative_values = (df.loc[nan_indices, base_feature] <= 0).sum()
                    if negative_values > 0:
                        f.write(f"  - 有{negative_values}个样本的'{base_feature}'小于或等于零\n")
            
            # 无穷值检查
            inf_count = np.isinf(df[feature]).sum()
            if inf_count > 0:
                f.write(f"  - 有{inf_count}个样本的'{feature}'为无穷值\n")
            
            # 依赖特征NaN检查
            if "_interaction" in feature:
                parts = feature.split('_interaction')[0].split('_')
                if len(parts) >= 2:
                    feat1, feat2 = parts[0], '_'.join(parts[1:])
                    if feat1 in df.columns and feat2 in df.columns:
                        nan_feat1 = df.loc[nan_indices, feat1].isna().sum()
                        nan_feat2 = df.loc[nan_indices, feat2].isna().sum()
                        if nan_feat1 > 0:
                            f.write(f"  - 有{nan_feat1}个样本的'{feat1}'为NaN\n")
                        if nan_feat2 > 0:
                            f.write(f"  - 有{nan_feat2}个样本的'{feat2}'为NaN\n")
            
            # 样本值分析
            f.write("\n样本值分析:\n")
            if len(nan_indices) > 0:
                # 选择前5个NaN样本进行详细分析
                sample_indices = nan_indices[:5]
                
                # 对于每个样本，显示相关特征的值
                for idx in sample_indices:
                    f.write(f"  样本 {idx}:\n")
                    
                    # 根据特征类型分析依赖的特征
                    if feature in ['injection_intensity_ratio']:
                        f.write(f"    pv_number = {df.loc[idx, 'pv_number'] if 'pv_number' in df.columns else 'N/A'}\n")
                        f.write(f"    porosity = {df.loc[idx, 'porosity'] if 'porosity' in df.columns else 'N/A'}\n")
                        f.write(f"    effective_thickness = {df.loc[idx, 'effective_thickness'] if 'effective_thickness' in df.columns else 'N/A'}\n")
                    
                    elif feature in ['gravity_number', 'gravity_viscous_ratio']:
                        f.write(f"    oil_density = {df.loc[idx, 'oil_density'] if 'oil_density' in df.columns else 'N/A'}\n")
                        f.write(f"    permeability = {df.loc[idx, 'permeability'] if 'permeability' in df.columns else 'N/A'}\n")
                        f.write(f"    effective_thickness = {df.loc[idx, 'effective_thickness'] if 'effective_thickness' in df.columns else 'N/A'}\n")
                        f.write(f"    oil_viscosity = {df.loc[idx, 'oil_viscosity'] if 'oil_viscosity' in df.columns else 'N/A'}\n")
            else:
                f.write("  无法获取样本详情，可能是特征计算过程中产生NaN\n")
        
        # 总结和建议
        f.write("\n" + "=" * 80 + "\n")
        f.write("总结和建议:\n")
        f.write("-" * 80 + "\n")
        
        # 最常见的问题
        if any('oil_viscosity' in feature for feature in nan_features.keys()):
            f.write("1. 多个特征因'oil_viscosity'为零而产生NaN，建议检查并修正原始数据\n")
        
        if any('porosity' in feature for feature in nan_features.keys()):
            f.write("2. 多个特征因'porosity'为零而产生NaN，建议检查并修正原始数据\n")
        
        if any('_log' in feature for feature in nan_features.keys()):
            f.write("3. 对数变换特征因原特征包含零或负值而产生NaN，建议对原特征进行偏移处理\n")
        
        f.write("\n4. 特征工程前的数据预处理建议:\n")
        f.write("   - 确保所有被除数不为零，可考虑添加小常数(如1e-6)避免除零错误\n")
        f.write("   - 对数变换前检查特征值是否大于零，可考虑使用log1p变换\n")
        f.write("   - 检查特征间是否存在强依赖关系，确保上游特征无NaN\n")
        
        f.write("\n以上是NaN值检测报告，请解决这些问题后再进行模型训练。\n")
    
    print(f"\n[NaN报告] 已生成NaN值报告: {output_path}\n")
    return output_path
def feature_selection(X_train, y_train, X_val=None, X_test=None, method='importance', n_features=30):
    """
    特征选择，减少特征数量
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征（可选）
        X_test: 测试特征（可选）
        method: 特征选择方法，可选 'importance', 'mutual_info', 'rfe', 'pca'
        n_features: 选择的特征数量
        
    Returns:
        tuple: (选择后的训练特征, 验证特征, 测试特征, 特征选择器)
    """
    from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_regression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    
    if method == 'importance':
        # 基于树模型的特征重要性
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 创建特征选择器
        selector = SelectFromModel(
            model, 
            max_features=n_features, 
            threshold=-np.inf,  # 保留前n_features个特征
            prefit=True
        )
        
        logger.info(f"基于随机森林特征重要性选择了{n_features}个特征")
        
    elif method == 'mutual_info':
        # 基于互信息
        mi_scores = mutual_info_regression(X_train, y_train)
        
        # 选择得分最高的n_features个特征
        mi_indices = np.argsort(mi_scores)[-n_features:]
        
        # 创建一个假选择器（仅包含选择逻辑，不是实际的sklearn选择器）
        class MISelector:
            def __init__(self, indices, feature_names=None):
                self.indices = indices
                self.feature_names = feature_names
                
            def transform(self, X):
                if isinstance(X, pd.DataFrame):
                    return X.iloc[:, self.indices]
                return X[:, self.indices]
                
            def get_support(self, indices=False):
                if indices:
                    return self.indices
                mask = np.zeros(X_train.shape[1], dtype=bool)
                mask[self.indices] = True
                return mask
        
        selector = MISelector(mi_indices, X_train.columns if isinstance(X_train, pd.DataFrame) else None)
        
        logger.info(f"基于互信息选择了{n_features}个特征")
        
    elif method == 'rfe':
        # 递归特征消除
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(model, n_features_to_select=n_features, step=0.2)
        selector.fit(X_train, y_train)
        
        logger.info(f"通过递归特征消除选择了{n_features}个特征")
        
    elif method == 'pca':
        # 主成分分析
        selector = PCA(n_components=n_features)
        selector.fit(X_train)
        
        logger.info(f"通过PCA降维到{n_features}个主成分")
    
    else:
        raise ValueError(f"不支持的特征选择方法: {method}")
    
    # 转换数据
    X_train_selected = selector.transform(X_train)
    
    if X_val is not None:
        X_val_selected = selector.transform(X_val)
    else:
        X_val_selected = None
        
    if X_test is not None:
        X_test_selected = selector.transform(X_test)
    else:
        X_test_selected = None
    
    # 如果输入是DataFrame，则保持输出也是DataFrame
    if isinstance(X_train, pd.DataFrame):
        if method != 'pca':  # PCA会创建新特征，无法保持原始列名
            selected_features = X_train.columns[selector.get_support(indices=True)]
            
            X_train_selected = pd.DataFrame(
                X_train_selected, 
                columns=selected_features, 
                index=X_train.index
            )
            
            if X_val_selected is not None:
                X_val_selected = pd.DataFrame(
                    X_val_selected,
                    columns=selected_features,
                    index=X_val.index
                )
                
            if X_test_selected is not None:
                X_test_selected = pd.DataFrame(
                    X_test_selected,
                    columns=selected_features,
                    index=X_test.index
                )
    
    return X_train_selected, X_val_selected, X_test_selected, selector
def create_physics_interaction_features(df):
    """
    创建基于物理原理的特征交互项
    
    Args:
        df: 特征DataFrame
        
    Returns:
        pandas.DataFrame: 增加了交互特征的DataFrame
    """
    # 定义基于物理意义的特征组
    flow_features = ['permeability', 'porosity', 'oil_viscosity']
    geometry_features = ['effective_thickness', 'well_spacing']
    thermodynamic_features = ['formation_temperature', 'formation_pressure']
    injection_features = ['injection_intensity_ratio', 'injection_efficiency']
    
    # 流动特性交互
    if all(feat in df.columns for feat in flow_features):
        # 流动阻力比 (Flow Resistance Ratio)
        df['flow_resistance_ratio'] = (df['oil_viscosity'] / df['permeability']) * df['porosity']
        
        # 流动异质性指数 (Flow Heterogeneity Index)
        # 描述流动路径的不均匀性
        df['flow_heterogeneity_index'] = df['permeability'] * df['porosity'] * df['oil_viscosity']
    
    # 几何特性与流动特性交互
    geo_flow_pairs = [(g, f) for g in geometry_features for f in flow_features]
    for geo, flow in geo_flow_pairs:
        if geo in df.columns and flow in df.columns:
            # 创建物理意义明确的交互特征
            if geo == 'effective_thickness' and flow == 'permeability':
                # 横向流动容量 (Lateral Flow Capacity)
                df['lateral_flow_capacity'] = df[geo] * df[flow]
            
            if geo == 'well_spacing' and flow == 'oil_viscosity':
                # 推进阻力 (Advancement Resistance)
                df['advancement_resistance'] = df[geo] * df[flow]
    
    # 热力学特性与注入特性交互
    thermo_inj_pairs = [(t, i) for t in thermodynamic_features for i in injection_features]
    for thermo, inj in thermo_inj_pairs:
        if thermo in df.columns and inj in df.columns:
            feature_name = f"{thermo}_{inj}_effect"
            df[feature_name] = df[thermo] * df[inj]
    
    return df
def main():
    """数据处理主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(PATHS['log_dir'], PATHS['log_filename'])
    )
    
    # 加载数据
    df = load_data()
    
    # 数据预处理
    df = preprocess_data(df)
    
    # 特征工程
    df = engineer_features(df)
    
    # 特征缩放
    df_scaled, _ = scale_features(df)
    
    # 数据质量报告
    quality_metrics = data_quality_report(df_scaled)
    logger.info(f"数据处理完成: 数据完整性 {quality_metrics['completeness']:.2f}%, 异常值比例 {quality_metrics['outliers_ratio']:.2f}%")
    
    return df_scaled

if __name__ == "__main__":
    main()
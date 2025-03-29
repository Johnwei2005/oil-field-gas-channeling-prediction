#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统配置文件

本文件包含系统的所有关键配置参数，包括数据处理、模型训练、可视化和系统路径等设置。
"""

import os
from datetime import datetime

# 数据处理配置
DATA_CONFIG = {
    'file_path': 'CO2气窜原始表.csv',  # 数据文件路径
    'encoding': 'gbk',                 # 文件编码
    'test_size': 0.2,                  # 测试集比例
    'validation_size': 0.1,            # 验证集比例
    'random_state': 42,                # 随机种子
    'target_column': 'pv_number'       # 预测目标列名 (对应 "pv数")
}

# 特征映射（中文列名到英文）
COLUMN_MAPPING = {
    '区块': 'block',
    '地层温度℃': 'formation_temperature',
    '地层压力mpa': 'formation_pressure',
    '注气前地层压力mpa': 'pre_injection_pressure',
    '压力水平': 'pressure_level',
    '渗透率md': 'permeability',
    '地层原油粘度mpas': 'oil_viscosity',
    '地层原油密度g/cm3': 'oil_density',
    '井组有效厚度m': 'effective_thickness',
    '注气井压裂': 'injection_well_fracturing',
    '井距m': 'well_spacing',
    '孔隙度/%': 'porosity',
    '注入前含油饱和度/%': 'pre_injection_oil_saturation',
    'pv数': 'pv_number'
}

# 物理参数配置
PHYSICS_CONFIG = {
    'co2_viscosity': 0.05,             # CO2粘度 (mPa·s)
    'co2_density': 0.6,                # CO2密度 (g/cm³)
    'gravity': 9.81,                   # 重力加速度 (m/s²)
    'water_viscosity': 1.0,            # 水粘度 (mPa·s)
    'default_compressibility': 4.5e-5, # 默认压缩系数 (1/MPa)
    'critical_temperature': 31.1,      # CO2临界温度 (°C)
    'critical_pressure': 7.38,         # CO2临界压力 (MPa)
    'reference_pressure': 0.1,         # 参考压力 (MPa)
}

# 特征工程配置
FEATURE_CONFIG = {
    # 基础特征列表
    'base_features': [
        'formation_temperature', 'formation_pressure', 'permeability',
        'oil_viscosity', 'oil_density', 'effective_thickness',
        'well_spacing', 'porosity', 'pre_injection_oil_saturation',
        'pre_injection_pressure', 'pressure_level'  
    ],
    
    # 要创建的物理派生特征
    'derived_features': [
        'mobility_ratio',               # 迁移性比
        'gravity_number',               # 重力数
        'fingering_index',              # 指进系数
        'pressure_diffusivity',         # 压力扩散系数
        'phase_transition_index',       # 相态转化指数
        'injection_intensity_ratio',    # 注入强度比
        'displacement_efficiency',      # 驱替效率
        'reservoir_energy',             # 储层能量
        'co2_solubility_factor',        # CO2溶解度因子
        'pressure_difference',          # 压差
    ],
    
    # 分类特征编码方法
    'categorical_encoding': 'one_hot',  # 可选: 'one_hot', 'label', 'target'
    
    # 标准化方法
    'scaling_method': 'standard',       # 可选: 'standard', 'minmax', 'robust'
}

# 机器学习模型配置
# 在 config.py 中的 MODEL_CONFIG 中添加高级集成模型
MODEL_CONFIG = {
    # 默认模型类型 - 更改为高级集成模型
    'default_model': 'advanced_ensemble',
    
    # 可用模型列表
    'available_models': [
        'physics_guided_xgboost',
        'physics_informed_nn',
        'bayesian_ridge',
        'gaussian_process',
        'ensemble',
        'advanced_ensemble',
        'block_ensemble' 
    ],
    
    # XGBoost模型参数 - 扩展搜索范围
    'xgboost_params': {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.2,
        'reg_alpha': 0.2,
        'reg_lambda': 1.0,
        'random_state': 42
    },
    
    # 高级集成模型参数
    'advanced_ensemble_params': {
        'value_threshold': 0.1,  # 低/高值分割阈值
        'use_physics_constraints': True
    },
    'gp_params': {
    'kernel': 'RBF + WhiteKernel',
    'alpha': 1e-6,
    'n_restarts_optimizer': 10
    },
    # 交叉验证设置
    'cv_folds': 5,
    'loocv': True,
    
    # 物理约束参数
    'physics_weight': 0.4,  # 增加物理约束权重
    'physics_regularization': True,
}

# 不确定性量化配置
UNCERTAINTY_CONFIG = {
    'monte_carlo_samples': 1000,       # 蒙特卡洛采样次数
    'confidence_level': 0.95,          # 置信水平
    'prediction_intervals': True,      # 是否计算预测区间
    'bootstrap_samples': 100,          # 自助法样本数
}

# 可视化设置
VIZ_CONFIG = {
    'figure_size': (12, 8),            # 图形尺寸
    'save_format': 'png',              # 保存格式
    'dpi': 300,                        # 分辨率
    'color_map': 'viridis',            # 颜色映射
    'contour_levels': 20,              # 等值线数量
    'plot_grid': True,                 # 是否显示网格
    'font_size': 12,                   # 字体大小
    'title_font_size': 14,             # 标题字体大小
    '3d_plot': True,                   # 是否启用3D绘图
    'interactive': True,               # 是否启用交互式图表
}

# 注入策略优化配置
OPTIMIZATION_CONFIG = {
    'objective': 'minimize_gas_migration',  # 优化目标
    'constraints': ['max_pressure', 'min_oil_recovery'],  # 约束条件
    'max_iterations': 100,             # 最大迭代次数
    'convergence_tolerance': 1e-4,     # 收敛容差
    'algorithm': 'bayesian',           # 优化算法
}

# 系统路径
PATHS = {
    # 基础目录
    'base_dir': os.path.dirname(os.path.abspath(__file__)),
    
    # 输入数据路径
    'data_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
    
    # 模型保存路径
    'model_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
    
    # 结果输出路径
    'results_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
    
    # 日志文件路径
    'log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
    
    # 模型保存文件名
    'model_filename': 'co2_migration_model.pkl',
    
    # 特征数据保存文件名
    'features_filename': 'engineered_features.csv',
    
    # 日志文件名
    'log_filename': f'ccus_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
}

# 创建必要的目录
for dir_path in [PATHS['data_dir'], PATHS['model_dir'], PATHS['results_dir'], PATHS['log_dir']]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 系统版本信息
VERSION = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release_date': '2025-03-21',
    'name': 'CCUS CO2气窜预测系统'
}

# 导出版本字符串
VERSION_STRING = f"{VERSION['name']} v{VERSION['major']}.{VERSION['minor']}.{VERSION['patch']}"
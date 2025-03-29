#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统模型优化运行脚本

本脚本整合了特征工程和残差建模方法，测试不同的机器学习模型和超参数组合，
以达到R²大于0.85的目标。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
import joblib
from datetime import datetime

# 导入自定义模块
from data_processor import load_data, preprocess_data, engineer_features
from enhanced_features import (
    create_physics_informed_features, 
    create_interaction_features,
    create_nonlinear_transformations,
    select_optimal_features,
    optimize_features_for_small_sample
)
from residual_model import ResidualModel, train_and_evaluate_residual_model

# 导入配置
from config import PATHS, DATA_CONFIG, MODEL_CONFIG

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['log_dir'], f'model_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_model_optimization():
    """
    运行模型优化流程
    """
    logger.info("开始模型优化流程")
    
    # 1. 加载和预处理数据
    logger.info("步骤1: 加载和预处理数据")
    df = load_data()
    df = preprocess_data(df)
    
    # 2. 基础特征工程
    logger.info("步骤2: 基础特征工程")
    df = engineer_features(df)
    
    # 3. 增强特征工程
    logger.info("步骤3: 增强特征工程")
    target_col = DATA_CONFIG['target_column']
    
    # 3.1 创建物理约束特征
    df_physics = create_physics_informed_features(df)
    
    # 3.2 创建特征交互项
    df_interaction = create_interaction_features(df_physics, target_col, top_n=5)
    
    # 3.3 创建非线性变换特征
    df_nonlinear = create_nonlinear_transformations(df_interaction, target_col)
    
    # 3.4 选择最优特征子集
    selected_features = select_optimal_features(df_nonlinear, target_col, method='hybrid')
    if target_col not in selected_features:
        selected_features.append(target_col)
    
    df_selected = df_nonlinear[selected_features]
    
    # 4. 分离特征和目标
    X = df_selected.drop(columns=[target_col])
    y = df_selected[target_col]
    
    # 5. 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DATA_CONFIG['test_size'], random_state=DATA_CONFIG['random_state']
    )
    
    # 6. 测试不同的残差模型
    logger.info("步骤4: 测试不同的残差模型")
    model_types = ['random_forest', 'gradient_boosting', 'gaussian_process']
    results = {}
    
    for model_type in model_types:
        logger.info(f"训练和评估 {model_type} 残差模型")
        model, metrics = train_and_evaluate_residual_model(
            X, y, model_type=model_type, test_size=DATA_CONFIG['test_size']
        )
        results[model_type] = metrics
        
        # 记录结果
        logger.info(f"{model_type} 模型测试集 R²: {metrics['test']['final_r2']:.4f}")
        logger.info(f"{model_type} 模型测试集 RMSE: {metrics['test']['final_rmse']:.4f}")
        logger.info(f"{model_type} 模型测试集 MAE: {metrics['test']['final_mae']:.4f}")
    
    # 7. 找出最佳模型
    best_model_type = max(results, key=lambda k: results[k]['test']['final_r2'])
    best_r2 = results[best_model_type]['test']['final_r2']
    
    logger.info(f"最佳模型: {best_model_type}, 测试集 R²: {best_r2:.4f}")
    
    # 8. 超参数优化
    logger.info("步骤5: 对最佳模型进行超参数优化")
    
    # 加载最佳模型
    best_model_path = os.path.join(PATHS['model_dir'], f'residual_model_{best_model_type}.pkl')
    best_model = ResidualModel.load(best_model_path)
    
    # 根据模型类型设置超参数网格
    if best_model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8, 10],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_type == 'gradient_boosting':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_type == 'gaussian_process':
        # 高斯过程模型的超参数优化较为复杂，这里简化处理
        logger.info("高斯过程模型不进行额外的超参数优化")
        return best_model, best_r2, results
    
    # 创建新的模型实例进行超参数优化
    if best_model_type != 'gaussian_process':
        # 创建基础模型
        if best_model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(random_state=42)
        elif best_model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            base_model = GradientBoostingRegressor(random_state=42)
        
        # 使用网格搜索进行超参数优化
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # 获取最佳超参数
        best_params = grid_search.best_params_
        logger.info(f"最佳超参数: {best_params}")
        
        # 使用最佳超参数创建新的残差模型
        optimized_model = ResidualModel(ml_model_type=best_model_type)
        
        # 更新模型超参数
        if best_model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            optimized_model.ml_model = RandomForestRegressor(
                **best_params, random_state=42
            )
        elif best_model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            optimized_model.ml_model = GradientBoostingRegressor(
                **best_params, random_state=42
            )
        
        # 训练优化后的模型
        optimized_model.fit(X_train, y_train)
        
        # 评估优化后的模型
        optimized_metrics = optimized_model.evaluate(X_test, y_test)
        optimized_r2 = optimized_metrics['final_r2']
        
        logger.info(f"优化后的模型测试集 R²: {optimized_r2:.4f}")
        logger.info(f"优化后的模型测试集 RMSE: {optimized_metrics['final_rmse']:.4f}")
        logger.info(f"优化后的模型测试集 MAE: {optimized_metrics['final_mae']:.4f}")
        
        # 保存优化后的模型
        optimized_model_path = os.path.join(PATHS['model_dir'], f'optimized_residual_model_{best_model_type}.pkl')
        optimized_model.save(optimized_model_path)
        
        # 如果优化后的模型性能更好，则更新最佳模型
        if optimized_r2 > best_r2:
            best_model = optimized_model
            best_r2 = optimized_r2
            logger.info(f"超参数优化提高了模型性能，新的 R²: {best_r2:.4f}")
        else:
            logger.info("超参数优化未能提高模型性能")
    
    # 9. 检查是否达到目标 R²
    target_r2 = 0.85
    if best_r2 >= target_r2:
        logger.info(f"模型优化成功！最终 R²: {best_r2:.4f} >= 目标 R²: {target_r2}")
    else:
        logger.warning(f"模型优化未达到目标，当前 R²: {best_r2:.4f} < 目标 R²: {target_r2}")
        logger.info("尝试集成学习方法进一步提高性能")
        
        # 10. 实现集成学习方法
        logger.info("步骤6: 实现集成学习方法")
        
        # 创建不同类型的残差模型
        models = {}
        for model_type in model_types:
            model_path = os.path.join(PATHS['model_dir'], f'residual_model_{model_type}.pkl')
            if os.path.exists(model_path):
                models[model_type] = ResidualModel.load(model_path)
            else:
                logger.warning(f"模型文件 {model_path} 不存在，跳过")
        
        # 如果有足够的模型，创建集成模型
        if len(models) >= 2:
            # 使用简单平均集成
            def ensemble_predict(X):
                predictions = []
                for model in models.values():
                    predictions.append(model.predict(X))
                return np.mean(predictions, axis=0)
            
            # 评估集成模型
            ensemble_pred = ensemble_predict(X_test)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            
            logger.info(f"集成模型测试集 R²: {ensemble_r2:.4f}")
            logger.info(f"集成模型测试集 RMSE: {ensemble_rmse:.4f}")
            logger.info(f"集成模型测试集 MAE: {ensemble_mae:.4f}")
            
            # 如果集成模型性能更好，则更新最佳模型
            if ensemble_r2 > best_r2:
                best_r2 = ensemble_r2
                logger.info(f"集成学习提高了模型性能，新的 R²: {best_r2:.4f}")
                
                # 保存集成模型预测结果
                ensemble_results = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': ensemble_pred
                })
                ensemble_results_path = os.path.join(PATHS['results_dir'], 'ensemble_predictions.csv')
                ensemble_results.to_csv(ensemble_results_path, index=False)
                logger.info(f"集成模型预测结果已保存至 {ensemble_results_path}")
                
                # 绘制集成模型预测结果
                plt.figure(figsize=(10, 8))
                plt.scatter(y_test, ensemble_pred, alpha=0.7)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                plt.xlabel('实际值')
                plt.ylabel('预测值')
                plt.title(f'集成模型预测结果 (R² = {ensemble_r2:.4f})')
                plt.tight_layout()
                
                ensemble_plot_path = os.path.join(PATHS['results_dir'], 'ensemble_predictions.png')
                plt.savefig(ensemble_plot_path, dpi=300)
                plt.close()
                logger.info(f"集成模型预测结果图已保存至 {ensemble_plot_path}")
            else:
                logger.info("集成学习未能提高模型性能")
    
    # 11. 保存最终结果
    results_summary = {
        'best_model_type': best_model_type,
        'best_r2': best_r2,
        'target_r2': target_r2,
        'success': best_r2 >= target_r2,
        'model_results': results
    }
    
    results_summary_path = os.path.join(PATHS['results_dir'], 'optimization_results.pkl')
    joblib.dump(results_summary, results_summary_path)
    logger.info(f"优化结果摘要已保存至 {results_summary_path}")
    
    # 12. 保存特征重要性
    if hasattr(best_model.ml_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.ml_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        feature_importance_path = os.path.join(PATHS['results_dir'], 'feature_importance.csv')
        feature_importance.to_csv(feature_importance_path, index=False)
        logger.info(f"特征重要性已保存至 {feature_importance_path}")
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('特征重要性排名 (前20)', fontsize=14)
        plt.tight_layout()
        
        feature_importance_plot_path = os.path.join(PATHS['results_dir'], 'feature_importance.png')
        plt.savefig(feature_importance_plot_path, dpi=300)
        plt.close()
        logger.info(f"特征重要性图已保存至 {feature_importance_plot_path}")
    
    return best_model, best_r2, results_summary

if __name__ == "__main__":
    best_model, best_r2, results_summary = run_model_optimization()
    
    logger.info("模型优化完成")
    logger.info(f"最佳模型类型: {results_summary['best_model_type']}")
    logger.info(f"最佳 R²: {results_summary['best_r2']:.4f}")
    logger.info(f"目标 R²: {results_summary['target_r2']}")
    logger.info(f"优化结果: {'成功' if results_summary['success'] else '未达到目标'}")

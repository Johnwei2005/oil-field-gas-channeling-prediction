#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统特征优化模块

本模块用于分析和优化特征工程，减少特征数量并提高模型性能。
重点：排除pv_number_original以避免数据泄露。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 从config导入路径配置
from config import PATHS

# 设置输出目录
RESULTS_DIR = PATHS['results_dir']
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def analyze_feature_correlations(X, threshold=0.85):
    """分析特征间的相关性"""
    print("\n=== 步骤1: 特征相关性分析 ===")
    
    # 计算相关矩阵
    corr_matrix = X.corr()
    
    # 可视化相关矩阵热图（仅显示前30个特征）
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix.iloc[:30, :30], dtype=bool))
    sns.heatmap(corr_matrix.iloc[:30, :30], mask=mask, cmap='coolwarm', 
                annot=False, center=0, square=True, linewidths=.5)
    plt.title('特征相关性热图 (前30个特征)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 找出高度相关的特征对
    high_corr_features = set()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]
                # 通常保留原始特征，移除衍生特征
                if any(x in colname_i for x in ['_interaction', '_squared', '_log']):
                    high_corr_features.add(colname_i)
                    high_corr_pairs.append((colname_i, colname_j, corr_matrix.iloc[i, j]))
                elif any(x in colname_j for x in ['_interaction', '_squared', '_log']):
                    high_corr_features.add(colname_j)
                    high_corr_pairs.append((colname_j, colname_i, corr_matrix.iloc[i, j]))
                else:
                    # 两者都是原始特征，移除一个（通常保留第一个出现的）
                    high_corr_features.add(colname_i)
                    high_corr_pairs.append((colname_i, colname_j, corr_matrix.iloc[i, j]))
    
    print(f"发现{len(high_corr_features)}个高度相关的冗余特征 (|r| > {threshold})")
    
    # 列出前10个高相关对
    print("\n高相关特征对示例:")
    for i, (feat1, feat2, corr) in enumerate(sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]):
        print(f"{i+1}. {feat1} 与 {feat2}: r = {corr:.4f}")
    
    return high_corr_features, corr_matrix

def identify_basic_physics_features(X):
    """识别基本物理特征"""
    print("\n=== 步骤2: 识别基本物理特征 ===")
    
    # 基本物理特征列表
    basic_physics_features = [
        'formation_temperature', 'formation_pressure', 
        'permeability', 'oil_viscosity', 'oil_density',
        'effective_thickness', 'well_spacing', 'porosity',
        'pre_injection_oil_saturation', 'pressure_difference',
        'mobility_ratio', 'gravity_number', 'fingering_index',
        'perm_porosity_ratio', 'spacing_thickness_ratio',
        'reservoir_energy', 'displacement_efficiency'
    ]
    
    # 确保这些特征在数据集中存在
    basic_features_available = [f for f in basic_physics_features if f in X.columns]
    print(f"可用的基本物理特征数量: {len(basic_features_available)}")
    print("保留的基本物理特征:")
    for i, feature in enumerate(basic_features_available):
        print(f"{i+1}. {feature}")
    
    return basic_features_available

def evaluate_feature_importance(X, y):
    """评估特征重要性"""
    print("\n=== 步骤3: 特征重要性评估 ===")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 使用随机森林评估特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled_df, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 可视化特征重要性（前20个）
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('特征重要性排名 (前20)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # 打印前20个最重要的特征
    print("\n前20个最重要的特征:")
    for i, (feature, importance) in enumerate(
        zip(feature_importance['Feature'].head(20), feature_importance['Importance'].head(20))
    ):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    return feature_importance

def process_block_features(X, data):
    """处理区块特征"""
    print("\n=== 步骤4: 区块特征处理 ===")
    
    # 提取所有区块特征
    block_features = [col for col in X.columns if col.startswith('block_')]
    
    if len(block_features) == 0:
        print("数据中不存在区块特征，跳过此步骤")
        return []
    
    print(f"原始区块特征数量: {len(block_features)}")
    block_data = X[block_features]
    
    # 检查区块数据是否有足够的变异性
    if block_data.nunique().sum() <= 5:
        print("区块数据变异性不足，使用简单分组替代聚类")
        # 使用简单的分组方法
        cluster_features = []
        
        # 创建国内/国外分组
        is_foreign = np.zeros(len(data))
        for i, col in enumerate(block_features):
            if any(term in col for term in ['EAST', 'FORD', 'SACROC', 'RANGELY', 'SLAUGHTER', 'Basin']):
                is_foreign = is_foreign | block_data[col].values
        
        data['block_cluster_foreign'] = is_foreign
        cluster_features.append('block_cluster_foreign')
        
        # 计算每个样本的区块数
        data['block_count'] = block_data.sum(axis=1)
        cluster_features.append('block_count')
        
        print("\n使用简单分组创建了区块特征:")
        print("1. block_cluster_foreign: 是否为国外区块")
        print("2. block_count: 区块数量")
        
        return cluster_features
    
    # 否则使用K-Means进行聚类
    try:
        # 自适应确定合适的聚类数
        from sklearn.metrics import silhouette_score
        
        max_clusters = min(5, len(block_features))
        best_score = -1
        best_n_clusters = 2  # 默认至少2个聚类
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(block_data)
            
            # 检查是否至少有两个不同的聚类标签
            if len(np.unique(cluster_labels)) < 2:
                continue
                
            try:
                # 计算轮廓系数
                score = silhouette_score(block_data, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except:
                # 如果轮廓系数计算失败，继续尝试下一个聚类数
                continue
        
        print(f"选择的最佳聚类数: {best_n_clusters}, 轮廓系数: {best_score:.4f}")
        
        # 使用最佳聚类数进行聚类
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        block_clusters = kmeans.fit_predict(block_data)
        
        # 创建新的区块类别特征
        cluster_features = []
        for i in range(best_n_clusters):
            feature_name = f'block_cluster_{i}'
            data[feature_name] = (block_clusters == i).astype(int)
            cluster_features.append(feature_name)
        
        # 分析聚类结果
        for i in range(best_n_clusters):
            cluster_indices = np.where(block_clusters == i)[0]
            blocks_in_cluster = []
            
            # 找出该聚类中的区块
            for idx, col in enumerate(block_features):
                # 检查该区块在聚类中的贡献
                if kmeans.cluster_centers_[i, idx] > 0.1:  # 使用聚类中心而不是索引
                    blocks_in_cluster.append(col)
            
            print(f"\n区块聚类{i}包含的区块 ({len(blocks_in_cluster)}个):")
            # 只显示前5个区块和总数
            for j, block in enumerate(blocks_in_cluster[:5]):
                print(f"  {j+1}. {block.replace('block_', '')}")
            if len(blocks_in_cluster) > 5:
                print(f"  ...共{len(blocks_in_cluster)}个区块")
        
        print(f"\n已将{len(block_features)}个区块特征聚类为{best_n_clusters}个区块类别")
        print(f"新的区块类别特征: {', '.join(cluster_features)}")
        
        return cluster_features
    
    except Exception as e:
        print(f"区块聚类失败: {str(e)}")
        print("使用简单分组替代")
        
        # 回退到简单分组
        data['block_count'] = block_data.sum(axis=1)
        print("创建了block_count特征: 区块数量")
        
        return ['block_count']

def generate_optimized_feature_set(X, y, basic_features, importance_df, high_corr_features, block_clusters):
    """生成优化后的特征集"""
    print("\n=== 步骤5: 生成优化后的特征集 ===")
    
    # 初始包含基本物理特征
    final_features = basic_features.copy()
    print(f"基础物理特征数量: {len(final_features)}")
    
    # 添加重要性排名靠前的特征(不重复且不在高相关列表中)
    added_top_features = []
    for feature in importance_df['Feature']:
        if (feature not in final_features and 
            feature not in high_corr_features and 
            not feature.startswith('block_') and
            feature != 'pv_number_original'):  # 排除pv_number_original和原始区块特征
            final_features.append(feature)
            added_top_features.append(feature)
            if len(final_features) >= 20:  # 限制特征总数不超过20个
                break
    
    print(f"添加了{len(added_top_features)}个重要性排名靠前的特征")
    
    # 添加新的区块聚类特征(如果有)
    for feature in block_clusters:
        if feature not in final_features and len(final_features) < 25:
            final_features.append(feature)
    
    if block_clusters:
        print(f"添加了{len(block_clusters)}个区块聚类特征")
    
    # 最终统计
    print(f"\n最终特征数量: {len(final_features)} (原始特征集有{X.shape[1]}个特征)")
    print("\n最终选择的特征:")
    for i, feature in enumerate(final_features):
        print(f"{i+1}. {feature}")
    
    return final_features

def evaluate_feature_set_performance(X, X_final, y):
    """评估新特征集性能"""
    print("\n=== 步骤6: 验证新特征集性能 ===")
    
    # 标准化特征
    scaler = StandardScaler()
    
    # 原始特征集的交叉验证性能
    X_orig_scaled = scaler.fit_transform(X)
    
    models = [
        ('Ridge', Ridge(alpha=1.0)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    
    metrics = {
        'r2': r2_score,
        'mse': mean_squared_error,
        'mae': mean_absolute_error
    }
    
    for name, model in models:
        print(f"\n{name}回归模型评估:")
        
        # 原始特征集性能
        scores_orig = cross_val_score(model, X_orig_scaled, y, cv=5, scoring='r2')
        valid_scores_orig = scores_orig[~np.isnan(scores_orig)]  # 过滤掉NaN值
        
        if len(valid_scores_orig) > 0:
            mean_r2_orig = np.mean(valid_scores_orig)
        else:
            mean_r2_orig = np.nan
        
        # 新特征集的交叉验证性能
        X_final_scaled = scaler.fit_transform(X_final)
        scores_final = cross_val_score(model, X_final_scaled, y, cv=5, scoring='r2')
        valid_scores_final = scores_final[~np.isnan(scores_final)]  # 过滤掉NaN值
        
        if len(valid_scores_final) > 0:
            mean_r2_final = np.mean(valid_scores_final)
        else:
            mean_r2_final = np.nan
        
        print(f"  原始特征集({X.shape[1]}个特征)的平均R²: {mean_r2_orig:.4f}")
        print(f"  新特征集({X_final.shape[1]}个特征)的平均R²: {mean_r2_final:.4f}")
        
        if not np.isnan(mean_r2_final) and not np.isnan(mean_r2_orig):
            change = ((mean_r2_final - mean_r2_orig) / abs(mean_r2_orig)) * 100 if mean_r2_orig != 0 else float('inf')
            print(f"  性能变化: {change:+.2f}%")
        
        # 打印每折R²值
        print("\n  每折R²详情:")
        for i, (orig, final) in enumerate(zip(scores_orig, scores_final)):
            orig_str = f"{orig:.4f}" if not np.isnan(orig) else "NaN"
            final_str = f"{final:.4f}" if not np.isnan(final) else "NaN"
            print(f"    第{i+1}折: 原始特征集 R²={orig_str}, 新特征集 R²={final_str}")
        
        # 补充MSE和MAE评估（仅针对新特征集）
        print("\n  新特征集其他评估指标:")
        mse_scores = cross_val_score(model, X_final_scaled, y, cv=5, scoring='neg_mean_squared_error')
        mae_scores = cross_val_score(model, X_final_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        
        print(f"    MSE: {-np.mean(mse_scores):.4f}±{np.std(mse_scores):.4f}")
        print(f"    MAE: {-np.mean(mae_scores):.4f}±{np.std(mae_scores):.4f}")

def save_optimized_features(data, final_features, output_path=None):
    """保存优化后的特征集"""
    if output_path is None:
        output_path = os.path.join(PATHS['data_dir'], 'optimized_features.csv')
    
    # 确保目标变量包含在内
    if 'pv_number' not in final_features and 'pv_number' in data.columns:
        final_features = final_features + ['pv_number']
    
    # 保存优化后的特征集
    data[final_features].to_csv(output_path, index=False)
    print(f"\n优化后的特征集已保存到: {output_path}")
    
    return output_path

def main():
    """主函数"""
    print("=== CCUS CO2气窜预测系统特征优化 (无数据泄露版) ===")
    
    # 加载数据
    data_path = os.path.join(PATHS['data_dir'], 'cleaned_engineered_features.csv')
    data = pd.read_csv(data_path)
    
    print(f"加载数据: {data_path}")
    print(f"样本数量: {data.shape[0]}, 特征数量: {data.shape[1]}")
    
    # 明确排除pv_number_original
    excluded_features = ['pv_number_original']
    if 'pv_number_original' in data.columns:
        print(f"明确排除特征 'pv_number_original' 以避免数据泄露")
        data = data.drop(columns=excluded_features)
    
    # 分离目标变量
    target_column = 'pv_number'  # 现在使用确定的目标列
    
    X = data.drop(columns=[target_column] if target_column in data.columns else [])
    y = data[target_column] if target_column in data.columns else None
    
    print(f"目标变量: {target_column}, 特征数量: {X.shape[1]}")
    
    # 步骤1: 特征相关性分析
    high_corr_features, corr_matrix = analyze_feature_correlations(X)
    
    # 步骤2: 识别基本物理特征
    basic_features = identify_basic_physics_features(X)
    
    # 步骤3: 特征重要性评估
    feature_importance = evaluate_feature_importance(X, y)
    
    # 步骤4: 区块特征处理
    block_clusters = process_block_features(X, data)
    
    # 步骤5: 生成优化后的特征集
    final_features = generate_optimized_feature_set(
        X, y, basic_features, feature_importance, high_corr_features, block_clusters)
    
    # 创建最终特征数据集
    X_final = data[final_features]
    
    # 步骤6: 验证新特征集性能
    evaluate_feature_set_performance(X, X_final, y)
    
    # 保存优化后的特征集
    save_optimized_features(data, final_features)
    
    print("\n特征优化分析完成!")

if __name__ == "__main__":
    main()
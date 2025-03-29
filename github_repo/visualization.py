#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统可视化模块

本模块负责生成各种可视化图表，帮助用户理解CO2气窜预测结果。
"""

import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib as mpl
# 导入配置
from config import VIZ_CONFIG, PATHS
from sklearn.metrics import mean_squared_error, r2_score
# 设置日志
logger = logging.getLogger(__name__)

# 设置Matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# 自定义颜色映射
RISK_CMAP = LinearSegmentedColormap.from_list('risk_cmap', ['#2ca02c', '#ffbb00', '#d62728'])
# 设置Matplotlib中文字体支持
# 配置支持中文的字体
try:
    # 尝试使用SimHei字体(黑体)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # 测试是否支持中文
    mpl.font_manager.findfont('SimHei')
    print("成功加载中文字体: SimHei")
except:
    try:
        # 尝试使用微软雅黑
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
        mpl.font_manager.findfont('Microsoft YaHei')
        print("成功加载中文字体: Microsoft YaHei")
    except:
        print("警告：未找到支持中文的字体，图表中的中文可能无法正常显示")
class Visualizer:
    """
    CO2气窜预测可视化器
    
    生成各种可视化图表，包括气窜分布图、参数敏感性图、预测误差分析等。
    """
    
    def __init__(self, figure_size=None, save_format=None, dpi=None, color_map=None):
        """
        初始化可视化器
        
        Args:
            figure_size: 图形尺寸
            save_format: 保存格式
            dpi: 分辨率
            color_map: 颜色映射
        """
        self.figure_size = figure_size if figure_size else VIZ_CONFIG['figure_size']
        self.save_format = save_format if save_format else VIZ_CONFIG['save_format']
        self.dpi = dpi if dpi else VIZ_CONFIG['dpi']
        self.color_map = color_map if color_map else VIZ_CONFIG['color_map']
        
        # 创建结果目录（如果不存在）
        if not os.path.exists(PATHS['results_dir']):
            os.makedirs(PATHS['results_dir'])
    
    def save_figure(self, fig, filename):
        """
        保存图形
        
        Args:
            fig: 图形对象
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(PATHS['results_dir'], f"{filename}.{self.save_format}")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"图形已保存至 {filepath}")
        return filepath
    
    def plot_feature_importance(self, features, importance, title="特征重要性"):
        """
        绘制特征重要性图
        
        Args:
            features: 特征名称列表
            importance: 特征重要性值列表
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 确保输入数据是可迭代的
        if not hasattr(features, '__iter__'):
            features = [features]
        if not hasattr(importance, '__iter__'):
            importance = [importance]
        
        # 排序
        sorted_idx = np.argsort(importance)
        pos = np.arange(len(sorted_idx)) + 0.5
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 绘制条形图
        barh = ax.barh(pos, [importance[i] for i in sorted_idx], align='center', color='#1f77b4')
        
        # 设置Y轴标签
        ax.set_yticks(pos)
        ax.set_yticklabels([features[i] for i in sorted_idx])
        
        # 添加数值标签
        for i, bar in enumerate(barh):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                    ha='left', va='center', fontsize=9)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        ax.set_xlabel('重要性', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置网格线
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        
        # 移除上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    def plot_prediction_vs_actual(self, y_true, y_pred, title="预测值与实际值对比"):
        """
        绘制预测值与实际值对比图
        
        Args:
            y_true: 实际值
            y_pred: 预测值
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 计算最小值和最大值以设置轴范围
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        
        # 计算误差指标
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        # 绘制散点图
        ax.scatter(y_true, y_pred, alpha=0.7, color='#1f77b4')
        
        # 绘制对角线（完美预测线）
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')
        
        # 设置轴标签和标题
        ax.set_xlabel('实际值', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('预测值', fontsize=VIZ_CONFIG['font_size'])
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        # 添加误差指标文本
        text = f'RMSE: {rmse:.4f}\nR²: {r2:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 设置轴范围
        padding = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)
        
        # 设置轴刻度相等
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        return fig
    
    def plot_residuals(self, y_true, y_pred, title="残差分析"):
        """
        绘制残差分析图
        
        Args:
            y_true: 实际值
            y_pred: 预测值
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 计算残差
        residuals = y_true - y_pred
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # 残差散点图
        ax1.scatter(y_pred, residuals, alpha=0.7, color='#1f77b4')
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('预测值', fontsize=VIZ_CONFIG['font_size'])
        ax1.set_ylabel('残差', fontsize=VIZ_CONFIG['font_size'])
        ax1.set_title('残差散点图', fontsize=VIZ_CONFIG['font_size'])
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 残差直方图
        ax2.hist(residuals, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('残差', fontsize=VIZ_CONFIG['font_size'])
        ax2.set_ylabel('频数', fontsize=VIZ_CONFIG['font_size'])
        ax2.set_title('残差分布', fontsize=VIZ_CONFIG['font_size'])
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 计算统计指标
        mean_residuals = np.mean(residuals)
        std_residuals = np.std(residuals)
        
        # 添加统计指标文本
        text = f'均值: {mean_residuals:.4f}\n标准差: {std_residuals:.4f}'
        ax2.text(0.95, 0.95, text, transform=ax2.transAxes, fontsize=10,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 设置整体标题
        fig.suptitle(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_co2_migration_map(self, x_coords, y_coords, migration_values, 
                             well_locations=None, title="CO2气窜分布图"):
        """
        绘制CO2气窜分布图
        
        Args:
            x_coords: X坐标
            y_coords: Y坐标
            migration_values: 气窜值
            well_locations: 井位置，格式为[(x1, y1, 'name1'), (x2, y2, 'name2'), ...]
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 创建等值线图
        contour = ax.tricontourf(x_coords, y_coords, migration_values, 
                                levels=VIZ_CONFIG['contour_levels'], 
                                cmap=self.color_map)
        
        # 添加等值线
        contour_lines = ax.tricontour(x_coords, y_coords, migration_values, 
                                    levels=10, colors='k', linewidths=0.5, alpha=0.5)
        
        # 添加等值线标签
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # 添加井位置
        if well_locations:
            well_x = [loc[0] for loc in well_locations]
            well_y = [loc[1] for loc in well_locations]
            well_names = [loc[2] for loc in well_locations]
            
            ax.scatter(well_x, well_y, marker='^', color='black', s=100, label='井位置')
            
            # 添加井名称
            for i, name in enumerate(well_names):
                ax.annotate(name, (well_x[i], well_y[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        # 添加颜色条
        cbar = fig.colorbar(contour, ax=ax, pad=0.01)
        cbar.set_label('气窜强度', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置标题和标签
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        ax.set_xlabel('X坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('Y坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置网格
        ax.grid(VIZ_CONFIG['plot_grid'], linestyle='--', alpha=0.4)
        
        # 设置轴刻度相等
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        return fig
    
    def plot_3d_migration(self, x_coords, y_coords, migration_values, 
                        title="CO2气窜3D分布"):
        """
        绘制CO2气窜3D分布图
        
        Args:
            x_coords: X坐标
            y_coords: Y坐标
            migration_values: 气窜值
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        if not VIZ_CONFIG['3d_plot']:
            logger.warning("3D绘图功能已禁用，请在配置中启用")
            return None
        
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建3D表面图
        surf = ax.plot_trisurf(x_coords, y_coords, migration_values, 
                             cmap=self.color_map, linewidth=0.1, antialiased=True)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        ax.set_xlabel('X坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('Y坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        ax.set_zlabel('气窜强度', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置视角
        ax.view_init(elev=30, azim=225)
        
        plt.tight_layout()
        
        return fig
    
    def plot_parameter_sensitivity(self, parameter_range, predictions, 
                                parameter_name, baseline=None, 
                                title=None):
        """
        绘制参数敏感性图
        
        Args:
            parameter_range: 参数值范围
            predictions: 对应的预测值
            parameter_name: 参数名称
            baseline: 基准值（如有）
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 默认标题
        if title is None:
            title = f"参数敏感性: {parameter_name}"
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 绘制敏感性曲线
        ax.plot(parameter_range, predictions, 'o-', linewidth=2, color='#1f77b4')
        
        # 如果有基准值，则添加基准线
        if baseline is not None:
            ax.axhline(y=baseline, color='r', linestyle='--', label='基准值')
            ax.legend(loc='best')
        
        # 计算敏感度指标（最大变化/平均值）
        sensitivity = (np.max(predictions) - np.min(predictions)) / np.mean(predictions)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        ax.set_xlabel(parameter_name, fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('预测值', fontsize=VIZ_CONFIG['font_size'])
        
        # 添加敏感度指标文本
        text = f'敏感度: {sensitivity:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        return fig
    
    def plot_learning_curve(self, train_sizes, train_scores, test_scores, 
                          title="学习曲线"):
        """
        绘制学习曲线
        
        Args:
            train_sizes: 训练集大小列表
            train_scores: 训练集得分列表（可以是每个训练集大小的多次交叉验证得分）
            test_scores: 测试集得分列表（可以是每个训练集大小的多次交叉验证得分）
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 计算训练集和测试集得分的均值和标准差
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # 绘制学习曲线
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集得分')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='blue')
        
        ax.plot(train_sizes, test_mean, 'o-', color='green', label='验证集得分')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                       alpha=0.1, color='green')
        
        # 设置标题和标签
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        ax.set_xlabel('训练样本数量', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('得分', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置图例
        ax.legend(loc='best')
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_matrix(self, df, method='pearson', title="特征相关性矩阵"):
        """
        绘制特征相关性矩阵
        
        Args:
            df: 数据DataFrame
            method: 相关系数计算方法，'pearson', 'kendall', 'spearman'之一
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 计算相关系数矩阵
        corr_matrix = df.corr(method=method)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        
        # 设置标题
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        # 调整标签字体大小
        ax.tick_params(axis='both', which='major', labelsize=VIZ_CONFIG['font_size'] - 2)
        
        plt.tight_layout()
        
        return fig
    
    def plot_risk_heatmap(self, x, y, risk_values, well_locations=None, 
                        title="气窜风险热力图"):
        """
        绘制气窜风险热力图
        
        Args:
            x: X坐标网格
            y: Y坐标网格
            risk_values: 风险值矩阵
            well_locations: 井位置列表
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 创建热力图
        im = ax.pcolormesh(x, y, risk_values, cmap=RISK_CMAP, shading='auto')
        
        # 添加等值线
        contour = ax.contour(x, y, risk_values, levels=5, colors='black', linewidths=0.5, alpha=0.7)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        
        # 添加井位置
        if well_locations:
            well_x = [loc[0] for loc in well_locations]
            well_y = [loc[1] for loc in well_locations]
            well_names = [loc[2] for loc in well_locations]
            
            ax.scatter(well_x, well_y, marker='^', color='black', s=100, label='井位置')
            
            # 添加井名称
            for i, name in enumerate(well_names):
                ax.annotate(name, (well_x[i], well_y[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('风险等级', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置风险等级标签
        ticks = np.linspace(np.min(risk_values), np.max(risk_values), 4)
        labels = ['低风险', '中低风险', '中高风险', '高风险']
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        ax.set_xlabel('X坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('Y坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置轴刻度相等
        ax.set_aspect('equal')
        
        # 设置网格
        ax.grid(VIZ_CONFIG['plot_grid'], linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        
        return fig
    
    def plot_uncertainty_map(self, x, y, mean_values, std_values, 
                           title="预测不确定性分布图"):
        """
        绘制预测不确定性分布图
        
        Args:
            x: X坐标网格
            y: Y坐标网格
            mean_values: 预测均值矩阵
            std_values: 预测标准差矩阵
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 创建子图
        fig = plt.figure(figsize=(self.figure_size[0] * 2, self.figure_size[1]))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        
        # 预测均值图
        ax1 = plt.subplot(gs[0])
        im1 = ax1.pcolormesh(x, y, mean_values, cmap='viridis', shading='auto')
        contour1 = ax1.contour(x, y, mean_values, levels=5, colors='black', linewidths=0.5, alpha=0.7)
        ax1.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')
        ax1.set_title('预测均值', fontsize=VIZ_CONFIG['font_size'])
        ax1.set_xlabel('X坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        ax1.set_ylabel('Y坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        ax1.set_aspect('equal')
        ax1.grid(VIZ_CONFIG['plot_grid'], linestyle='--', alpha=0.4)
        
        # 预测标准差图
        ax2 = plt.subplot(gs[1], sharey=ax1)
        im2 = ax2.pcolormesh(x, y, std_values, cmap='plasma', shading='auto')
        contour2 = ax2.contour(x, y, std_values, levels=5, colors='black', linewidths=0.5, alpha=0.7)
        ax2.clabel(contour2, inline=True, fontsize=8, fmt='%.2f')
        ax2.set_title('预测标准差', fontsize=VIZ_CONFIG['font_size'])
        ax2.set_xlabel('X坐标 (m)', fontsize=VIZ_CONFIG['font_size'])
        ax2.set_aspect('equal')
        ax2.grid(VIZ_CONFIG['plot_grid'], linestyle='--', alpha=0.4)
        
        # 添加颜色条
        cax1 = plt.subplot(gs[2])
        cbar1 = fig.colorbar(im2, cax=cax1)
        cbar1.set_label('标准差', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置全局标题
        fig.suptitle(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_optimization_results(self, iterations, objective_values, best_params, 
                               param_history=None, title="注入优化结果"):
        """
        绘制注入参数优化结果
        
        Args:
            iterations: 迭代次数列表
            objective_values: 目标函数值列表
            best_params: 最优参数字典
            param_history: 参数历史字典，格式为{'param_name': [values]}
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 确定子图数量
        n_params = len(best_params) if param_history else 1
        
        # 创建子图
        fig, axes = plt.subplots(n_params + 1, 1, figsize=(self.figure_size[0], self.figure_size[1] * (n_params + 1) / 2))
        
        # 确保axes是可迭代的
        if n_params == 0:
            axes = [axes]
        
        # 绘制目标函数收敛曲线
        ax = axes[0]
        ax.plot(iterations, objective_values, 'o-', color='#1f77b4')
        ax.set_title('目标函数收敛曲线', fontsize=VIZ_CONFIG['font_size'])
        ax.set_xlabel('迭代次数', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('目标函数值', fontsize=VIZ_CONFIG['font_size'])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 添加最优值标记
        best_iter = np.argmin(objective_values)
        best_value = objective_values[best_iter]
        ax.plot(iterations[best_iter], best_value, 'ro', markersize=8, label=f'最优值: {best_value:.4f}')
        ax.legend(loc='best')
        
                    # 如果有参数历史，则绘制参数收敛曲线
        if param_history:
            for i, (param_name, param_values) in enumerate(param_history.items(), 1):
                ax = axes[i]
                ax.plot(iterations, param_values, 'o-', color='#2ca02c')
                ax.set_title(f'{param_name}参数收敛曲线', fontsize=VIZ_CONFIG['font_size'])
                ax.set_xlabel('迭代次数', fontsize=VIZ_CONFIG['font_size'])
                ax.set_ylabel(param_name, fontsize=VIZ_CONFIG['font_size'])
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # 添加最优值标记
                best_value = param_values[best_iter]
                ax.plot(iterations[best_iter], best_value, 'ro', markersize=8, 
                       label=f'最优值: {best_value:.4f}')
                ax.legend(loc='best')
        
        # 设置全局标题
        fig.suptitle(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        return fig
    
    def plot_dashboard(self, data, predictions, feature_importance=None, uncertainty=None, title="预测仪表盘"):
        """
        绘制增强版综合仪表盘，展示模型性能和特征影响
        
        Args:
            data: 特征数据
            predictions: 预测结果字典
            feature_importance: 特征重要性字典
            uncertainty: 不确定性结果
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 创建更大的仪表盘布局
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 子图1: 预测结果对比
        ax1 = fig.add_subplot(gs[0, 0:2])
        if 'actual' in predictions:
            x_range = np.arange(len(predictions['actual']))
            
            # 绘制实际值
            ax1.scatter(x_range, predictions['actual'], color='green', marker='o', label='实际值', s=60, alpha=0.7)
            
            # 绘制预测值
            ax1.scatter(x_range, predictions['mean'], color='blue', marker='x', label='预测值', s=60)
            
            # 绘制预测区间
            if 'lower_bound' in predictions and 'upper_bound' in predictions:
                ax1.fill_between(x_range, predictions['lower_bound'], predictions['upper_bound'], 
                            color='blue', alpha=0.2, label='95%预测区间')
            
            # 计算并显示性能指标
            r2 = r2_score(predictions['actual'], predictions['mean'])
            rmse = np.sqrt(mean_squared_error(predictions['actual'], predictions['mean']))
            
            # 添加性能指标文本
            metrics_text = f"R² = {r2:.4f}\nRMSE = {rmse:.4f}"
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax1.set_xlabel('样本索引', fontsize=12)
            ax1.set_ylabel('PV数值', fontsize=12)
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.6)
        else:
            ax1.text(0.5, 0.5, '无实际值用于比较', ha='center', va='center', fontsize=12)
        
        ax1.set_title('预测结果对比', fontsize=14)
        
        # 子图2: 特征重要性（TOP 10）
        ax2 = fig.add_subplot(gs[0, 2])
        if feature_importance:
            # 获取前10个最重要的特征
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features = [item[0] for item in sorted_features]
            importance = [item[1] for item in sorted_features]
            
            # 创建水平条形图
            y_pos = np.arange(len(features))
            bars = ax2.barh(y_pos, importance, align='center', color='#1f77b4')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features])  # 截断长名称
            ax2.invert_yaxis()  # 最重要的特征在顶部
            
            # 添加标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
        else:
            ax2.text(0.5, 0.5, '特征重要性数据不可用', ha='center', va='center', fontsize=12)
        
        ax2.set_title('特征重要性 (TOP 10)', fontsize=14)
        ax2.set_xlabel('重要性', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6, axis='x')
        
        # 子图3: 预测误差分析
        ax3 = fig.add_subplot(gs[1, 0])
        if 'actual' in predictions:
            # 计算误差
            errors = predictions['actual'] - predictions['mean']
            
            # 绘制误差散点图
            ax3.scatter(predictions['mean'], errors, color='red', alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # 添加趋势线
            try:
                z = np.polyfit(predictions['mean'], errors, 1)
                p = np.poly1d(z)
                ax3.plot(predictions['mean'], p(predictions['mean']), "r--", alpha=0.7)
            except:
                pass
            
            ax3.set_xlabel('预测值', fontsize=12)
            ax3.set_ylabel('误差 (实际值 - 预测值)', fontsize=12)
        else:
            ax3.text(0.5, 0.5, '无误差数据', ha='center', va='center', fontsize=12)
        
        ax3.set_title('预测误差分析', fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # 子图4: 预测值与实际值的散点图
        ax4 = fig.add_subplot(gs[1, 1])
        if 'actual' in predictions:
            # 绘制散点图
            ax4.scatter(predictions['actual'], predictions['mean'], alpha=0.7)
            
            # 添加对角线
            min_val = min(min(predictions['actual']), min(predictions['mean']))
            max_val = max(max(predictions['actual']), max(predictions['mean']))
            padding = (max_val - min_val) * 0.05
            ax4.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 'r--')
            
            ax4.set_xlabel('实际值', fontsize=12)
            ax4.set_ylabel('预测值', fontsize=12)
            
            # 设置相同的轴范围
            ax4.set_xlim(min_val-padding, max_val+padding)
            ax4.set_ylim(min_val-padding, max_val+padding)
            ax4.set_aspect('equal')
        else:
            ax4.text(0.5, 0.5, '无实际值用于比较', ha='center', va='center', fontsize=12)
        
        ax4.set_title('预测值 vs 实际值', fontsize=14)
        ax4.grid(True, linestyle='--', alpha=0.6)
        
        # 子图5: 误差分布直方图
        ax5 = fig.add_subplot(gs[1, 2])
        if 'actual' in predictions:
            # 绘制误差直方图
            errors = predictions['actual'] - predictions['mean']
            ax5.hist(errors, bins=10, color='#1f77b4', alpha=0.7, edgecolor='black')
            ax5.axvline(x=0, color='red', linestyle='--')
            
            # 添加误差统计信息
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            stats_text = f"平均误差: {mean_error:.4f}\n标准差: {std_error:.4f}"
            ax5.text(0.95, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax5.set_xlabel('预测误差', fontsize=12)
            ax5.set_ylabel('频数', fontsize=12)
        else:
            ax5.text(0.5, 0.5, '无误差数据', ha='center', va='center', fontsize=12)
        
        ax5.set_title('误差分布', fontsize=14)
        ax5.grid(True, linestyle='--', alpha=0.6)
        
        # 子图6: 不确定性可视化
        ax6 = fig.add_subplot(gs[2, 0])
        if uncertainty and 'std' in predictions:
            # 创建不确定性热图
            sc = ax6.scatter(range(len(predictions['std'])), predictions['mean'], 
                            c=predictions['std'], cmap='plasma', alpha=0.8, s=60)
            
            # 添加颜色条
            cbar = fig.colorbar(sc, ax=ax6)
            cbar.set_label('预测标准差', fontsize=10)
            
            ax6.set_xlabel('样本索引', fontsize=12)
            ax6.set_ylabel('预测值', fontsize=12)
        else:
            ax6.text(0.5, 0.5, '不确定性数据不可用', ha='center', va='center', fontsize=12)
        
        ax6.set_title('预测不确定性', fontsize=14)
        ax6.grid(True, linestyle='--', alpha=0.6)
        
        # 子图7: 模型拟合质量评估
        ax7 = fig.add_subplot(gs[2, 1:])
        if 'actual' in predictions:
            # 计算残差
            residuals = predictions['actual'] - predictions['mean']
            
            # 计算标准化残差
            if np.std(residuals) > 0:
                std_residuals = residuals / np.std(residuals)
                
                # 创建QQ图
                from scipy import stats
                (osm, osr), (slope, intercept, r) = stats.probplot(std_residuals, dist="norm", plot=None)
                ax7.scatter(osm, osr, color='blue', alpha=0.7)
                ax7.plot(osm, osm * slope + intercept, 'r-')
                
                ax7.set_xlabel('理论分位数', fontsize=12)
                ax7.set_ylabel('样本分位数', fontsize=12)
                
                # 添加正态性检验结果
                k2, p = stats.normaltest(std_residuals)
                normality_text = f"正态性检验: p-value = {p:.4f}"
                ax7.text(0.05, 0.95, normality_text, transform=ax7.transAxes, fontsize=10,
                        va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax7.text(0.5, 0.5, '残差标准差为零，无法创建QQ图', ha='center', va='center', fontsize=12)
        else:
            ax7.text(0.5, 0.5, '无实际值用于比较', ha='center', va='center', fontsize=12)
        
        ax7.set_title('残差正态性QQ图', fontsize=14)
        ax7.grid(True, linestyle='--', alpha=0.6)
        
        # 设置全局标题
        fig.suptitle(title, fontsize=16, weight='bold')
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.94)
        
        return fig
    
    def plot_cross_validation_results(self, cv_results, title="交叉验证结果"):
        """
        绘制交叉验证结果
        
        Args:
            cv_results: 交叉验证结果字典，包含均值和标准差
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 提取指标
        metrics = [key.split('_')[0] for key in cv_results.keys() if key.endswith('_mean')]
        means = [cv_results[f'{metric}_mean'] for metric in metrics]
        stds = [cv_results[f'{metric}_std'] for metric in metrics]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 绘制条形图
        x = np.arange(len(metrics))
        width = 0.6
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.7, color='#1f77b4')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                   f'{means[i]:.4f}±{stds[i]:.4f}',
                   ha='center', va='bottom', fontsize=9, rotation=0)
        
        # 设置轴标签
        ax.set_ylabel('得分', fontsize=VIZ_CONFIG['font_size'])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        
        # 设置标题
        ax.set_title(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def plot_injection_strategy(self, time_points, injection_rates, pressure_values=None, 
                              migration_values=None, title="注入策略与响应"):
        """
        绘制注入策略和响应曲线
        
        Args:
            time_points: 时间点列表
            injection_rates: 注入速率列表
            pressure_values: 压力值列表（可选）
            migration_values: 气窜值列表（可选）
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 确定子图数量
        n_plots = 1 + (pressure_values is not None) + (migration_values is not None)
        
        # 创建图形
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figure_size[0], self.figure_size[1] * n_plots / 2), 
                               sharex=True)
        
        # 确保axes是列表
        if n_plots == 1:
            axes = [axes]
        
        # 子图1: 注入速率
        ax = axes[0]
        ax.plot(time_points, injection_rates, 'o-', color='#1f77b4', linewidth=2)
        ax.set_ylabel('注入速率', fontsize=VIZ_CONFIG['font_size'])
        ax.set_title('CO2注入策略', fontsize=VIZ_CONFIG['font_size'])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 子图2: 压力响应（如果有）
        if pressure_values is not None:
            ax = axes[1]
            ax.plot(time_points, pressure_values, 's-', color='#ff7f0e', linewidth=2)
            ax.set_ylabel('压力响应', fontsize=VIZ_CONFIG['font_size'])
            ax.set_title('储层压力响应', fontsize=VIZ_CONFIG['font_size'])
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # 子图3: 气窜响应（如果有）
        if migration_values is not None:
            ax = axes[-1]
            ax.plot(time_points, migration_values, '^-', color='#d62728', linewidth=2)
            ax.set_ylabel('气窜指数', fontsize=VIZ_CONFIG['font_size'])
            ax.set_title('CO2气窜响应', fontsize=VIZ_CONFIG['font_size'])
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # 设置X轴标签
        axes[-1].set_xlabel('时间', fontsize=VIZ_CONFIG['font_size'])
        
        # 设置全局标题
        fig.suptitle(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
    
    def create_comparison_report(self, models_data, metrics=None, title="模型比较报告"):
        """
        创建模型比较报告
        
        Args:
            models_data: 模型数据字典，格式为{'model_name': {'predictions': [...], 'actual': [...], 'metrics': {...}}}
            metrics: 要比较的指标列表，如果为None，则使用所有可用指标
            title: 图形标题
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 收集所有模型的指标
        all_metrics = {}
        for model_name, model_data in models_data.items():
            if 'metrics' in model_data:
                for metric_name, metric_value in model_data['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = {}
                    all_metrics[metric_name][model_name] = metric_value
        
        # 如果未指定要比较的指标，则使用所有可用指标
        if metrics is None:
            metrics = list(all_metrics.keys())
        else:
            # 过滤掉不可用的指标
            metrics = [m for m in metrics if m in all_metrics]
        
        # 创建图形
        n_metrics = len(metrics)
        fig_height = max(self.figure_size[1], n_metrics * 2)
        fig = plt.figure(figsize=(self.figure_size[0] * 2, fig_height))
        
        # 创建网格布局
        gs = gridspec.GridSpec(n_metrics, 2, figure=fig)
        
        # 绘制每个指标的条形图
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[i, 0])
            
            # 提取该指标的所有模型值
            model_names = list(all_metrics[metric].keys())
            metric_values = [all_metrics[metric][model] for model in model_names]
            
            # 排序（从大到小或从小到小，取决于指标类型）
            # 这里假设对于'r2'等指标是越大越好，对于'rmse'等指标是越小越好
            if metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'auc']:
                # 越大越好的指标，降序排序
                sorted_indices = np.argsort(metric_values)[::-1]
            else:
                # 越小越好的指标，升序排序
                sorted_indices = np.argsort(metric_values)
            
            sorted_models = [model_names[i] for i in sorted_indices]
            sorted_values = [metric_values[i] for i in sorted_indices]
            
            # 绘制条形图
            bars = ax.barh(range(len(sorted_models)), sorted_values, align='center', alpha=0.7, color='#1f77b4')
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                       va='center', fontsize=9)
            
            # 设置标签
            ax.set_yticks(range(len(sorted_models)))
            ax.set_yticklabels(sorted_models)
            ax.set_xlabel(metric, fontsize=VIZ_CONFIG['font_size'])
            ax.set_title(f'{metric}指标比较', fontsize=VIZ_CONFIG['font_size'])
            ax.grid(True, linestyle='--', alpha=0.6, axis='x')
        
        # 绘制所有模型的预测vs实际散点图
        ax = fig.add_subplot(gs[:, 1])
        
        # 为每个模型选择不同的颜色和标记
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p', 'h']
        
        # 绘制每个模型的预测vs实际散点图
        for i, (model_name, model_data) in enumerate(models_data.items()):
            if 'predictions' in model_data and 'actual' in model_data:
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                ax.scatter(model_data['actual'], model_data['predictions'], 
                          alpha=0.7, color=color, marker=marker, label=model_name)
        
        # 添加对角线
        if 'actual' in next(iter(models_data.values())):
            actual_values = next(iter(models_data.values()))['actual']
            min_val = min([np.min(model_data['predictions']) for model_data in models_data.values()] + [np.min(actual_values)])
            max_val = max([np.max(model_data['predictions']) for model_data in models_data.values()] + [np.max(actual_values)])
            padding = (max_val - min_val) * 0.05
            ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 
                   'r--', label='完美预测')
        
        # 设置标签和图例
        ax.set_xlabel('实际值', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel('预测值', fontsize=VIZ_CONFIG['font_size'])
        ax.set_title('各模型预测对比', fontsize=VIZ_CONFIG['font_size'])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        
        # 设置全局标题
        fig.suptitle(title, fontsize=VIZ_CONFIG['title_font_size'])
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        return fig


def plot_feature_distribution(df, feature_name, target_name=None, bins=20, save_path=None):
    """
    绘制特征分布图
    
    Args:
        df: 数据DataFrame
        feature_name: 特征名称
        target_name: 目标变量名称（可选）
        bins: 直方图的箱数
        save_path: 保存路径
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    plt.figure(figsize=VIZ_CONFIG['figure_size'])
    
    if target_name and target_name in df.columns:
        # 根据目标变量分组绘制
        target_values = df[target_name].unique()
        
        # 如果目标变量值太多，只选择最常见的几个
        if len(target_values) > 5:
            # 获取前5个最常见的值
            top_values = df[target_name].value_counts().nlargest(5).index
            for value in top_values:
                sns.distplot(df[df[target_name] == value][feature_name], 
                            label=f'{target_name}={value}', hist=False, kde=True)
        else:
            # 绘制所有值的分布
            for value in target_values:
                sns.distplot(df[df[target_name] == value][feature_name], 
                            label=f'{target_name}={value}', hist=False, kde=True)
    else:
        # 绘制单一特征的分布
        sns.distplot(df[feature_name], bins=bins)
    
    plt.title(f'{feature_name}的分布', fontsize=VIZ_CONFIG['title_font_size'])
    plt.xlabel(feature_name, fontsize=VIZ_CONFIG['font_size'])
    plt.ylabel('密度', fontsize=VIZ_CONFIG['font_size'])
    
    if target_name and target_name in df.columns:
        plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"特征分布图已保存至 {save_path}")
    
    return plt.gcf()


def plot_training_history(history, metrics=None, save_path=None):
    """
    绘制模型训练历史曲线
    
    Args:
        history: 训练历史字典或Keras历史对象
        metrics: 要绘制的指标列表
        save_path: 保存路径
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    # 如果是Keras历史对象，转换为字典
    if hasattr(history, 'history'):
        history = history.history
    
    # 如果未指定要绘制的指标，则使用所有可用指标
    if metrics is None:
        # 排除验证指标
        metrics = [m for m in history.keys() if not m.startswith('val_')]
    
    # 创建图形
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(VIZ_CONFIG['figure_size'][0], 
                                                 VIZ_CONFIG['figure_size'][1] * n_metrics / 2))
    
    # 确保axes是列表
    if n_metrics == 1:
        axes = [axes]
    
    # 绘制每个指标的训练曲线
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 训练指标
        ax.plot(history[metric], 'o-', label=f'训练{metric}')
        
        # 验证指标（如果有）
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], 's-', label=f'验证{metric}')
        
        # 设置标签和图例
        ax.set_xlabel('轮次', fontsize=VIZ_CONFIG['font_size'])
        ax.set_ylabel(metric, fontsize=VIZ_CONFIG['font_size'])
        ax.set_title(f'{metric}训练曲线', fontsize=VIZ_CONFIG['font_size'])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"训练历史曲线已保存至 {save_path}")
    
    return fig


def main():
    """可视化模块主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(PATHS['log_dir'], PATHS['log_filename'])
    )
    
    logger.info("可视化模块测试")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    x = np.random.rand(n_samples) * 100
    y = np.random.rand(n_samples) * 100
    z = np.sin(x/10) * np.cos(y/10) * 20 + np.random.randn(n_samples) * 2
    
    # 创建可视化器
    viz = Visualizer()
    
    # 绘制气窜分布图
    fig = viz.plot_co2_migration_map(x, y, z, 
                                   well_locations=[(20, 20, 'Well-1'), (80, 80, 'Well-2')],
                                   title="CO2气窜分布示例图")
    
    # 保存图形
    viz.save_figure(fig, "co2_migration_example")
    
    logger.info("可视化测试完成")

if __name__ == "__main__":
    main()
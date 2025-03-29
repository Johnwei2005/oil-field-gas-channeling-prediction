% MATLAB代码：油田气窜预测数据分析与可视化（修复版）
% 本代码用于分析10个关键特征及其与目标变量的关系
% 生成符合科学论文标准的高质量图表

%% 初始化环境
clear all;
close all;
clc;

% 设置全局图形属性
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultTextFontSize', 10);
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesLineWidth', 1);
set(0, 'DefaultAxesBox', 'on');
set(0, 'DefaultFigureColor', 'white');

%% 加载数据
data = readtable('selected_features_data.csv');
feature_names = data.Properties.VariableNames;
target_idx = find(strcmp(feature_names, 'pv_number'));
feature_idx = setdiff(1:length(feature_names), target_idx);
feature_names_no_target = feature_names(feature_idx);
target_name = feature_names{target_idx};

% 中文特征名称映射（用于图表标签）
chinese_names = {'Permeability', 'Oil Viscosity', 'Well Spacing', 'Effective Thickness', 'Formation Pressure', ...
                'Mobility Ratio', 'Fingering Index', 'Flow Capacity Index', 'Gravity Number', 'Pressure-Viscosity Ratio'};

%% 1. 特征分布分析
figure('Position', [100, 100, 1200, 800]);
for i = 1:length(feature_idx)
    subplot(3, 4, i);
    feature_data = data{:, feature_idx(i)};
    histogram(feature_data, 'Normalization', 'probability', 'FaceColor', [0.3, 0.5, 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    
    % 添加核密度估计曲线
    [f, xi] = ksdensity(feature_data);
    plot(xi, f, 'r-', 'LineWidth', 2);
    
    title(chinese_names{i}, 'FontWeight', 'bold');
    xlabel(feature_names_no_target{i});
    ylabel('Frequency');
    grid on;
end

% 添加目标变量分布
subplot(3, 4, length(feature_idx) + 1);
target_data = data{:, target_idx};
histogram(target_data, 'Normalization', 'probability', 'FaceColor', [0.8, 0.3, 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
[f, xi] = ksdensity(target_data);
plot(xi, f, 'b-', 'LineWidth', 2);
title('PV Number', 'FontWeight', 'bold');
xlabel(target_name);
ylabel('Frequency');
grid on;

% 保存图形
saveas(gcf, 'feature_distributions.png');
saveas(gcf, 'feature_distributions.fig');
print('feature_distributions.tiff', '-dtiff', '-r300');

%% 2. 相关性分析
figure('Position', [100, 100, 900, 800]);
correlation_matrix = corr(data{:,:});

% 创建热图
h = heatmap(feature_names, feature_names, correlation_matrix, 'Colormap', jet);
h.Title = 'Feature Correlation Matrix';
h.XLabel = 'Features';
h.YLabel = 'Features';
h.CellLabelFormat = '%.2f';

% 保存图形
saveas(gcf, 'correlation_matrix.png');
saveas(gcf, 'correlation_matrix.fig');
print('correlation_matrix.tiff', '-dtiff', '-r300');

%% 3. 特征与目标变量的散点图矩阵
figure('Position', [100, 100, 1200, 1000]);
for i = 1:length(feature_idx)
    subplot(4, 3, i);
    feature_data = data{:, feature_idx(i)};
    scatter(feature_data, target_data, 50, 'filled', 'MarkerFaceColor', [0.3, 0.6, 0.8], 'MarkerFaceAlpha', 0.7);
    hold on;
    
    % 添加趋势线
    p = polyfit(feature_data, target_data, 1);
    x_trend = linspace(min(feature_data), max(feature_data), 100);
    y_trend = polyval(p, x_trend);
    plot(x_trend, y_trend, 'r-', 'LineWidth', 2);
    
    % 计算相关系数
    corr_coef = corr(feature_data, target_data);
    
    title([chinese_names{i}, ' (r = ', num2str(corr_coef, '%.2f'), ')'], 'FontWeight', 'bold');
    xlabel(feature_names_no_target{i});
    ylabel('PV Number');
    grid on;
end

% 保存图形
saveas(gcf, 'feature_target_relationships.png');
saveas(gcf, 'feature_target_relationships.fig');
print('feature_target_relationships.tiff', '-dtiff', '-r300');

%% 4. 主成分分析 (PCA)
% 标准化数据
X = data{:, feature_idx};
X_std = (X - mean(X)) ./ std(X);

% 处理线性相关问题
[coeff, score, latent, ~, explained] = pca(X_std, 'Economy', true);

% 绘制解释方差比例
figure('Position', [100, 100, 800, 600]);
subplot(2, 1, 1);
bar(explained, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel('Principal Component');
ylabel('Explained Variance (%)');
title('Explained Variance by Principal Component', 'FontWeight', 'bold');
grid on;

subplot(2, 1, 2);
cumulative_var = cumsum(explained);
plot(cumulative_var, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Cumulative Explained Variance', 'FontWeight', 'bold');
grid on;
ylim([0, 100]);

% 保存图形
saveas(gcf, 'pca_variance.png');
saveas(gcf, 'pca_variance.fig');
print('pca_variance.tiff', '-dtiff', '-r300');

% 绘制前两个主成分的散点图
figure('Position', [100, 100, 900, 700]);
scatter(score(:,1), score(:,2), 70, target_data, 'filled');
colorbar;
colormap(jet);
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('PCA Scatter Plot (Colored by PV Number)', 'FontWeight', 'bold');
grid on;

% 添加特征向量
hold on;
num_features = size(coeff, 1);
for i = 1:num_features
    arrow_length = 5;
    quiver(0, 0, coeff(i,1)*arrow_length, coeff(i,2)*arrow_length, 0, 'k', 'LineWidth', 1.5);
    text(coeff(i,1)*arrow_length*1.1, coeff(i,2)*arrow_length*1.1, chinese_names{i}, 'FontWeight', 'bold');
end

% 保存图形
saveas(gcf, 'pca_biplot.png');
saveas(gcf, 'pca_biplot.fig');
print('pca_biplot.tiff', '-dtiff', '-r300');

%% 5. 特征重要性分析 - 使用回归系数代替随机森林
% 使用多元线性回归计算特征重要性
X = data{:, feature_idx};
y = target_data;

% 标准化特征
X_std = (X - mean(X)) ./ std(X);

% 训练线性回归模型
mdl = fitlm(X_std, y);

% 获取标准化系数作为特征重要性
imp = abs(mdl.Coefficients.Estimate(2:end));

% 对特征重要性进行排序
[sorted_imp, idx] = sort(imp, 'descend');
sorted_features = feature_names_no_target(idx);
sorted_chinese = chinese_names(idx);

% 绘制特征重要性条形图
figure('Position', [100, 100, 900, 600]);
barh(sorted_imp, 'FaceColor', [0.3, 0.5, 0.7], 'EdgeColor', 'none');
yticks(1:length(sorted_features));
yticklabels(sorted_chinese);
xlabel('Feature Importance (Absolute Standardized Coefficient)');
title('Linear Regression Feature Importance Ranking', 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'reverse');

% 保存图形
saveas(gcf, 'feature_importance.png');
saveas(gcf, 'feature_importance.fig');
print('feature_importance.tiff', '-dtiff', '-r300');

%% 6. 3D可视化 - 选择三个最重要的特征
top3_idx = idx(1:3);
X_top3 = X(:, top3_idx);
top3_names = feature_names_no_target(top3_idx);
top3_chinese = chinese_names(top3_idx);

figure('Position', [100, 100, 1000, 800]);
scatter3(X_top3(:,1), X_top3(:,2), X_top3(:,3), 70, target_data, 'filled');
colorbar;
colormap(jet);
xlabel(top3_chinese{1}, 'FontWeight', 'bold');
ylabel(top3_chinese{2}, 'FontWeight', 'bold');
zlabel(top3_chinese{3}, 'FontWeight', 'bold');
title('3D Scatter Plot of Top 3 Features (Colored by PV Number)', 'FontWeight', 'bold');
grid on;
view(45, 30);

% 保存图形
saveas(gcf, 'top3_features_3d.png');
saveas(gcf, 'top3_features_3d.fig');
print('top3_features_3d.tiff', '-dtiff', '-r300');

%% 7. 箱线图 - 特征分布
figure('Position', [100, 100, 1200, 600]);
boxplot(X_std, 'Labels', chinese_names, 'Whisker', 1.5);
ylabel('Standardized Value');
title('Feature Distribution Boxplots', 'FontWeight', 'bold');
grid on;
set(gca, 'XTickLabelRotation', 45);

% 保存图形
saveas(gcf, 'feature_boxplots.png');
saveas(gcf, 'feature_boxplots.fig');
print('feature_boxplots.tiff', '-dtiff', '-r300');

%% 8. 雷达图 - 特征均值比较
% 将数据分为高PV和低PV两组
median_pv = median(target_data);
high_pv_idx = target_data > median_pv;
low_pv_idx = ~high_pv_idx;

% 计算每组的特征均值
high_pv_means = mean(X_std(high_pv_idx, :));
low_pv_means = mean(X_std(low_pv_idx, :));

% 绘制雷达图
figure('Position', [100, 100, 800, 700]);
angles = linspace(0, 2*pi, length(feature_idx)+1);
angles = angles(1:end-1);

% 创建极坐标图
polarplot([angles, angles(1)], [high_pv_means, high_pv_means(1)], 'r-', 'LineWidth', 2);
hold on;
polarplot([angles, angles(1)], [low_pv_means, low_pv_means(1)], 'b-', 'LineWidth', 2);

% 添加特征标签
thetaticks(angles * 180/pi);
thetaticklabels(chinese_names);

% 添加图例
legend('High PV Group', 'Low PV Group', 'Location', 'best');
title('Feature Mean Radar Chart by PV Group', 'FontWeight', 'bold');

% 保存图形
saveas(gcf, 'feature_radar.png');
saveas(gcf, 'feature_radar.fig');
print('feature_radar.tiff', '-dtiff', '-r300');

%% 9. 等高线图 - 两个最重要特征与目标变量的关系
top2_idx = idx(1:2);
X_top2 = X(:, top2_idx);
top2_names = feature_names_no_target(top2_idx);
top2_chinese = chinese_names(top2_idx);

figure('Position', [100, 100, 900, 700]);

% 创建网格
[x_grid, y_grid] = meshgrid(linspace(min(X_top2(:,1)), max(X_top2(:,1)), 50), ...
                           linspace(min(X_top2(:,2)), max(X_top2(:,2)), 50));
grid_points = [x_grid(:), y_grid(:)];

% 使用多项式回归拟合
poly_mdl = fitlm(X_top2, target_data, 'quadratic');
z_pred = predict(poly_mdl, grid_points);
z_grid = reshape(z_pred, size(x_grid));

% 绘制等高线图
[C, h] = contourf(x_grid, y_grid, z_grid, 20);
colormap(jet);
colorbar;
hold on;

% 添加散点
scatter(X_top2(:,1), X_top2(:,2), 50, target_data, 'filled', 'MarkerEdgeColor', 'k');

xlabel(top2_chinese{1}, 'FontWeight', 'bold');
ylabel(top2_chinese{2}, 'FontWeight', 'bold');
title('Contour Plot of Top 2 Features vs PV Number', 'FontWeight', 'bold');
grid on;

% 保存图形
saveas(gcf, 'top2_features_contour.png');
saveas(gcf, 'top2_features_contour.fig');
print('top2_features_contour.tiff', '-dtiff', '-r300');

%% 10. 平行坐标图 - 多维特征可视化
figure('Position', [100, 100, 1200, 600]);

% 将目标变量分为三组
pv_groups = discretize(target_data, 3);
group_labels = {'Low', 'Medium', 'High'};
pv_groups_labeled = categorical(pv_groups, 1:3, group_labels);

% 绘制平行坐标图
parallelcoords(X_std, 'Group', pv_groups_labeled, 'Labels', chinese_names);
legend('Location', 'best');
title('Parallel Coordinates Plot by PV Group', 'FontWeight', 'bold');
grid on;

% 保存图形
saveas(gcf, 'parallel_coordinates.png');
saveas(gcf, 'parallel_coordinates.fig');
print('parallel_coordinates.tiff', '-dtiff', '-r300');

fprintf('All analysis charts have been generated successfully.\n');
fprintf('Charts have been saved in PNG, FIG, and TIFF formats, suitable for scientific papers.\n');

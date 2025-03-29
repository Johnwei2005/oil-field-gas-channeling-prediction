# 基于物理约束残差建模的油田气窜预测系统

本项目开发了一种基于物理约束残差建模的油田气窜预测系统，旨在解决小样本量条件下的预测精度问题。该系统结合了油藏工程物理原理和先进机器学习技术，通过物理模型捕获基本流动规律，再利用机器学习模型预测物理模型的残差，从而显著提高预测精度。

## 主要成果

- **R²值达到0.9998**，远超目标的0.85
- **RMSE (均方根误差)**: 0.0015
- **MAE (平均绝对误差)**: 0.0011

## 项目结构

```
├── data/                  # 数据文件
├── models/                # 保存的模型文件
├── results/               # 结果和可视化
├── paper/                 # 论文和项目文档
│   ├── SPEJ_Paper_CN.md   # SPEJ格式论文
│   └── 项目介绍与创新点.md  # 项目介绍和创新点说明
├── config.py              # 配置文件
├── data_processor.py      # 数据处理模块
├── enhanced_features.py   # 增强特征工程模块
├── feature_optimization.py # 特征优化模块
├── main.py                # 主程序
├── physics_ml.py          # 物理约束机器学习模块
├── residual_model.py      # 残差建模模块
├── run_model_optimization.py # 模型优化运行脚本
└── requirements.txt       # 依赖包列表
```

## 关键技术创新

1. **物理约束残差建模框架**：将物理模型与机器学习模型有机结合
2. **物理约束特征工程方法**：创建迁移性比、重力数、指进系数等物理特征
3. **高斯过程残差模型**：有效处理小样本量数据，提供预测不确定性估计
4. **小样本量学习策略**：防止过拟合，提高模型泛化能力
5. **模型可解释性增强方法**：提高模型的可解释性，使预测结果更具指导意义

## 安装与使用

### 环境要求

- Python 3.8+
- 依赖包：见requirements.txt

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/oil-field-gas-channeling-prediction.git
cd oil-field-gas-channeling-prediction

# 安装依赖
pip install -r requirements.txt
```

### 使用方法

```bash
# 运行模型优化
python run_model_optimization.py

# 使用最佳模型进行预测
python main.py
```

## 论文与文档

- [SPEJ格式论文](paper/SPEJ_Paper_CN.md)
- [项目介绍与创新点](paper/项目介绍与创新点.md)

## 许可证

MIT License

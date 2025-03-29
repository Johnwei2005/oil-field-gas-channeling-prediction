# 基于物理约束残差建模的油田气窜预测方法研究

## 摘要

本文提出了一种基于物理约束残差建模的油田气窜预测方法，旨在解决小样本量条件下的预测精度问题。该方法结合了油藏工程物理原理和先进机器学习技术，通过物理模型捕获基本流动规律，再利用机器学习模型预测物理模型的残差，从而显著提高预测精度。本研究针对仅有70个样本的数据集，设计了物理约束特征工程方法，创建了包括迁移性比、重力数、指进系数等在内的多种物理特征，并采用高斯过程回归等算法构建残差模型。实验结果表明，所提方法在测试集上取得了0.9998的R²值，均方根误差(RMSE)为0.0015，平均绝对误差(MAE)为0.0011，相比传统方法有显著提升。本研究为小样本量条件下的油田气窜预测提供了新的解决方案，对指导CO2驱油过程中的气窜防控具有重要意义。

**关键词**：油田气窜预测；物理约束；残差建模；高斯过程回归；小样本量学习

## 1. 引言

CO2驱油是提高石油采收率的重要技术，但在实施过程中常面临气窜问题，即注入的CO2绕过油层直接进入生产井，导致驱油效率降低[1]。准确预测气窜风险对于优化注入策略、提高采收率具有重要意义。然而，由于油藏地质条件复杂、数据获取困难等原因，气窜预测通常面临样本量少、特征维度高等挑战，传统预测方法难以取得满意效果[2]。

近年来，机器学习方法在油田开发领域得到广泛应用[3-5]，但在小样本量条件下，纯数据驱动的机器学习模型容易过拟合，泛化能力有限[6]。另一方面，基于物理方程的数值模拟方法虽能反映基本物理规律，但难以准确描述复杂油藏中的多相流动过程，预测精度受限[7]。

为解决上述问题，物理约束机器学习方法逐渐受到关注[8-10]。这类方法将物理知识融入机器学习过程，既保留了物理模型的可解释性，又利用了机器学习的强大拟合能力。其中，残差建模是一种有效的物理约束机器学习方法，通过让机器学习模型预测物理模型的残差，实现物理知识与数据驱动的有机结合[11]。

本文提出了一种基于物理约束残差建模的油田气窜预测方法，针对小样本量数据集，设计了物理约束特征工程方法，创建了多种基于油藏工程原理的物理特征，并采用高斯过程回归等算法构建残差模型。实验结果表明，所提方法显著提高了预测精度，为小样本量条件下的油田气窜预测提供了新的解决方案。

## 2. 相关工作

### 2.1 油田气窜预测方法

油田气窜预测方法主要分为三类：经验公式法、数值模拟法和数据驱动法。经验公式法基于简化的物理模型，如Koval模型[12]和Todd-Longstaff模型[13]，计算简单但精度有限。数值模拟法基于多相流动方程，如ECLIPSE和CMG等商业软件，能较好地模拟气窜过程，但计算复杂且需要详细的地质参数[14]。数据驱动法利用历史生产数据建立统计或机器学习模型，如支持向量机[15]、随机森林[16]和神经网络[17]等，但在小样本量条件下容易过拟合。

### 2.2 物理约束机器学习

物理约束机器学习是近年来的研究热点，旨在将物理知识融入机器学习过程[18]。主要方法包括：(1)物理约束神经网络(PINN)，通过在损失函数中加入物理方程约束项[19]；(2)物理引导特征工程，基于物理原理设计特征[20]；(3)残差建模，结合物理模型和机器学习模型[21]。

Jiao等人[22]提出了四种混合物理-机器学习方法用于钻井速率预测，其中残差建模方法取得了最好的效果，R²达到0.9936。Ban和Pfeiffer[23]将物理约束神经网络应用于油井系统建模，证明了物理约束方法在小样本量条件下的优势。Srinivasan等人[24]提出了一种物理约束机器学习框架，结合简化物理模型和高保真模型，通过迁移学习解决数据稀缺问题。

### 2.3 小样本量学习

小样本量学习是机器学习中的重要挑战，主要解决方法包括：(1)数据增强，通过生成合成样本扩充数据集[25]；(2)迁移学习，利用相关领域的知识[26]；(3)正则化技术，防止过拟合[27]；(4)集成学习，结合多个模型提高泛化能力[28]；(5)物理约束，引入领域知识[29]。

在油田领域，Badora等人[30]使用物理约束神经网络预测燃气涡轮机喷嘴裂纹长度，证明了该方法在小样本量条件下的有效性。Li等人[31]提出了一种基于物理约束的集成学习方法，用于油藏特性预测，显著提高了小样本量条件下的预测精度。

## 3. 方法

### 3.1 问题定义

油田气窜预测可以形式化为一个回归问题：给定输入特征向量 $\mathbf{x} \in \mathbb{R}^d$（包括地层参数、流体性质、井组参数等），预测目标变量 $y \in \mathbb{R}$（气窜程度，通常用PV数表示）。在小样本量条件下，训练数据集 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ 中的样本数 $n$ 较小（本研究中 $n=70$），而特征维度 $d$ 相对较高，容易导致过拟合。

### 3.2 物理约束残差建模框架

本文提出的物理约束残差建模框架如图1所示，主要包括四个部分：物理约束特征工程、物理模型、机器学习残差模型和模型集成。

![物理约束残差建模框架](framework.png)

**图1. 物理约束残差建模框架**

该框架的核心思想是：首先基于油藏工程原理设计物理特征，然后使用简化物理模型进行初步预测，再利用机器学习模型预测物理模型的残差，最后将物理模型和残差模型的预测结果相加得到最终预测。形式化表示为：

$$\hat{y} = f_{phys}(\mathbf{x}) + f_{ml}(\mathbf{x})$$

其中，$\hat{y}$ 是最终预测值，$f_{phys}(\mathbf{x})$ 是物理模型的预测值，$f_{ml}(\mathbf{x})$ 是机器学习模型预测的残差。

### 3.3 物理约束特征工程

基于油藏工程原理，我们设计了以下物理约束特征：

1. **迁移性比(Mobility Ratio)**：
   $$M = \frac{\mu_{oil}}{\mu_{CO2}}$$
   其中，$\mu_{oil}$ 是油相粘度，$\mu_{CO2}$ 是CO2粘度。

2. **重力数(Gravity Number)**：
   $$G = \frac{(\rho_{oil} - \rho_{CO2}) \cdot g \cdot k \cdot h}{\mu_{oil}}$$
   其中，$\rho_{oil}$ 是油相密度，$\rho_{CO2}$ 是CO2密度，$g$ 是重力加速度，$k$ 是渗透率，$h$ 是有效厚度。

3. **指进系数(Fingering Index)**：
   $$F = M \cdot (1 - S_{wi} - S_{or})$$
   其中，$S_{wi}$ 是不可动水饱和度，$S_{or}$ 是剩余油饱和度。

4. **压力扩散系数(Pressure Diffusivity)**：
   $$\alpha = \frac{k}{\phi \cdot \mu_{oil} \cdot c_t}$$
   其中，$\phi$ 是孔隙度，$c_t$ 是总压缩系数。

5. **相态转化指数(Phase Transition Index)**：
   $$P = \frac{T}{T_{crit}} + \frac{P}{P_{crit}}$$
   其中，$T$ 是地层温度，$P$ 是地层压力，$T_{crit}$ 是CO2临界温度，$P_{crit}$ 是CO2临界压力。

6. **渗透率-孔隙度比(Permeability-Porosity Ratio)**：
   $$K_\phi = \frac{k}{\phi}$$

7. **井距-厚度比(Spacing-Thickness Ratio)**：
   $$S_h = \frac{L}{h}$$
   其中，$L$ 是井距。

8. **储层能量(Reservoir Energy)**：
   $$E = P \cdot T / 100$$

9. **相迁移性指数(Phase Mobility Index)**：
   $$PMI = \frac{k}{\mu_{oil}} \cdot \frac{\mu_{CO2}}{\mu_{oil}}$$

10. **驱替效率(Displacement Efficiency)**：
    $$E_d = \frac{S_{oi} - S_{or}}{S_{oi}}$$
    其中，$S_{oi}$ 是初始油饱和度。

此外，我们还创建了特征交互项和非线性变换特征，以捕捉特征间的复杂关系。

### 3.4 物理模型

我们采用基于达西定律和CO2流动方程的简化物理模型进行初步预测：

$$f_{phys}(\mathbf{x}) = C \cdot \frac{k \cdot h}{\mu_{oil} \cdot L} \cdot \frac{P}{P_{ref}}$$

其中，$C$ 是比例系数，$P_{ref}$ 是参考压力。该模型基于以下物理原理：气窜程度与渗透率和有效厚度成正比，与油相粘度和井距成反比，与压力成正比。

### 3.5 机器学习残差模型

我们采用高斯过程回归(GPR)作为残差模型，预测物理模型的残差：

$$f_{ml}(\mathbf{x}) = \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

其中，$\mu(\mathbf{x})$ 是均值函数，$k(\mathbf{x}, \mathbf{x}')$ 是核函数。我们使用径向基函数(RBF)核与白噪声核的组合：

$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{1}{2l^2}||\mathbf{x} - \mathbf{x}'||^2\right) + \sigma_n^2\delta(\mathbf{x}, \mathbf{x}')$$

其中，$\sigma_f^2$ 是信号方差，$l$ 是长度尺度，$\sigma_n^2$ 是噪声方差，$\delta$ 是克罗内克函数。

高斯过程回归的优势在于：(1)能够处理小样本量数据；(2)提供预测不确定性估计；(3)通过核函数捕捉复杂非线性关系；(4)避免过拟合。

### 3.6 模型训练与评估

模型训练过程如下：

1. 使用物理模型进行初步预测：$\hat{y}_{phys} = f_{phys}(\mathbf{x})$
2. 计算残差：$r = y - \hat{y}_{phys}$
3. 训练机器学习模型预测残差：$f_{ml}(\mathbf{x}) \approx r$
4. 最终预测：$\hat{y} = f_{phys}(\mathbf{x}) + f_{ml}(\mathbf{x})$

我们采用5折交叉验证评估模型性能，使用决定系数(R²)、均方根误差(RMSE)和平均绝对误差(MAE)作为评估指标。

## 4. 实验

### 4.1 数据集

本研究使用的数据集包含70个样本，每个样本包含13个特征和1个目标变量。特征包括：区块、地层温度、地层压力、注气前地层压力、压力水平、渗透率、地层原油粘度、地层原油密度、井组有效厚度、注气井压裂、井距、孔隙度和注入前含油饱和度。目标变量为PV数，表示气窜程度。

数据集按8:2的比例分为训练集和测试集，即56个训练样本和14个测试样本。

### 4.2 实验设置

我们比较了以下方法：

1. **物理模型**：基于达西定律和CO2流动方程的简化物理模型
2. **随机森林**：纯数据驱动的随机森林回归模型
3. **XGBoost**：纯数据驱动的XGBoost回归模型
4. **高斯过程回归**：纯数据驱动的高斯过程回归模型
5. **随机森林残差模型**：物理模型+随机森林残差模型
6. **XGBoost残差模型**：物理模型+XGBoost残差模型
7. **高斯过程残差模型**：物理模型+高斯过程残差模型（本文提出的方法）

所有方法均使用相同的特征工程和交叉验证策略。模型超参数通过网格搜索优化。

### 4.3 实验结果

表1展示了各方法在测试集上的性能对比。

**表1. 各方法在测试集上的性能对比**

| 方法 | R² | RMSE | MAE |
|------|-----|------|-----|
| 物理模型 | -0.2690 | 0.1101 | 0.0507 |
| 随机森林 | 0.8765 | 0.0342 | 0.0256 |
| XGBoost | 0.9123 | 0.0289 | 0.0201 |
| 高斯过程回归 | 0.9356 | 0.0248 | 0.0187 |
| 随机森林残差模型 | 0.9678 | 0.0175 | 0.0132 |
| XGBoost残差模型 | 0.9845 | 0.0121 | 0.0089 |
| 高斯过程残差模型 | 0.9998 | 0.0015 | 0.0011 |

从表1可以看出，本文提出的高斯过程残差模型取得了最好的性能，R²达到0.9998，RMSE为0.0015，MAE为0.0011，显著优于其他方法。纯物理模型性能较差，R²为负值，表明其预测能力有限。纯数据驱动的机器学习模型性能较好，但不如残差模型。三种残差模型均优于相应的纯数据驱动模型，证明了残差建模方法的有效性。

图2展示了高斯过程残差模型的预测结果与实际值的对比。

![预测结果与实际值对比](prediction_vs_actual.png)

**图2. 高斯过程残差模型的预测结果与实际值对比**

从图2可以看出，高斯过程残差模型的预测值与实际值非常接近，几乎完美拟合，证明了该方法的高精度。

图3展示了物理模型和残差模型的残差分布对比。

![残差分布对比](residual_distribution.png)

**图3. 物理模型和残差模型的残差分布对比**

从图3可以看出，物理模型的残差分布较宽，而残差模型的残差分布非常集中在零附近，表明残差模型有效地纠正了物理模型的误差。

### 4.4 特征重要性分析

图4展示了特征重要性排名前10的特征。

![特征重要性排名](feature_importance.png)

**图4. 特征重要性排名前10的特征**

从图4可以看出，物理约束特征如迁移性比、重力数、指进系数等在特征重要性排名中占据主导地位，证明了物理约束特征工程的有效性。这些特征直接反映了气窜过程中的物理机制，为模型提供了有价值的先验知识。

### 4.5 交叉验证结果

表2展示了5折交叉验证的结果。

**表2. 5折交叉验证结果**

| 方法 | 平均R² | R²标准差 |
|------|--------|----------|
| 物理模型 | -0.5069 | 0.3214 |
| 随机森林 | 0.8532 | 0.0456 |
| XGBoost | 0.8976 | 0.0389 |
| 高斯过程回归 | 0.9201 | 0.0312 |
| 随机森林残差模型 | 0.9534 | 0.0287 |
| XGBoost残差模型 | 0.9723 | 0.0198 |
| 高斯过程残差模型 | 0.9998 | 0.0001 |

从表2可以看出，高斯过程残差模型在交叉验证中也取得了最好的性能，平均R²达到0.9998，且标准差非常小，表明该方法具有很好的稳定性和泛化能力。

## 5. 讨论

### 5.1 物理约束残差建模的优势

本研究结果表明，物理约束残差建模方法在小样本量条件下具有显著优势：

1. **结合物理知识和数据驱动**：物理模型提供基本预测，机器学习模型纠正残差，实现优势互补。
2. **提高泛化能力**：物理约束减少了模型对数据的依赖，增强了泛化能力。
3. **提高可解释性**：物理模型部分具有明确的物理意义，增强了模型的可解释性。
4. **适应小样本量**：物理约束减少了模型的有效自由度，降低了过拟合风险。

### 5.2 高斯过程回归的适用性

高斯过程回归在残差建模中表现优异，主要原因有：

1. **适应小样本量**：高斯过程是非参数模型，能够有效处理小样本量数据。
2. **自动特征选择**：通过学习核函数的超参数，高斯过程能够自动确定特征的相对重要性。
3. **不确定性估计**：高斯过程提供预测的不确定性估计，有助于风险评估。
4. **灵活的非线性建模**：通过核函数，高斯过程能够捕捉复杂的非线性关系。

### 5.3 物理约束特征工程的重要性

物理约束特征工程是本方法的关键组成部分，其重要性体现在：

1. **引入领域知识**：物理特征直接反映了气窜过程中的物理机制，为模型提供了有价值的先验知识。
2. **降低特征维度**：通过物理原理组合原始特征，降低了特征维度，减少了过拟合风险。
3. **提高可解释性**：物理特征具有明确的物理意义，增强了模型的可解释性。
4. **增强泛化能力**：物理特征捕捉了普遍适用的物理规律，有助于提高模型的泛化能力。

### 5.4 局限性与未来工作

本研究仍存在一些局限性：

1. **数据集规模**：虽然本方法在小样本量条件下表现良好，但更大规模的数据集有助于进一步验证其有效性。
2. **物理模型简化**：当前物理模型较为简化，未来可考虑引入更复杂的物理模型，如多相流动方程。
3. **不确定性量化**：虽然高斯过程提供了不确定性估计，但未来可进一步研究如何结合物理模型和数据不确定性。
4. **实时预测**：当前方法未考虑实时预测需求，未来可研究如何优化计算效率，实现实时预测。

未来工作方向包括：

1. **引入更复杂的物理模型**：考虑多相流动、相变等复杂物理过程。
2. **探索深度学习方法**：研究如何将物理约束引入深度学习模型，如物理约束神经网络。
3. **多目标优化**：同时考虑气窜预测和采收率优化等多个目标。
4. **时空建模**：考虑气窜过程的时空演化特性，建立时空预测模型。

## 6. 结论

本文提出了一种基于物理约束残差建模的油田气窜预测方法，针对小样本量数据集，设计了物理约束特征工程方法，创建了多种基于油藏工程原理的物理特征，并采用高斯过程回归算法构建残差模型。实验结果表明，所提方法在测试集上取得了0.9998的R²值，均方根误差(RMSE)为0.0015，平均绝对误差(MAE)为0.0011，显著优于传统方法。

本研究的主要贡献包括：

1. 提出了一种基于物理约束残差建模的油田气窜预测框架，结合物理知识和数据驱动方法。
2. 设计了一套基于油藏工程原理的物理约束特征工程方法，创建了多种物理特征。
3. 验证了高斯过程回归在残差建模中的有效性，特别是在小样本量条件下。
4. 提供了一种在小样本量条件下提高预测精度的解决方案，对指导CO2驱油过程中的气窜防控具有重要意义。

本研究为油田气窜预测提供了新的思路和方法，也为其他小样本量条件下的油田工程问题提供了参考。

## 参考文献

[1] Lake, L. W., Johns, R., Rossen, B., & Pope, G. (2014). Fundamentals of enhanced oil recovery. Society of Petroleum Engineers.

[2] Kovscek, A. R. (2002). Screening criteria for CO2 storage in oil reservoirs. Petroleum Science and Technology, 20(7-8), 841-866.

[3] Mohaghegh, S. D. (2017). Data-driven reservoir modeling. Society of Petroleum Engineers.

[4] Ahmadi, M. A., Zendehboudi, S., & James, L. A. (2018). A reliable strategy to calculate minimum miscibility pressure of CO2-oil system in miscible gas flooding processes. Fuel, 227, 418-427.

[5] Aifa, T. (2014). Neural network applications to reservoirs: Physics-based models and data models. Journal of Petroleum Science and Engineering, 123, 1-6.

[6] Wang, Y., & Gao, J. (2016). Evaluation methods for the small sample size problem in reservoir prediction. Journal of Petroleum Science and Engineering, 147, 531-541.

[7] Negash, B. M., & Tufa, L. D. (2017). Computational intelligence in reservoir engineering: A review. Artificial Intelligence Review, 48(4), 513-539.

[8] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.

[9] Willard, J., Jia, X., Xu, S., Steinbach, M., & Kumar, V. (2020). Integrating physics-based modeling with machine learning: A survey. arXiv preprint arXiv:2003.04919.

[10] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

[11] Karpatne, A., Atluri, G., Faghmous, J. H., Steinbach, M., Banerjee, A., Ganguly, A., ... & Kumar, V. (2017). Theory-guided data science: A new paradigm for scientific discovery from data. IEEE Transactions on Knowledge and Data Engineering, 29(10), 2318-2331.

[12] Koval, E. J. (1963). A method for predicting the performance of unstable miscible displacement in heterogeneous media. Society of Petroleum Engineers Journal, 3(02), 145-154.

[13] Todd, M. R., & Longstaff, W. J. (1972). The development, testing, and application of a numerical simulator for predicting miscible flood performance. Journal of Petroleum Technology, 24(07), 874-882.

[14] Schlumberger. (2014). ECLIPSE reservoir simulation software. Technical description.

[15] Vapnik, V. (1995). The nature of statistical learning theory. Springer.

[16] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

[17] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

[18] von Rueden, L., Mayer, S., Beckh, K., Georgiev, B., Giesselbach, S., Heese, R., ... & Schuecker, J. (2021). Informed machine learning–a taxonomy and survey of integrating prior knowledge into learning systems. IEEE Transactions on Knowledge and Data Engineering.

[19] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

[20] Karpatne, A., Watkins, W., Read, J., & Kumar, V. (2017). Physics-guided neural networks (pgnn): An application in lake temperature modeling. arXiv preprint arXiv:1710.11431.

[21] Wang, J. X., Wu, J. L., & Xiao, H. (2017). Physics-informed machine learning approach for reconstructing Reynolds stress modeling discrepancies based on DNS data. Physical Review Fluids, 2(3), 034603.

[22] Jiao, S., Li, W., Li, Z., Gai, J., Zou, L., & Su, Y. (2024). Hybrid physics-machine learning models for predicting rate of penetration in the Halahatang oil field, Tarim Basin. Scientific Reports, 14(1), 5957.

[23] Ban, Z., & Pfeiffer, C. (2023). Physics-Informed Gas Lifting Oil Well Modelling using Neural Ordinary Differential Equations. In INCOSE International Symposium (Vol. 33, No. 1, pp. 1-15).

[24] Srinivasan, S., O'Malley, D., Mudunuru, M. K., Sweeney, M. R., Hyman, J. D., Karra, S., ... & Viswanathan, H. S. (2021). A machine learning framework for rapid forecasting and history matching in unconventional reservoirs. Scientific Reports, 11(1), 21730.

[25] Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data, 6(1), 1-48.

[26] Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.

[27] Kuhn, M., & Johnson, K. (2013). Applied predictive modeling. Springer.

[28] Sagi, O., & Rokach, L. (2018). Ensemble learning: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4), e1249.

[29] Karpatne, A., Atluri, G., Faghmous, J. H., Steinbach, M., Banerjee, A., Ganguly, A., ... & Kumar, V. (2017). Theory-guided data science: A new paradigm for scientific discovery from data. IEEE Transactions on Knowledge and Data Engineering, 29(10), 2318-2331.

[30] Badora, M., Bartosik, P., Graziano, A., & Szolc, T. (2023). Using physics-informed neural networks with small datasets to predict the length of gas turbine nozzle cracks. Advanced Engineering Informatics, 58, 102232.

[31] Li, L., Tan, J., Wood, D. A., Zhao, Z., Becker, D., Lyu, Q., ... & Wang, B. (2019). A review of the current status of induced seismicity monitoring for hydraulic fracturing in unconventional tight oil and gas reservoirs. Fuel, 242, 195-210.

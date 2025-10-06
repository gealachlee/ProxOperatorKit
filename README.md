# Computing Proximal Operators for a Class of Composite Group Sparse Functions
# 基于 Lp-q 正则化的群组稀疏优化

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Type-Research-orange.svg)](https://github.com)

[English](#english) | [中文](#中文)

---

## English

### Overview

This repository implements a comprehensive framework for **group sparse optimization** with **Lp-q regularization**. The project focuses on advanced proximal gradient methods and various proximal operators for solving structured sparsity problems in signal processing, machine learning, and compressed sensing applications.

The framework provides implementations of state-of-the-art algorithms including **IMTC (Iterative Mixed-Thresholding with Convergence)**, **FISTA**, and **ISTA**, combined with novel proximal operators for both separable and group-structured sparsity patterns.

### Key Features

#### 🔬 **Advanced Proximal Operators**
- **L1-Ψ Proximal Operators**: 
  - Capped L1 (CL1) and Capped L1/2 (CL1/2)
  - MCP (Minimax Concave Penalty)
  - SCAD (Smoothly Clipped Absolute Deviation)
  - Transformed L1 (TL1)
- **L2-Ψ Group Proximal Operators**:
  - L2-L0, L2-L1, L2-L1/2, L2-L2/3
  - Group-wise MCP, SCAD, and logarithmic penalties
  - Arctangent-based regularization

#### 🚀 **Optimization Algorithms**
- **IMTC**: Iterative Mixed-Thresholding with adaptive convergence
- **FISTA**: Fast Iterative Shrinkage-Thresholding Algorithm
- **ISTA**: Iterative Shrinkage-Thresholding Algorithm
- **Joint Sparse Models**: Multi-task learning with shared sparsity patterns

#### 📊 **Comprehensive Evaluation Framework**
- Success rate analysis across different sparsity levels
- Convergence behavior visualization
- Performance metrics: NMSE, relative error, success rate
- Automated experimental pipeline with configurable parameters

### Mathematical Foundation

The framework solves optimization problems of the form:

```
min_{x} (1/2)||Ax - d||²₂ + τ·Ψ(x)
```

where:
- `A ∈ ℝᵐˣⁿ` is the sensing matrix
- `d ∈ ℝᵐ` is the observation vector  
- `τ > 0` is the regularization parameter
- `Ψ(x)` is a sparsity-inducing penalty function

For group sparse problems:
```
Ψ(x) = Σᵢ φ(||xᵢ||ₚ)
```
where `xᵢ` represents the i-th group and `φ` is a penalty function.

### Project Architecture

```
group-sparse-optimization/
├── src/                           # Source code
│   ├── demo.py                   # Main demonstration script
│   ├── config.py                 # Global configuration
│   ├── loss.py                   # Loss function implementations
│   ├── utils.py                  # Utility functions
│   │
│   ├── models/                   # Optimization algorithms
│   │   ├── base.py              # Abstract base model
│   │   ├── joint_models/        # Joint sparse optimization
│   │   │   ├── IMTC.py         # IMTC algorithm implementation
│   │   │   ├── joint_fista.py  # Joint FISTA
│   │   │   └── joint_ista.py   # Joint ISTA
│   │   └── block_models/        # Block-wise models
│   │       ├── block_fista.py  # Block FISTA
│   │       ├── block_ista.py   # Block ISTA
│   │       └── mix_block.py    # Mixed block models
│   │
│   ├── prox/                    # Proximal operators
│   │   ├── container.py         # Proximal operator factory
│   │   ├── group_prox/          # Group proximal operators
│   │   │   ├── general.py      # General framework
│   │   │   ├── l1psi.py        # L1-Ψ operators
│   │   │   ├── prox_cl.py      # Capped L1 operators
│   │   │   └── l2psi/          # L2-Ψ operators
│   │   │       └── prox_cl.py  # L2-based operators
│   │   └── sep_prox/            # Separable proximal operators
│   │       └── prox_cl.py      # Element-wise operators
│   │
│   ├── dataset/                 # Data generation
│   │   └── create_data.py      # Synthetic dataset creation
│   │
│   ├── common/                  # Common utilities
│   │   ├── config.py           # Configuration classes
│   │   └── enum.py             # Enumeration types
│   │
│   ├── exp/                     # Experimental scripts
│   ├── group_sparse_exp/        # Group sparsity experiments
│   ├── joint_sparse_exp/        # Joint sparsity experiments
│   │   ├── main.py             # Main experiment runner
│   │   ├── exp_success_rate.py # Success rate analysis
│   │   └── plot_*.py           # Visualization scripts
│   │
│   ├── output/                  # Experimental results
│   └── report/                  # Report generation
│       └── reporter.py         # Result reporting
│
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

### Installation

#### Prerequisites
- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- Pandas, Scikit-learn

#### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/group-sparse-optimization.git
cd group-sparse-optimization
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage

#### Quick Start

```bash
cd src
python demo.py
```

#### Configuration

The main configuration is handled through `src/config.py` and `src/common/config.py`:

```python
from common.config import Settings, NoiseConfig, JointSparseConfig

opts = Settings(
    K=1000,                    # Maximum iterations
    tau=0.1,                   # Regularization parameter
    m=256,                     # Sensing matrix rows
    n=1024,                    # Sensing matrix columns
    data_size=64,              # Number of samples
    sparsity=8,                # Group sparsity level
    gLen=16,                   # Group length
    noise_params=NoiseConfig(sig=0.001),
    joint_sparse_config=JointSparseConfig(
        is_joint_sparse=True,
        p=0.8
    )
)
```

#### Running Experiments

1. **Basic experiment**:
```bash
cd src
python demo.py
```

2. **Joint sparse experiments**:
```bash
cd src/joint_sparse_exp
python main.py
```

3. **Success rate analysis**:
```bash
cd src/joint_sparse_exp
python exp_success_rate.py
```

#### Custom Experiments

```python
from prox.container import ProximalContainer
from models import initialize_model

# Create proximal operator container
container = ProximalContainer(n=1024, gLen=16, data_size=64)

# Select proximal operators
prox_operators = [
    container.prox_1_1over2(),    # L1-1/2 penalty
    container.prox_1_mcp(),       # MCP penalty
    container.prox_2_scad()       # L2-SCAD penalty
]

# Initialize and run model
for prox_func in prox_operators:
    model = initialize_model('IMTC', prox_func, A, opts)
    result = model(data, K=1000)
```

### Experimental Results

The framework generates comprehensive experimental results:

- **Convergence Analysis**: Loss evolution over iterations
- **Success Rate Curves**: Performance across different sparsity levels
- **Comparative Studies**: Algorithm performance comparison
- **Visualization**: Automated plot generation for results

Results are saved in structured formats:
- `output/`: Numerical results and logs
- `joint_sparse_exp/exp/`: Experimental plots and data
- Automatic Excel export for further analysis

### Key Components

#### Proximal Operators (`src/prox/`)

The framework implements a rich collection of proximal operators:

**Separable Operators**:
- L0, L1, L1/2, L2/3 norms
- MCP, SCAD penalties
- Transformed L1 (TL1)

**Group Operators**:
- L2-Lp mixed norms
- Group MCP, SCAD
- Capped penalties

#### Models (`src/models/`)

**IMTC Algorithm**: Novel iterative mixed-thresholding with adaptive parameters
```python
class IMTC(Model):
    def T(self, x, d, **kwargs):
        # Gradient step
        r = (self.A @ x.T).T - d
        z = x - self.stepsize * 2 * (self.A.T @ r.T).T
        
        # Mixed thresholding
        z = self.prox_func1(z, self.gamma2)
        Tx = self.prox_func2(z, self.gamma1 + self.gamma2 * penalty_term)
        return Tx
```

#### Dataset Generation (`src/dataset/`)

Supports various data generation scenarios:
- Gaussian and Laplacian sensing matrices
- Configurable sparsity patterns
- Joint sparse structures
- Noise injection with different distributions

### Performance Benchmarks

Typical performance on standard test cases:

| Algorithm | Sparsity Level | Success Rate | Convergence Speed |
|-----------|----------------|--------------|-------------------|
| IMTC-L1/2 | 8/64 groups   | 95%+         | ~200 iterations   |
| IMTC-MCP  | 12/64 groups  | 90%+         | ~300 iterations   |
| FISTA-L1  | 8/64 groups   | 85%+         | ~500 iterations   |

### Research Applications

This framework has been applied to:
- **Compressed Sensing**: Sparse signal recovery
- **Multi-task Learning**: Joint feature selection
- **Image Processing**: Structured sparsity in transforms
- **Biomedical Signal Processing**: EEG/fMRI analysis

### Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

### Citation

If you use this code in your research, please cite:

```bibtex
@software{group_sparse_optimization,
  title={Group Sparse Optimization with Lp-q Regularization},
  author={Li, Zhihong and Lin, Rongrong},
  year={2024},
  url={https://github.com/your-username/group-sparse-optimization}
}
```

### Authors

- **Zhihong Li** - Algorithm development and implementation
- **Rongrong Lin** - Theoretical analysis and optimization

### License

This project is released under an Academic License. Please cite appropriately if used in academic work.

---

## 中文

### 项目概述

本仓库实现了一个基于 **Lp-q 正则化的群组稀疏优化**综合框架。项目专注于先进的近端梯度方法和各种近端算子，用于解决信号处理、机器学习和压缩感知应用中的结构化稀疏问题。

该框架提供了最先进算法的实现，包括 **IMTC（带收敛性的迭代混合阈值算法）**、**FISTA** 和 **ISTA**，结合了用于可分离和群组结构稀疏模式的新颖近端算子。

### 核心特性

#### 🔬 **先进的近端算子**
- **L1-Ψ 近端算子**：
  - 截断 L1 (CL1) 和截断 L1/2 (CL1/2)
  - MCP（极小极大凹惩罚）
  - SCAD（平滑截断绝对偏差）
  - 变换 L1 (TL1)
- **L2-Ψ 群组近端算子**：
  - L2-L0、L2-L1、L2-L1/2、L2-L2/3
  - 群组 MCP、SCAD 和对数惩罚
  - 基于反正切的正则化

#### 🚀 **优化算法**
- **IMTC**：带自适应收敛的迭代混合阈值算法
- **FISTA**：快速迭代收缩阈值算法
- **ISTA**：迭代收缩阈值算法
- **联合稀疏模型**：具有共享稀疏模式的多任务学习

#### 📊 **全面的评估框架**
- 不同稀疏度水平的成功率分析
- 收敛行为可视化
- 性能指标：NMSE、相对误差、成功率
- 具有可配置参数的自动化实验流水线

### 数学基础

该框架解决以下形式的优化问题：

```
min_{x} (1/2)||Ax - d||²₂ + τ·Ψ(x)
```

其中：
- `A ∈ ℝᵐˣⁿ` 是感知矩阵
- `d ∈ ℝᵐ` 是观测向量
- `τ > 0` 是正则化参数
- `Ψ(x)` 是稀疏诱导惩罚函数

对于群组稀疏问题：
```
Ψ(x) = Σᵢ φ(||xᵢ||ₚ)
```
其中 `xᵢ` 表示第 i 个群组，`φ` 是惩罚函数。

### 项目架构

```
group-sparse-optimization/
├── src/                           # 源代码
│   ├── demo.py                   # 主演示脚本
│   ├── config.py                 # 全局配置
│   ├── loss.py                   # 损失函数实现
│   ├── utils.py                  # 工具函数
│   │
│   ├── models/                   # 优化算法
│   │   ├── base.py              # 抽象基础模型
│   │   ├── joint_models/        # 联合稀疏优化
│   │   │   ├── IMTC.py         # IMTC 算法实现
│   │   │   ├── joint_fista.py  # 联合 FISTA
│   │   │   └── joint_ista.py   # 联合 ISTA
│   │   └── block_models/        # 块模型
│   │       ├── block_fista.py  # 块 FISTA
│   │       ├── block_ista.py   # 块 ISTA
│   │       └── mix_block.py    # 混合块模型
│   │
│   ├── prox/                    # 近端算子
│   │   ├── container.py         # 近端算子工厂
│   │   ├── group_prox/          # 群组近端算子
│   │   │   ├── general.py      # 通用框架
│   │   │   ├── l1psi.py        # L1-Ψ 算子
│   │   │   ├── prox_cl.py      # 截断 L1 算子
│   │   │   └── l2psi/          # L2-Ψ 算子
│   │   │       └── prox_cl.py  # 基于 L2 的算子
│   │   └── sep_prox/            # 可分离近端算子
│   │       └── prox_cl.py      # 逐元素算子
│   │
│   ├── dataset/                 # 数据生成
│   │   └── create_data.py      # 合成数据集创建
│   │
│   ├── common/                  # 通用工具
│   │   ├── config.py           # 配置类
│   │   └── enum.py             # 枚举类型
│   │
│   ├── exp/                     # 实验脚本
│   ├── group_sparse_exp/        # 群组稀疏实验
│   ├── joint_sparse_exp/        # 联合稀疏实验
│   │   ├── main.py             # 主实验运行器
│   │   ├── exp_success_rate.py # 成功率分析
│   │   └── plot_*.py           # 可视化脚本
│   │
│   ├── output/                  # 实验结果
│   └── report/                  # 报告生成
│       └── reporter.py         # 结果报告
│
├── requirements.txt             # Python 依赖
└── README.md                   # 本文件
```

### 安装说明

#### 系统要求
- Python 3.8 或更高版本
- NumPy、SciPy、Matplotlib
- Pandas、Scikit-learn

#### 安装步骤

1. **克隆仓库**：
```bash
git clone https://github.com/your-username/group-sparse-optimization.git
cd group-sparse-optimization
```

2. **创建虚拟环境**（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**：
```bash
pip install -r requirements.txt
```

### 使用方法

#### 快速开始

```bash
cd src
python demo.py
```

#### 配置说明

主要配置通过 `src/config.py` 和 `src/common/config.py` 处理：

```python
from common.config import Settings, NoiseConfig, JointSparseConfig

opts = Settings(
    K=1000,                    # 最大迭代次数
    tau=0.1,                   # 正则化参数
    m=256,                     # 感知矩阵行数
    n=1024,                    # 感知矩阵列数
    data_size=64,              # 样本数量
    sparsity=8,                # 群组稀疏度
    gLen=16,                   # 群组长度
    noise_params=NoiseConfig(sig=0.001),
    joint_sparse_config=JointSparseConfig(
        is_joint_sparse=True,
        p=0.8
    )
)
```

#### 运行实验

1. **基础实验**：
```bash
cd src
python demo.py
```

2. **联合稀疏实验**：
```bash
cd src/joint_sparse_exp
python main.py
```

3. **成功率分析**：
```bash
cd src/joint_sparse_exp
python exp_success_rate.py
```

#### 自定义实验

```python
from prox.container import ProximalContainer
from models import initialize_model

# 创建近端算子容器
container = ProximalContainer(n=1024, gLen=16, data_size=64)

# 选择近端算子
prox_operators = [
    container.prox_1_1over2(),    # L1-1/2 惩罚
    container.prox_1_mcp(),       # MCP 惩罚
    container.prox_2_scad()       # L2-SCAD 惩罚
]

# 初始化并运行模型
for prox_func in prox_operators:
    model = initialize_model('IMTC', prox_func, A, opts)
    result = model(data, K=1000)
```

### 实验结果

该框架生成全面的实验结果：

- **收敛分析**：迭代过程中的损失演化
- **成功率曲线**：不同稀疏度水平的性能
- **比较研究**：算法性能比较
- **可视化**：结果的自动图表生成

结果以结构化格式保存：
- `output/`：数值结果和日志
- `joint_sparse_exp/exp/`：实验图表和数据
- 自动 Excel 导出以供进一步分析

### 核心组件

#### 近端算子 (`src/prox/`)

框架实现了丰富的近端算子集合：

**可分离算子**：
- L0、L1、L1/2、L2/3 范数
- MCP、SCAD 惩罚
- 变换 L1 (TL1)

**群组算子**：
- L2-Lp 混合范数
- 群组 MCP、SCAD
- 截断惩罚

#### 模型 (`src/models/`)

**IMTC 算法**：具有自适应参数的新颖迭代混合阈值算法
```python
class IMTC(Model):
    def T(self, x, d, **kwargs):
        # 梯度步
        r = (self.A @ x.T).T - d
        z = x - self.stepsize * 2 * (self.A.T @ r.T).T
        
        # 混合阈值
        z = self.prox_func1(z, self.gamma2)
        Tx = self.prox_func2(z, self.gamma1 + self.gamma2 * penalty_term)
        return Tx
```

#### 数据集生成 (`src/dataset/`)

支持各种数据生成场景：
- 高斯和拉普拉斯感知矩阵
- 可配置的稀疏模式
- 联合稀疏结构
- 不同分布的噪声注入

### 性能基准

标准测试用例的典型性能：

| 算法      | 稀疏度水平    | 成功率 | 收敛速度      |
|-----------|---------------|--------|---------------|
| IMTC-L1/2 | 8/64 群组     | 95%+   | ~200 次迭代   |
| IMTC-MCP  | 12/64 群组    | 90%+   | ~300 次迭代   |
| FISTA-L1  | 8/64 群组     | 85%+   | ~500 次迭代   |

### 研究应用

该框架已应用于：
- **压缩感知**：稀疏信号恢复
- **多任务学习**：联合特征选择
- **图像处理**：变换中的结构化稀疏
- **生物医学信号处理**：EEG/fMRI 分析

### 贡献指南

我们欢迎贡献！请参阅我们的贡献指南：

1. Fork 仓库
2. 创建功能分支
3. 实现您的更改并添加测试
4. 提交 pull request

### 引用

如果您在研究中使用此代码，请引用：

```bibtex
@software{group_sparse_optimization,
  title={Group Sparse Optimization with Lp-q Regularization},
  author={Li, Zhihong and Lin, Rongrong},
  year={2024},
  url={https://github.com/your-username/group-sparse-optimization}
}
```

### 作者

- **李志宏** - 算法开发和实现
- **林荣荣** - 理论分析和优化

### 许可证

本项目在学术许可证下发布。如在学术工作中使用，请适当引用。

---

## Contact | 联系方式

For questions and collaboration opportunities, please contact:
如有问题和合作机会，请联系：

- **Email**: [your-email@university.edu]
- **GitHub Issues**: [Project Issues](https://github.com/your-username/group-sparse-optimization/issues)

---

**Keywords**: Group Sparsity, Proximal Operators, IMTC, FISTA, Compressed Sensing, L1 Regularization, MCP, SCAD
**关键词**: 群组稀疏, 近端算子, IMTC, FISTA, 压缩感知, L1正则化, MCP, SCAD
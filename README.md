# Group Sparse Optimization with Lp-q Regularization

[English](#english) | [中文](#中文)

---

## English

### Overview

This project implements and compares various group sparse optimization algorithms with Lp-q regularization. It focuses on proximal gradient methods and different proximal operators for solving group sparse problems in signal processing and machine learning.

### Features

- **Multiple Proximal Operators**: Implementation of various proximal operators including:
  - Capped L1 proximal operator
  - Capped 1/2 proximal operator  
  - L2-based proximal operators (2/3, arctan, logsum)
  - General proximal L2-Ψ operators

- **Optimization Algorithms**: 
  - PGAC-JSO (Proximal Gradient Algorithm with Convergence - Joint Sparse Optimization)
  - FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
  - ISTA (Iterative Shrinkage-Thresholding Algorithm)

- **Comprehensive Evaluation**: 
  - Success rate analysis across different sparsity levels
  - Loss comparison and convergence analysis
  - Visualization tools for results

### Project Structure

```
group_optimize_lpq/
├── src/                    # Source code
│   ├── main.py            # Main execution script
│   ├── config.py          # Configuration settings
│   ├── loss.py            # Loss function implementations
│   ├── utils.py           # Utility functions
│   ├── models/            # Model implementations
│   │   ├── block_models/  # Block-wise models
│   │   └── joint_models/  # Joint sparse models
│   ├── prox/              # Proximal operators
│   │   ├── group_prox/    # Group-wise proximal operators
│   │   └── sep_prox/      # Separable proximal operators
│   ├── dataset/           # Dataset creation and management
│   ├── common/            # Common utilities and enums
│   └── report/            # Report generation
├── history/               # Experimental history and results
├── output/                # Output files and logs
├── doc/                   # Documentation
└── requirements.txt       # Python dependencies
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd group_optimize_lpq
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. **Basic Usage**:
```bash
cd src
python main.py
```

2. **Configuration**: 
   - Modify `src/config.py` to adjust experimental parameters
   - Key parameters include:
     - `K`: Number of iterations (default: 1000)
     - `tau`: Regularization parameter (default: 0.1)
     - `m`, `n`: Matrix dimensions (default: 256×1024)
     - `sparsity`: Group sparsity level (default: 8)
     - `gLen`: Group length (default: 16)

3. **Running Experiments**:
   - The main script compares different algorithms and proximal operators
   - Results are saved in the `output/` directory
   - Plots are generated showing convergence behavior

### Key Components

- **Proximal Operators**: Located in `src/prox/`
  - Group proximal operators for structured sparsity
  - Separable proximal operators for element-wise operations

- **Models**: Located in `src/models/`
  - Block model factory for creating optimization models
  - Support for various optimization algorithms

- **Dataset Generation**: Located in `src/dataset/`
  - Synthetic data generation for sparse coding problems
  - Configurable noise and sparsity patterns

### Results

The project generates comprehensive results including:
- Convergence plots comparing different algorithms
- Success rate analysis across sparsity levels
- Performance metrics and timing comparisons

Results are automatically saved to the `output/` directory and experimental history is maintained in the `history/` folder.

### Authors

- Zhihong Li
- Rongrong Lin

### License

This project is for research purposes. Please cite appropriately if used in academic work.

---

## 中文

### 项目概述

本项目实现并比较了各种带有 Lp-q 正则化的群组稀疏优化算法。重点关注近端梯度方法和不同的近端算子，用于解决信号处理和机器学习中的群组稀疏问题。

### 主要特性

- **多种近端算子**：实现了多种近端算子，包括：
  - 截断 L1 近端算子
  - 截断 1/2 近端算子
  - 基于 L2 的近端算子（2/3、反正切、对数和）
  - 通用近端 L2-Ψ 算子

- **优化算法**：
  - PGAC-JSO（带收敛性的近端梯度算法 - 联合稀疏优化）
  - FISTA（快速迭代收缩阈值算法）
  - ISTA（迭代收缩阈值算法）

- **全面评估**：
  - 不同稀疏度水平的成功率分析
  - 损失比较和收敛性分析
  - 结果可视化工具

### 项目结构

```
group_optimize_lpq/
├── src/                    # 源代码
│   ├── main.py            # 主执行脚本
│   ├── config.py          # 配置设置
│   ├── loss.py            # 损失函数实现
│   ├── utils.py           # 工具函数
│   ├── models/            # 模型实现
│   │   ├── block_models/  # 块模型
│   │   └── joint_models/  # 联合稀疏模型
│   ├── prox/              # 近端算子
│   │   ├── group_prox/    # 群组近端算子
│   │   └── sep_prox/      # 可分离近端算子
│   ├── dataset/           # 数据集创建和管理
│   ├── common/            # 通用工具和枚举
│   └── report/            # 报告生成
├── history/               # 实验历史和结果
├── output/                # 输出文件和日志
├── doc/                   # 文档
└── requirements.txt       # Python 依赖
```

### 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd group_optimize_lpq
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 使用方法

1. **基本使用**：
```bash
cd src
python main.py
```

2. **配置**：
   - 修改 `src/config.py` 来调整实验参数
   - 主要参数包括：
     - `K`：迭代次数（默认：1000）
     - `tau`：正则化参数（默认：0.1）
     - `m`, `n`：矩阵维度（默认：256×1024）
     - `sparsity`：群组稀疏度（默认：8）
     - `gLen`：群组长度（默认：16）

3. **运行实验**：
   - 主脚本比较不同的算法和近端算子
   - 结果保存在 `output/` 目录中
   - 生成显示收敛行为的图表

### 核心组件

- **近端算子**：位于 `src/prox/`
  - 用于结构化稀疏的群组近端算子
  - 用于逐元素操作的可分离近端算子

- **模型**：位于 `src/models/`
  - 用于创建优化模型的块模型工厂
  - 支持各种优化算法

- **数据集生成**：位于 `src/dataset/`
  - 稀疏编码问题的合成数据生成
  - 可配置的噪声和稀疏模式

### 结果

项目生成全面的结果，包括：
- 比较不同算法的收敛图
- 跨稀疏度水平的成功率分析
- 性能指标和时间比较

结果自动保存到 `output/` 目录，实验历史保存在 `history/` 文件夹中。

### 作者

- 李志宏
- 林荣荣

### 许可证

本项目用于研究目的。如在学术工作中使用，请适当引用。
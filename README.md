# Group Sparse Optimization (src/)

Language | 语言
- [简体中文](#简体中文)
- [English](#english)

---

## 简体中文

> 组稀疏（Group Sparsity）/联合稀疏（Joint Sparsity）优化研究项目。核心代码在 src/，包含：迭代模型（ISTA / FISTA / GROUPPGAC）、近端算子族（ℓ1/ℓ0/ℓ1/2/ℓ2/3/MCP/SCAD/TL1/Cappedℓ1/2 等）、数据生成、实验框架、结构化日志与绘图。

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](./)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./)
[![TypeHints](https://img.shields.io/badge/code-typed-informational.svg)](./)

### 目录（Table of Contents）
- [项目定位与特性](#项目定位与特性)
- [环境与安装](#环境与安装)
- [快速开始](#快速开始)
- [目录结构（src/）](#目录结构src)
- [命令与运行](#命令与运行)
- [API 概览](#api-概览)
- [实验复现流程](#实验复现流程)
- [开发与贡献](#开发与贡献)
- [版本与兼容性](#版本与兼容性)
- [许可证与引用](#许可证与引用)
- [联系方式](#联系方式)

---

### 项目定位与特性
- 研究主题：线性观测 b = A x + noise 下的组/联合稀疏重建与优化。
- 算法实现：ISTA、FISTA、GROUPPGAC 等迭代法，核心步骤为近端映射（Proximal Mapping）。
- 近端算子族：支持凸与非凸（ℓ1、ℓ0、ℓ1/2、ℓ2/3、MCP、SCAD、TL1、Cappedℓ1、Cappedℓ1/2；含 ℓ1/3 原型）。
- 指标：NMSE、Relative Error、SUCCESS_TIMES（成功率）、收敛曲线。
- 工程能力：
  - Pydantic v2 配置（Settings、NoiseConfig、JointSparseConfig、LogConfig）
  - 依赖注入（dependency-injector）：ProximalContainer 统一产出算子实例
  - 结构化日志（structlog）：report/Logger 单例
  - 可视化：figure_generater（JSON 配置 + 对数坐标）

---

### 环境与安装
- 要求：Python 3.10+
- 安装（PowerShell）
  ```
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -U pip
  pip install numpy pandas matplotlib pydantic structlog dependency-injector munch
  ```
- 可选：使用 requirements.txt 锁定版本以确保复现性（如需我生成，请告知目标版本）。

---

### 快速开始
- 演示（FISTA + CL/L1/2 家族）：
  ```
  python src/demo.py
  ```
- ℓ1/3 非凸近端实验（GROUPPGAC 原型）：
  ```
  python src/demo_l1over3.py
  ```
- 完整实验入口：
  ```
  # 组稀疏
  python src/group_sparse_exp/exp_compare_loss.py
  python src/group_sparse_exp/exp_success_rate.py

  # 联合稀疏
  python src/joint_sparse_exp/exp_compare_loss.py
  python src/joint_sparse_exp/exp_success_rate.py
  ```

---

### 目录结构（src/）
- config.py：旧版噪声参数草案（NoiseParams）；推荐使用 common/config.py
- loss.py：指标/目标（calculate_normalized_mean_squared_error、objective_val）
- utils.py：通用工具（save_df）
- common/
  - common/config.py：Settings、NoiseConfig、JointSparseConfig、LogConfig；DistributionType、ObjectiveType、SeparableSparsityMode
  - common/enum.py：枚举定义
- dataset/
  - dataset/create_data.py：create_sc_dataset → ((x_test, d_test), A, b)
- experiment/
  - experiment/__init__.py：Experiment 抽象；Record/RecordContainer（Pydantic）
  - experiment/experiment.py：MSELossExperiment、SuccessRateExperiment
- models/
  - models/base.py：Model/GroupModel/JointModel 抽象
  - models/block_models/：ISTA、FISTA、GROUPPGAC
  - models/joint_models/：JointISTA、JointFISTA、PGAC
- prox/
  - prox/__init__.py：ProximalOperator / GroupProximalOperator 抽象
  - prox/container.py：ProximalContainer（依赖注入工厂）
  - prox/sep_prox/：分量型近端族（ℓ1、ℓ1/2、ℓ2/3、ℓ0、MCP、SCAD、TL1、Cappedℓ1、Cappedℓ1/2）；dev.py 含 ℓ1/3
  - prox/group_prox/：组型近端族（L1_1/2、L1_2/3、GeneralProxL2Psi）
  - prox/group_prox/l2psi/prox_cl.py：L2_ψ（|·|、|·|^{1/2}、|·|^{2/3}、LOG、Arctan、CL1/2）
- figure_generater/
  - plot_config.py（Pydantic 配置）、plot.py；plot_config_compare_exp.json
- report/
  - reporter.py：Logger 单例、mkdir
- group_sparse_exp/ 与 joint_sparse_exp/
  - exp_compare_loss.py、exp_success_rate.py（含 if __name__ == '__main__'）

---

### 命令与运行
- 列出含入口的脚本：
  ```
  Get-ChildItem src -Recurse -Filter *.py | Select-String "__name__ == '__main__'" -List
  ```
- 常用命令：
  ```
  # 组稀疏对比实验
  python src/group_sparse_exp/exp_compare_loss.py

  # 联合稀疏成功率实验
  python src/joint_sparse_exp/exp_success_rate.py

  # 生成图像（需 figure_generater 配置）
  python src/demo_l1over3.py
  ```

---

### API 概览
- 配置（src/common/config.py）
  - Settings：K、tau、m、n、data_size、dist、gLen、sparsity、objective、plot_figs、log_config、noise_params、joint_sparse_config
  - JointSparseConfig：is_joint_sparse、mode（percentage 等）、p
  - NoiseConfig：sig；LogConfig：file_dir、file_name、file_mode
- 数据（src/dataset/create_data.py）
  - create_sc_dataset(opts) -> ((x_test, d_test), A, b)
- 指标（src/loss.py）
  - calculate_normalized_mean_squared_error(x, x_gt)
  - objective_val(x, d, x_gt, objective): 'Repeat NMSE' / 'GT' / 'NMSE' / 'RELATIVE' / 'SUCCESS_TIMES'
- 近端（src/prox/）
  - 抽象：ProximalOperator / GroupProximalOperator（prox、obj、name、latex_name）
  - 分量型：ProxL1、ProxL1over2、ProxL2over3、ProxL0、ProxMCP、ProxSCAD、ProxTransformedl1（TL1）、ProxCappedL1、ProxCapped1over2
  - 组型：ProxL1_1over2、ProxL1_2over3、GeneralProxL2Psi、L2_ψ 系列
  - 容器：ProximalContainer（依赖注入与组合）
- 模型（src/models/）
  - 抽象：Model、GroupModel、JointModel
  - 组稀疏：ISTA、FISTA、GROUPPGAC；联合稀疏：JointISTA、JointFISTA、PGAC
- 日志（src/report/reporter.py）
  - Logger 单例：`Logger(**opts.log_config.model_dump()).logger`

---

### 实验复现流程
1) 准备环境与依赖（建议锁定版本）
2) 选择脚本（demo / demo_l1over3 / group_* / joint_*）
3) 配置 Settings（K、tau、sparsity、gLen、data_seed、objective）
4) 使用 ProximalContainer 组合近端族
5) 运行 MSELossExperiment 或 SuccessRateExperiment
6) 使用 figure_generater 绘制曲线（可选）
7) 查看日志输出与图像（默认 `1.png`）

---

### 开发与贡献
- 风格：PEP8 + 类型注解；Pydantic v2 用于配置与模型
- 架构：函数式优先、模块化；src/ 下分层清晰（common/dataset/experiment/models/prox/report）
- 扩展：
  - 近端：在 prox/sep_prox/ 或 prox/group_prox/ 中新增，遵循抽象接口；接入 ProximalContainer
  - 模型：在 models/block_models/ 或 models/joint_models/ 中新增，继承基类
  - 实验：在 experiment/ 中实现，入口放 group_sparse_exp/ 或 joint_sparse_exp/
- PR 规范：包含变更说明、最小复现实例（命令、配置、日志/图像）
- Issues：附环境信息、配置、复现步骤与日志片段

---

### 版本与兼容性
- Python：3.10+（建议）
- 依赖：numpy / pandas / matplotlib / pydantic v2 / structlog / dependency-injector / munch
- 非凸近端（ℓ1/3、ℓ1/2、TL1、Cappedℓ1/2）对阈值与数值稳定性敏感；tau 调参建议结合 Lipschitz 估计。

---

### 许可证与引用
- License：如未声明，默认研究用途；其他用途请征询作者授权
- Citation：如使用本仓库或近端设计，请在论文/报告中引用作者与仓库链接

---

### 联系方式
- Authors: Zhihong Li (gealachlee@126.com), Rongrong Lin (linrr@gdut.edu.cn)
- Issues：请在仓库 Issue 区反馈

---

## English

> An engineering and reproducible project for Group/Joint Sparse Optimization. Core code under src/ includes iterative models (ISTA / FISTA / GROUPPGAC), a rich proximal operator family (ℓ1/ℓ0/ℓ1/2/ℓ2/3/MCP/SCAD/TL1/Cappedℓ1/2), data generation, experiment framework, structured logging, and plotting.

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](./)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./)
[![TypeHints](https://img.shields.io/badge/code-typed-informational.svg)](./)

### Table of Contents
- [Features](#features)
- [Environment & Installation](#environment--installation)
- [Quick Start](#quick-start)
- [Folder Tree (src/)](#folder-tree-src)
- [Commands & Running](#commands--running)
- [API Overview](#api-overview)
- [Reproducibility Pipeline](#reproducibility-pipeline)
- [Development & Contributing](#development--contributing)
- [Version & Compatibility](#version--compatibility)
- [License & Citation](#license--citation)
- [Contact](#contact)

---

### Features
- Topic: Sparse reconstruction under group/joint priors (b = A x + noise).
- Algorithms: ISTA, FISTA, GROUPPGAC with proximal mappings.
- Proximal families: convex and nonconvex (ℓ1, ℓ0, ℓ1/2, ℓ2/3, MCP, SCAD, TL1, Cappedℓ1/2; ℓ1/3 prototype).
- Metrics: NMSE, Relative Error, SUCCESS_TIMES; convergence curves and success rates.
- Engineering:
  - Pydantic v2 settings (Settings, NoiseConfig, JointSparseConfig, LogConfig).
  - Dependency injection via ProximalContainer.
  - Structured logging with structlog (singleton Logger).
  - Plotting via figure_generater with JSON config and log-scale axes.

---

### Environment & Installation
- Requirements: Python 3.10+
- Install (Windows PowerShell):
  ```
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -U pip
  pip install numpy pandas matplotlib pydantic structlog dependency-injector munch
  ```
- Optional: pin versions in requirements.txt for reproducibility.

---

### Quick Start
- Demo (FISTA + CL/L1/2 family):
  ```
  python src/demo.py
  ```
- ℓ1/3 nonconvex proximal experiment (GROUPPGAC prototype):
  ```
  python src/demo_l1over3.py
  ```
- Full entrypoints:
  ```
  # Group sparsity
  python src/group_sparse_exp/exp_compare_loss.py
  python src/group_sparse_exp/exp_success_rate.py

  # Joint sparsity
  python src/joint_sparse_exp/exp_compare_loss.py
  python src/joint_sparse_exp/exp_success_rate.py
  ```

---

### Folder Tree (src/)
- config.py: legacy NoiseParams (prefer common/config.py)
- loss.py: metrics/objectives (calculate_normalized_mean_squared_error, objective_val)
- utils.py: utilities (save_df)
- common/
  - common/config.py: Settings, NoiseConfig, JointSparseConfig, LogConfig; DistributionType, ObjectiveType, SeparableSparsityMode
  - common/enum.py: enums
- dataset/
  - dataset/create_data.py: create_sc_dataset → ((x_test, d_test), A, b)
- experiment/
  - experiment/__init__.py: Experiment abstraction; Record/RecordContainer (Pydantic)
  - experiment/experiment.py: MSELossExperiment, SuccessRateExperiment
- models/
  - models/base.py: Model/GroupModel/JointModel abstractions
  - block_models/: ISTA, FISTA, GROUPPGAC
  - joint_models/: JointISTA, JointFISTA, PGAC
- prox/
  - prox/__init__.py: ProximalOperator / GroupProximalOperator abstractions
  - container.py: ProximalContainer (DI factory)
  - sep_prox/: separable proximals (ℓ1, ℓ1/2, ℓ2/3, ℓ0, MCP, SCAD, TL1, Cappedℓ1, Cappedℓ1/2); dev.py with ℓ1/3
  - group_prox/: group proximals (L1_1/2, L1_2/3, GeneralProxL2Psi)
  - group_prox/l2psi/prox_cl.py: L2_ψ series (|·|, |·|^{1/2}, |·|^{2/3}, LOG, Arctan, CL1/2)
- figure_generater/
  - plot_config.py (Pydantic), plot.py; plot_config_compare_exp.json
- report/
  - reporter.py: Logger (structlog singleton), mkdir
- group_sparse_exp/ & joint_sparse_exp/
  - exp_compare_loss.py, exp_success_rate.py (with if __name__ == '__main__')

---

### Commands & Running
- List main-entry scripts:
  ```
  Get-ChildItem src -Recurse -Filter *.py | Select-String "__name__ == '__main__'" -List
  ```
- Examples:
  ```
  python src/group_sparse_exp/exp_compare_loss.py
  python src/joint_sparse_exp/exp_success_rate.py
  python src/demo_l1over3.py  # saves 1.png by default
  ```

---

### API Overview
- Config (src/common/config.py)
  - Settings: K, tau, m, n, data_size, dist, gLen, sparsity, objective, plot_figs, log_config, noise_params, joint_sparse_config
  - JointSparseConfig: is_joint_sparse, mode (percentage), p
  - NoiseConfig: sig; LogConfig: file_dir, file_name, file_mode
- Data (src/dataset/create_data.py)
  - create_sc_dataset(opts) -> ((x_test, d_test), A, b)
- Metrics (src/loss.py)
  - calculate_normalized_mean_squared_error(x, x_gt)
  - objective_val(x, d, x_gt, objective): 'Repeat NMSE' / 'GT' / 'NMSE' / 'RELATIVE' / 'SUCCESS_TIMES'
- Proximal (src/prox/)
  - Abstract: ProximalOperator / GroupProximalOperator
  - Separable: ProxL1, ProxL1over2, ProxL2over3, ProxL0, ProxMCP, ProxSCAD, ProxTransformedl1 (TL1), ProxCappedL1, ProxCapped1over2
  - Group: ProxL1_1over2, ProxL1_2over3, GeneralProxL2Psi, L2_ψ series
  - Container: ProximalContainer for DI and composition
- Models (src/models/)
  - Abstract: Model, GroupModel, JointModel
  - Group: ISTA, FISTA, GROUPPGAC; Joint: JointISTA, JointFISTA, PGAC
- Logging (src/report/reporter.py)
  - Logger singleton: `Logger(**opts.log_config.model_dump()).logger`

---

### Reproducibility Pipeline
1) Prepare environment (pin deps if needed)
2) Choose script (demo / demo_l1over3 / group_* / joint_*)
3) Configure Settings (K, tau, sparsity, gLen, data_seed, objective)
4) Compose proximal families via ProximalContainer
5) Run MSELossExperiment / SuccessRateExperiment
6) Plot with figure_generater (optional)
7) Inspect logs and figures (default `1.png`)

---

### Development & Contributing
- Style: PEP8 + type hints; Pydantic v2 for config/models
- Architecture: functional-first, modular; layered under src/ (common/dataset/experiment/models/prox/report)
- Extensions:
  - Proximals: add under prox/sep_prox/ or prox/group_prox/ per abstract interfaces; wire into ProximalContainer
  - Models: add under models/block_models/ or models/joint_models/
  - Experiments: implement under experiment/ and add entry in group_sparse_exp/ or joint_sparse_exp/
- PRs: include change summary and minimal repro (commands, settings, logs/figures)
- Issues: include env, settings, repro steps, and log snippets

---

### Version & Compatibility
- Python: 3.10+ (recommended)
- Deps: numpy / pandas / matplotlib / pydantic v2 / structlog / dependency-injector / munch
- Notes: Nonconvex proximals (ℓ1/3, ℓ1/2, TL1, Cappedℓ1/2) are sensitive to thresholds and stability; tune tau with Lipschitz estimates when possible.

---

### License & Citation
- License: If unspecified, assume research-only; seek authors’ permission for other uses.
- Citation: Cite the authors and repository when using this codebase or proximal designs.

---

### Contact
- Authors: Zhihong Li (gealachlee@126.com), Rongrong Lin (linrr@gdut.edu.cn)
- Issues: Use repository Issues for questions and suggestions
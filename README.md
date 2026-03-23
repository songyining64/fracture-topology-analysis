# 地质断裂拓扑智能分析系统

面向地质工程领域的断裂网络拓扑分析与机器学习融合系统，支持准噶尔盆地、塔里木盆地、柯坪断隆等多区域断裂数据的一体化分析与勘探价值评估。

---

## 一、作品简介

本系统将**断裂网络拓扑分析**与**机器学习**相结合，实现从原始迹线数据到融合分析、预测与可解释性的完整流程。核心功能包括：

- **拓扑分析**：基于 fractopo 进行分支/节点分类（CC/CI/II、X/Y/I）、玫瑰图、三元图、密度热力图、等值线等可视化
- **多维融合**：支持 PCA、自编码器、UMAP、VAE 等多种降维融合，以及专家规则加权、GAT 图注意力网络融合
- **机器学习**：XGBoost 回归/分类、Optuna 超参调优、SHAP 可解释性分析
- **空间-拓扑融合**：特征工程 → 加权融合 → GAT/GraphSAGE → 多尺度金字塔 → XGBoost → SHAP 的端到端流水线

支持准噶尔盆地车莫古隆起、柯坪断隆KB11、塔里木盆地英买2 三个典型区域数据的自由切换。

---

## 二、开源代码与组件使用情况说明

### 2.1 核心依赖（开源组件）

| 组件 | 版本 | 用途 | 协议 |
|------|------|------|------|
| fractopo | ≥0.6.0 | 断裂网络拓扑分析、分支节点分类、网格采样 | MIT |
| geopandas | ≥0.12.0 | 地理数据读写与几何计算 | BSD |
| PyQt5 | ≥5.15 | 桌面 GUI 界面 | GPL/商业双协议 |
| scikit-learn | ≥1.0 | 特征工程、PCA、聚类、归一化 | BSD |
| xgboost | ≥1.6 | 梯度提升回归/分类 | Apache 2.0 |
| shap | ≥0.42 | 模型可解释性分析 | MIT |
| optuna | ≥3.0 | 超参数优化 | MIT |
| matplotlib | ≥3.5 | 图表绘制 | PSF |
| numpy / pandas / scipy | - | 数值计算与数据处理 | BSD |
| networkx | - | 图结构、骨架分析 | BSD |
| ternary | - | 三元图 | MIT |
| powerlaw | - | 幂律拟合 | MIT |
| pyshp | - | Shapefile 读写 | MIT |

### 2.2 可选依赖

| 组件 | 用途 |
|------|------|
| torch | 自编码器、VAE、自适应加权、GAT |
| umap-learn | UMAP 降维融合 |
| torch_geometric | GAT、GraphSAGE 图神经网络 |
| scikit-image / opencv-python | 栅格图像预处理、骨架化（new plot 模块） |

### 2.3 自研代码说明

- **fusion_algorithm.py**：加权融合、自适应加权（MLP）、GAT 融合、网格图构建等为自研实现
- **topology_fusion.py**：PCA/AE/UMAP/VAE 融合与聚类流水线为自研封装
- **spatial_topology_framework.py**：空间-拓扑融合学习完整流水线为自研设计
- **multiscale_features.py**：多尺度特征金字塔为自研实现
- **gnn_embeddings.py**：GraphSAGE/GAT/GIN 图嵌入为基于 PyTorch Geometric 的封装与调用
- **feature_engineering.py**：特征工程流水线（归一化、异常值处理、方差/互信息筛选）为自研实现
- **utils/matplotlib_chinese.py**：跨平台中文字体配置为自研工具

---

## 三、作品安装说明

### 3.1 环境要求

- Python ≥ 3.9（推荐 3.10 或 3.11）
- Windows / macOS / Linux

### 3.2 安装步骤

```bash
# 1. 克隆或下载项目
cd 断裂拓扑分析

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 可选：若需自编码器/VAE/UMAP/GAT 功能
pip install torch
pip install umap-learn
pip install torch torch_geometric  # GAT 融合
```

### 3.3 运行主界面

```bash
cd program
python main.py
```

### 3.4 数据准备

- 迹线与研究区数据置于 `program/THK/`、`program/KB11/`、`program/MY/` 对应目录
- 融合/ML 需先运行网格导出生成 CSV：
  ```bash
  python export_grid_csv.py THK   # 或 KB11、MY
  ```

---

## 四、设计思路

### 4.1 整体架构

系统采用「数据层 → 拓扑分析层 → 融合层 → 机器学习层 → 展示层」的分层设计：

```
┌─────────────────────────────────────────────────────────────────┐
│                        展示层（PyQt5 GUI）                        │
│  数据源切换 | 拓扑可视化 | 融合分析 | ML训练 | SHAP解释            │
├─────────────────────────────────────────────────────────────────┤
│                      机器学习层                                   │
│  XGBoost | Optuna 调参 | SHAP 可解释性 | 推理与评估               │
├─────────────────────────────────────────────────────────────────┤
│                      融合层                                      │
│  加权融合 | GAT | 自适应加权 | PCA/AE/UMAP/VAE | 多尺度金字塔     │
├─────────────────────────────────────────────────────────────────┤
│                      拓扑分析层（fractopo）                       │
│  分支节点分类 | 网格采样 | 拓扑参数 | 玫瑰图/三元图/等值线         │
├─────────────────────────────────────────────────────────────────┤
│                      数据层                                      │
│  GeoJSON 迹线/研究区 | 网格 CSV（拓扑+几何指标）                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 核心流程设计

1. **拓扑特征提取**：基于 fractopo 对断裂迹线进行分支/节点拓扑分类，按网格采样得到每个单元的拓扑指标（断裂强度、连通率、迹线长度等）
2. **特征工程**：归一化、异常值处理、方差/互信息筛选，构建标准特征矩阵
3. **多维融合**：将高维拓扑特征通过 PCA/深度学习方法降至低维，并结合专家规则加权、GAT 图模型进行空间-拓扑融合
4. **预测与解释**：XGBoost 回归预测目标属性，SHAP 量化各特征贡献，形成可解释的勘探价值评估

### 4.3 配置驱动

通过 `config.yaml` 统一管理高价值属性、融合权重、训练参数等，便于调优与复现。

---

## 五、设计重点与难点

### 5.1 重点

| 重点 | 说明 |
|------|------|
| **地质领域适配** | 将拓扑指标与油气勘探中的「连通率」「断裂强度」等高价值属性关联，设计专家规则加权策略 |
| **多源数据统一** | 支持三区数据切换，统一迹线/研究区/网格 CSV 的对应关系，保证拓扑分析与融合分析数据一致性 |
| **融合链路完整性** | 将传统拓扑分析、多种降维方法、图神经网络、树模型、可解释性串联成可复现的端到端流水线 |
| **跨平台与鲁棒性** | 中文字体跨平台配置、目标列恒定的校验、样本不足时的友好提示、多尺度金字塔在非规则网格下的降级处理 |

### 5.2 难点

| 难点 | 解决方式 |
|------|----------|
| **网格与图结构的对应** | 将规则网格抽象为邻接图，通过 `build_grid_graph` 构建边索引，使 GAT/GraphSAGE 能利用空间邻接关系 |
| **融合方法的选择与组合** | 提供多种融合方式（规则加权、自适应加权、GAT、PCA/VAE 等），用户可根据数据规模与需求选择，并支持加权 vs GAT 的对比实验 |
| **目标列的有效性** | 识别恒定目标（如网格面积）导致的回归失效，默认推荐有变化的拓扑指标作为目标，并做方差校验 |
| **样本过滤后的兼容** | 特征工程会过滤全零行，导致样本数变化；多尺度金字塔需样本数满足规则网格，否则自动跳过多尺度 |
| **可选依赖的降级** | GAT、VAE、UMAP 依赖 PyTorch 等，未安装时相关功能优雅降级，不影响主流程运行 |

---

## 六、目录结构概览

```
断裂拓扑分析/
├── README.md                 # 本文件
├── requirements.txt          # 依赖列表
├── 运行说明.md               # 运行指南
├── 项目说明-模块与使用.md     # 模块详细说明
├── config.yaml               # 融合/训练等配置（program 下）
└── program/
    ├── main.py               # GUI 主入口
    ├── demo.py               # 界面布局定义
    ├── export_grid_csv.py    # 网格 CSV 导出
    ├── feature_engineering.py# 特征工程
    ├── fusion_algorithm.py   # 融合算法
    ├── topology_fusion.py    # 属性融合（PCA/AE/UMAP/VAE）
    ├── spatial_topology_framework.py  # 空间-拓扑融合流水线
    ├── multiscale_features.py# 多尺度特征
    ├── gnn_embeddings.py     # 图神经网络嵌入
    ├── evaluation.py         # 评估指标
    ├── ml/                   # 机器学习模块
    │   ├── train.py          # XGBoost 训练
    │   ├── tune.py           # Optuna 调参
    │   ├── infer.py          # 推理
    │   └── explain.py        # SHAP 解释
    ├── utils/                # 工具模块
    ├── THK/ KB11/ MY/        # 区域数据目录
    └── new plot/             # 栅格图分析（可选）
```

---

## 七、快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主界面
cd program && python main.py

# 选择数据源 → 点击「原断裂数据地图」等按钮进行拓扑分析
# 或运行「一键空间-拓扑融合」进行完整 ML 流程（需先 export_grid_csv 生成对应区域 CSV）
```

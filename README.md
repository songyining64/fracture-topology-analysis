# 油气区断裂网络连通性智能分析与预测系统

**油气区断裂网络连通性智能分析与预测系统**面向**油气勘探工区**，将 **fractopo 断裂网络拓扑**与 **连通性相关指标**（如分支/迹线连接度、连接频率等）和 **机器学习预测**放在同一套桌面工作流中：从原始迹线 + 研究区到网格化拓扑属性、多维融合与聚类，再到 XGBoost 等指标预测与 SHAP 解释，实现「看得清（可视化）—算得准（建模）—讲得明（解释）」。默认示例数据为塔里木盆地英买 2（MY）。

---

## 一、作品简介

**核心功能包括：**

- **拓扑与连通性分析**：基于 fractopo 进行分支类型划分（CC / CI / II）与节点类型划分（X / Y / I），输出全局拓扑参数、**连通性相关统计**（如 Connections per Branch/Trace、Connection Frequency 等）、方位集交叉与相邻关系及系列成果图；
- **可视化成果**：支持原始断裂地图、分类地图、拓扑化展示、方位角集图、密度热力图、玫瑰图、三元图、等值线/轮廓图，以及交叉与相邻（cross-cut / abutting）关系图等，用于直观刻画断裂几何—拓扑特征。
- **多维融合**：支持 PCA、自编码器、UMAP、VAE 等多种降维融合，以及专家规则加权、GAT 图注意力网络融合，便于进行区块对比与预测。
- **机器学习**：XGBoost 回归/分类（用于断裂强度、连通性等相关指标的估计与泛化评估）、Optuna 超参调优、SHAP 可解释性分析，使预测结果可追溯、可核对。
- **空间-拓扑融合**：特征工程 → 加权融合 → GAT/GraphSAGE → 多尺度金字塔 → XGBoost → SHAP 的端到端流水线，实现从空间—拓扑表征到可解释预测的闭环。

当前默认数据为**塔里木盆地英买2（MY）**（`program/MY/` 迹线与研究区；网格 CSV 可导出用于 GIS 与进一步分析）。其他工区可在本地自行扩展 `DATA_SOURCES`，与仓库默认配置无关。

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
```

若希望按**经过测试的版本组合**直接复现，推荐使用仓库根目录的 `environment.yml`：

```bash
conda env create -f environment.yml
conda activate fracture-connectivity
```

### 3.3 运行主界面

```bash
cd program
python main.py
```
也可在项目根目录一条命令启动（无需先 cd program）：
```bash
py program\main.py
```
### 3.4 数据准备

- 迹线与研究区数据置于 `program/MY/`（英买 2）；若扩展多工区可增设子目录并在 `DATA_SOURCES` 中配置
- 融合/ML 需先运行网格导出生成 CSV：
  ```bash
  python export_grid_csv.py MY   # 英买 2 网格 CSV（默认参数亦可省略）
  ```

---

### 3.5 本系统如何定义和量化断裂网络连通性

在本项目中，「连通性」主要指**断裂网络在几何–拓扑上的连接难易程度**，由 fractopo 在网格尺度上输出的、可直接进入机器学习的一套标量指标来表示：

- **Connections per Branch**（每个分支上的连接点数）：分支越常被其他结构“挂靠”，该值越高，反映局部网络越密集、越易于形成渗流路径。
- **Connections per Trace**（每条迹线上的连接点数）：刻画迹线作为断裂段在端点/交点处被网络连接的程度。
- **Connection Frequency**（连接频率类指标）：在网格内对连接事件或连接密度的统计频率，用于粗粒度对比不同区块的连通活跃程度。

这三项在代码中集中定义为 **「连通性特征组」**（`CONNECTIVITY_FEATURE_COLUMNS` / `config.yaml` 中 `high_value_attrs`），在加权融合时默认赋予更高权重；训练完成后，界面与日志中会单独给出 **连通性特征的 SHAP 排名**及其在 mean|SHAP| 归一意义下的 **累计贡献占比**，便于说明「哪些连通指标在驱动当前目标列」。

> 注意：特征工程阶段可能因方差过小或互信息筛选会**暂时去掉**某一连通列；若 SHAP 表中没有某项，通常表示该列未进入最终模型特征子集。

### 3.6 在 QGIS / ArcGIS 中快速使用导出图层

1. **加载 GPKG**：在 QGIS 中「添加矢量图层」选择导出的 `.gpkg`；聚类图层名形如 `clusters_pca` / `clusters_umap`，预测图层名为 `predictions_xgb`（XGBoost 预测结果）；属性表中包含 `cluster_id`、`cluster_name`（若由融合流程生成）、`prediction_rank`、`risk_level`（低/中/高）等字段。ArcGIS Pro 使用「添加数据」同样可载入同一 GPKG。
2. **按 cluster_id 着色**：图层属性 → 符号化 → 分类 / 唯一值 → 字段选 `cluster_id`；可同时**叠加井位 shapefile/geojson** 作为最上层点要素，检查井–簇归属关系。
3. **按预测值分级**：对 `prediction_xgboost` 或流水线中的 `xgb_pred` 使用「分级色彩」或「等间距 / 分位数」断点，突出高值异常区；`risk_level` 可作为三档定性图例（勘探关注区示意）。
4. **井位叠加**：将所有图层统一到**同一 CRS**（与本程序分析所用的投影一致），避免几何错位。

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

通过 `config.yaml`（`program/` 下）统一管理：

- 网格导出参数：`export_grid.cell_width`
- 聚类参数：`clustering.n_clusters`、降维维数、随机种子
- 训练参数：`train.target_column`、`random_state`、折数、测试集比例、`stability_seeds`（多随机种子下的测试集 R² 波动）、是否导出预测结果
- SHAP/日志/环境说明：`explain`、`logging`、`environment`

主界面「聚类 k / 训练目标列 / 网格步长」会**覆盖本次运行**所用的参数（摘要区会同时显示 config 默认值与 GUI 当前值）。完整复现仍建议以 `config.yaml` 为准。

---

## 五、设计重点与难点

### 5.1 重点

| 重点 | 说明 |
|------|------|
| **地质领域适配** | 将拓扑指标与油气勘探中的「连通率」「断裂强度」等高价值属性关联，设计专家规则加权策略 |
| **配置驱动复现** | 通过 `config.yaml` 固定网格步长、聚类数、训练目标列、随机种子与导出行为，减少手工改代码造成的偏差 |
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
| **结果落地到 GIS** | 聚类 `cluster_id`、预测值与区间可一键导出为 CSV / GPKG，便于与井位、构造图在 GIS 中叠加 |

---

## 六、目录结构概览

```
断裂拓扑分析/
├── README.md                 # 本文件
├── environment.yml           # 推荐复现实验环境（conda）
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
    ├── MY/                   # 英买 2 区域数据目录（迹线、研究区 GeoJSON）
    └── new plot/             # 栅格图分析（可选）
```

---
## 七、快速开始

```bash
# 进入项目根目录后安装依赖（建议先创建并激活虚拟环境，见第三节）
pip install -r requirements.txt
# 启动主界面（推荐：根目录直接启动，跨平台）
python run.py
```

也可使用平台脚本：

```bash
bash run.sh
```

Windows:

```bat
run.bat
```

使用说明：在界面中选择数据源，点击「原断裂数据地图」等进行拓扑分析；若要跑「一键空间-拓扑融合」等完整 ML 流程，需先对对应区域执行 `export_grid_csv` 生成网格 CSV。聚类结果会导出带 `cluster_id` 的 CSV/GPKG，训练后会导出预测值、预测区间及 SHAP 表。

Windows 若 python 无法运行，可改用：
```bash
cd program
py main.py
```
或在项目根目录启动（无需先cd program）
```bash
py program/main.py
```

---
## 八、GUI 向导与异步任务

- 左侧新增「流程向导（网格 → 训练 → 解释）」状态区，实时显示三步是否完成。
- 「一键空间-拓扑融合」改为后台线程执行，运行时可点击顶部「取消任务」中止，避免界面卡死。
- 运行后会更新「最近一次运行结果」与 `program/data/processed/gui_run_history.jsonl`。

---
## 九、可信度与稳定性输出

- 训练结果同时给出：
  - 常规随机 CV 指标；
  - 空间 Block-CV 指标（避免空间邻近泄漏）；
  - split-conformal 预测区间与测试覆盖率。
- 预测导出（CSV/GPKG）新增字段：
  - `uncertainty_width`、`uncertainty_level`、`uncertainty_score`；
  - `processing_run_id`、`run_timestamp_utc`、`config_hash_sha256`。
- 每次运行会额外导出 `*_run_manifest.json` 记录工件路径和元信息。

---
## 十、QGIS 样式与井数据联接

### 10.1 样式模板

仓库提供 `qgis_styles/`：

- `predictions_xgb.qml`：预测值分级色带；
- `uncertainty_score.qml`：不确定性分级色带；
- `clusters_unique.qml`：按 `cluster_id` 分类着色。

在 QGIS 中：图层属性 → 样式 → 加载样式（`.qml`）即可一键套用。

### 10.2 井数据联接脚本

```bash
python program/tools/join_well_data.py \
  --grid "program/model/xgboost_predictions.gpkg" --layer predictions_xgb \
  --wells "/path/to/wells.geojson" \
  --out "program/data/processed/wells_joined.gpkg"
```

脚本会把网格预测字段（如 `prediction_xgboost`、`cluster_id`、`risk_level`、`uncertainty_score`）空间连接到井点。

---
## 十一、工程治理与发布

- 启动时会校验 `program/config.yaml` 的关键字段范围与互斥关系，提前提示配置错误。
- 新增 CSV 契约测试（必备网格顶点列 + 至少一个特征列），降低升级依赖后的隐性破坏。
- 版本与变更：
  - `VERSION`
  - `CHANGELOG.md`
- 支持 `pip install -e .` 的基础安装结构（见 `pyproject.toml`）。


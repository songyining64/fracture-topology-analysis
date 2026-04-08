# 油气区断裂网络连通性智能分析与预测系统

**油气区断裂网络连通性智能分析与预测系统**面向**油气勘探工区**，将 **fractopo 断裂网络拓扑**与 **连通性相关指标**（如分支/迹线连接度、连接频率等）和 **机器学习预测**放在同一套桌面工作流中：从原始迹线 + 研究区到网格化拓扑属性、多维融合与聚类，再到 XGBoost 等指标预测与 SHAP 解释，实现「看得清（可视化）—算得准（建模）—讲得明（解释）」。默认示例数据为塔里木盆地英买 2（MY）。

**分发形态：** 本仓库支持 **macOS（`.app`）** 与 **Windows（文件夹内 `.exe` + `_internal`）** 的 PyInstaller 打包；**界面与业务逻辑在两端一致**，仅系统级差异（路径、安全提示等）可能不同。

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

## 三、安装、配置与使用（桌面安装包用户）

本节面向 **已拿到打包产物** 的使用者（**不**需要在本机安装 Python）。若你从源码运行，请直接看 **第四节**。

### 3.1 macOS（`.app`）

1. **安装**：将 **`油气区断裂网络连通性智能分析与预测系统_Mac.app`**（或发行方提供的同名 `.app`）拖入 **「应用程序」** 文件夹，从 **启动台** 或 **访达 → 应用程序** 中启动。  
   - 若应用来自网络下载，首次打开请在图标上 **右键 → 打开**，或在 **系统设置 → 隐私与安全性** 中允许运行。  
   - **不要长期从「下载」文件夹直接双击运行**：系统可能启用 App Translocation，导致应用包处于只读挂载，写入日志或缓存失败。
2. **配置**：默认读取打包在应用内的 `program/config.yaml` 及示例数据。若发行方说明支持用户配置目录，请以随包说明为准；一般可在界面中调整 **聚类 k、训练目标列、网格步长** 等（部分项会覆盖配置文件默认值）。
3. **使用**：启动后在上方的 **数据源** 中选择工区；**地图与拓扑图** 依赖 `program/MY/` 等随包数据；**融合 / 机器学习 / SHAP / 一键流水线** 依赖对应区域的 **网格 CSV**（若包内已带 `program/Yingmai 2 area in Tarim Basin.csv` 可直接使用，否则需按说明导出或替换数据）。  
4. **详细说明**：界面按钮与推荐操作顺序见根目录 **`功能说明.md`**；命令行脚本与依赖说明见 **`运行说明.md`**。

### 3.2 Windows（文件夹分发）

1. **安装**：解压发行方提供的 **整个文件夹**（内含 **`.exe`** 与 **`_internal`** 等子目录）。**勿只拷贝单个 `.exe`**，必须与 **`_internal`** 保持相对位置不变，否则无法启动。
2. **启动**：双击主程序 **`.exe`**。若出现 **Windows 安全中心 / SmartScreen** 提示，选择「仍要运行」或按发行方说明添加信任。
3. **配置与使用**：与 macOS 相同逻辑——数据源、网格 CSV、界面参数与 **`program/config.yaml`** 的关系一致；详细操作见 **`功能说明.md`**、**`运行说明.md`**。
4. **构建说明**：若在 **本机从源码重新打包** Windows 版，见 **第五节** 与 **`Windows打包说明.md`**。

### 3.3 平台对照

| 项目 | macOS | Windows |
|------|--------|---------|
| 典型产物 | `.app` 或 `dist/` 下 onedir 文件夹 | `dist\油气区断裂网络连通性智能分析与预测系统\` 内含 `.exe` + `_internal` |
| 功能与界面 | 与源码运行一致 | 与 macOS、源码运行一致 |
| 注意 | 放入「应用程序」再运行更稳妥 | 整夹分发，勿删 `_internal` |

---

## 四、从源码安装与运行（开发者）

适用于 **克隆仓库、二次开发或论文复现**，需自行安装 Python 环境。

### 4.1 环境要求

- Python **3.10 / 3.11**（与 `environment.yml`、打包脚本推荐版本一致；**3.11** 为打包脚本首选）
- Windows / macOS / Linux 均可运行源码

### 4.2 安装依赖

```bash
cd 断裂拓扑分析

# 方式 A：venv + pip（常用）
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

若希望按**经过测试的版本组合**复现，推荐使用 **`environment.yml`**：

```bash
conda env create -f environment.yml
conda activate fracture-connectivity
```

`requirements.txt` 已包含 `torch`、`umap-learn`、`torch-geometric` 等，与 GUI 中自编码器、VAE、UMAP、GAT 等高级功能一致；若安装失败，可按各库官方说明补装。

### 4.3 启动主界面（推荐）

在**项目根目录**执行（路径相对根目录，跨平台一致）：

```bash
python run.py
```

也可使用：

- macOS / Linux：`bash run.sh`
- Windows：双击 **`run.bat`**

等价方式：`cd program` 后 `python main.py`，或在根目录 `py program/main.py`（Windows 若 `python` 未进 PATH，可用 `py`）。

### 4.4 数据准备

- 迹线与研究区示例在 **`program/MY/`**；扩展多工区可在代码中的 `DATA_SOURCES` 配置。
- **融合 / 机器学习** 需先有网格 CSV，可在 `program/` 下执行：
  ```bash
  python export_grid_csv.py MY
  ```
  生成（或更新）与英买 2 对应的网格 CSV（默认输出名见脚本与 `config.yaml`）。

### 4.5 使用提示（源码与安装包相同）

在界面中选择数据源，使用「原断裂数据地图」等做拓扑可视化；「一键空间-拓扑融合」等完整 ML 流程需 **网格 CSV 就绪**。聚类结果可导出带 `cluster_id` 的 CSV/GPKG；训练后可导出预测值、预测区间及 SHAP 表。更细的按钮说明见 **`功能说明.md`**。

---

## 五、构建桌面安装包（macOS / Windows）

用于 **在本机从源码生成可分发的 `.app` 或 Windows 程序文件夹**。  
**重要：PyInstaller 必须在目标操作系统上执行**——在 macOS 上只能打 macOS 包，在 Windows 上只能打 Windows 包，**不能交叉编译**。

### 5.1 通用前提

- 已安装 **Python 3.11**（推荐，与 `build_mac.sh` / `build_windows.bat` 一致）
- 磁盘与网络：首次安装依赖体积较大（含 PyTorch 等），请预留足够空间
- 打包前建议已能 **`pip install -r requirements.txt`** 并成功 **`python run.py`** 跑通 GUI

### 5.2 macOS

在仓库根目录执行：

```bash
bash build_mac.sh
```

脚本会创建 **`.venv_mac_build`**、安装依赖与 **PyInstaller**，并执行 **`build_app.spec`**。完成后常见输出在 **`dist/`** 下，例如 **`油气区断裂网络连通性智能分析与预测系统_Mac.app`** 及同名 onedir 文件夹（具体以 `build_app.spec` 与脚本 echo 为准）。  
**分发**：将 **`.app`** 或 **整个 onedir 文件夹** 压缩后分发给其他 Mac 用户。

### 5.3 Windows

在 **Windows** 机器上进入仓库根目录，双击或命令行运行：

```bat
build_windows.bat
```

将创建 **`.venv_win_build`** 并完成依赖安装与打包；产物通常在 **`dist\油气区断裂网络连通性智能分析与预测系统\`** 下（**`.exe` + `_internal`**）。  
**详细步骤、路径与排错**见 **`Windows打包说明.md`**。

### 5.4 与手动 PyInstaller 的关系

仓库以 **`build_app.spec`** 集中维护打包选项（数据文件、隐藏导入等）。若仅做试验，也可在已激活环境中执行 `pyinstaller build_app.spec --noconfirm`，但仍建议优先使用 **`build_mac.sh` / `build_windows.bat`** 以保证环境一致。

---

## 六、本系统如何定义和量化断裂网络连通性

在本项目中，「连通性」主要指**断裂网络在几何–拓扑上的连接难易程度**，由 fractopo 在网格尺度上输出的、可直接进入机器学习的一套标量指标来表示：

- **Connections per Branch**（每个分支上的连接点数）：分支越常被其他结构“挂靠”，该值越高，反映局部网络越密集、越易于形成渗流路径。
- **Connections per Trace**（每条迹线上的连接点数）：刻画迹线作为断裂段在端点/交点处被网络连接的程度。
- **Connection Frequency**（连接频率类指标）：在网格内对连接事件或连接密度的统计频率，用于粗粒度对比不同区块的连通活跃程度。

这三项在代码中集中定义为 **「连通性特征组」**（`CONNECTIVITY_FEATURE_COLUMNS` / `config.yaml` 中 `high_value_attrs`），在加权融合时默认赋予更高权重；训练完成后，界面与日志中会单独给出 **连通性特征的 SHAP 排名**及其在 mean|SHAP| 归一意义下的 **累计贡献占比**，便于说明「哪些连通指标在驱动当前目标列」。

> 注意：特征工程阶段可能因方差过小或互信息筛选会**暂时去掉**某一连通列；若 SHAP 表中没有某项，通常表示该列未进入最终模型特征子集。

---

## 七、在 QGIS / ArcGIS 中快速使用导出图层

1. **加载 GPKG**：在 QGIS 中「添加矢量图层」选择导出的 `.gpkg`；聚类图层名形如 `clusters_pca` / `clusters_umap`，预测图层名为 `predictions_xgb`（XGBoost 预测结果）；属性表中包含 `cluster_id`、`cluster_name`（若由融合流程生成）、`prediction_rank`、`risk_level`（低/中/高）等字段。ArcGIS Pro 使用「添加数据」同样可载入同一 GPKG。
2. **按 cluster_id 着色**：图层属性 → 符号化 → 分类 / 唯一值 → 字段选 `cluster_id`；可同时**叠加井位 shapefile/geojson** 作为最上层点要素，检查井–簇归属关系。
3. **按预测值分级**：对 `prediction_xgboost` 或流水线中的 `xgb_pred` 使用「分级色彩」或「等间距 / 分位数」断点，突出高值异常区；`risk_level` 可作为三档定性图例（勘探关注区示意）。
4. **井位叠加**：将所有图层统一到**同一 CRS**（与本程序分析所用的投影一致），避免几何错位。

---

## 八、设计思路

### 8.1 整体架构

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

### 8.2 核心流程设计

1. **拓扑特征提取**：基于 fractopo 对断裂迹线进行分支/节点拓扑分类，按网格采样得到每个单元的拓扑指标（断裂强度、连通率、迹线长度等）
2. **特征工程**：归一化、异常值处理、方差/互信息筛选，构建标准特征矩阵
3. **多维融合**：将高维拓扑特征通过 PCA/深度学习方法降至低维，并结合专家规则加权、GAT 图模型进行空间-拓扑融合
4. **预测与解释**：XGBoost 回归预测目标属性，SHAP 量化各特征贡献，形成可解释的勘探价值评估

### 8.3 配置驱动

通过 **`program/config.yaml`** 统一管理：

- 网格导出参数：`export_grid.cell_width`
- 聚类参数：`clustering.n_clusters`、降维维数、随机种子
- 训练参数：`train.target_column`、`random_state`、折数、测试集比例、`stability_seeds`（多随机种子下的测试集 R² 波动）、是否导出预测结果
- SHAP/日志/环境说明：`explain`、`logging`、`environment`

主界面「聚类 k / 训练目标列 / 网格步长」会**覆盖本次运行**所用的参数（摘要区会同时显示 config 默认值与 GUI 当前值）。完整复现仍建议以 **`program/config.yaml`** 为准。

---

## 九、设计重点与难点

### 9.1 重点

| 重点 | 说明 |
|------|------|
| **地质领域适配** | 将拓扑指标与油气勘探中的「连通率」「断裂强度」等高价值属性关联，设计专家规则加权策略 |
| **配置驱动复现** | 通过 `config.yaml` 固定网格步长、聚类数、训练目标列、随机种子与导出行为，减少手工改代码造成的偏差 |
| **融合链路完整性** | 将传统拓扑分析、多种降维方法、图神经网络、树模型、可解释性串联成可复现的端到端流水线 |
| **跨平台与鲁棒性** | 中文字体跨平台配置、目标列恒定的校验、样本不足时的友好提示、多尺度金字塔在非规则网格下的降级处理 |

### 9.2 难点

| 难点 | 解决方式 |
|------|----------|
| **网格与图结构的对应** | 将规则网格抽象为邻接图，通过 `build_grid_graph` 构建边索引，使 GAT/GraphSAGE 能利用空间邻接关系 |
| **融合方法的选择与组合** | 提供多种融合方式（规则加权、自适应加权、GAT、PCA/VAE 等），用户可根据数据规模与需求选择，并支持加权 vs GAT 的对比实验 |
| **目标列的有效性** | 识别恒定目标（如网格面积）导致的回归失效，默认推荐有变化的拓扑指标作为目标，并做方差校验 |
| **样本过滤后的兼容** | 特征工程会过滤全零行，导致样本数变化；多尺度金字塔需样本数满足规则网格，否则自动跳过多尺度 |
| **可选依赖的降级** | GAT、VAE、UMAP 依赖 PyTorch 等，未安装时相关功能优雅降级，不影响主流程运行 |
| **结果落地到 GIS** | 聚类 `cluster_id`、预测值与区间可一键导出为 CSV / GPKG，便于与井位、构造图在 GIS 中叠加 |

---

## 十、目录结构概览

```
断裂拓扑分析/
├── README.md                      # 本文件
├── CHANGELOG.md                   # 版本变更记录
├── VERSION                        # 版本号
├── environment.yml                # conda 推荐环境
├── requirements.txt               # pip 依赖
├── pyproject.toml                 # 包元数据与可编辑安装
├── run.py / run.sh / run.bat      # 从源码启动 GUI（根目录）
├── build_app.spec                 # PyInstaller 规格文件
├── build_mac.sh                   # macOS 一键打包
├── build_windows.bat              # Windows 一键打包（需在 Windows 上运行）
├── Windows打包说明.md             # Windows 打包详细说明
├── 运行说明.md                    # 运行与脚本命令速查
├── 功能说明.md                    # 界面按钮与功能详解
├── 项目说明-模块与使用.md         # 模块说明（若有）
├── qgis_styles/                   # QGIS 图层样式模板（.qml）
├── pyi_rth_multiprocessing.py     # PyInstaller 运行时 hook（多进程）
├── tests/                         # 单元测试
└── program/
    ├── main.py                    # GUI 主入口
    ├── demo.py                    # 界面布局
    ├── config.yaml                # 融合/训练/导出等配置
    ├── export_grid_csv.py         # 网格 CSV 导出
    ├── feature_engineering.py
    ├── fusion_algorithm.py
    ├── topology_fusion.py
    ├── spatial_topology_framework.py
    ├── multiscale_features.py
    ├── gnn_embeddings.py
    ├── evaluation.py
    ├── ml/                        # 训练、调参、推理、SHAP
    ├── utils/                     # 含 config_validation、export_utils、crs_metric 等
    ├── tools/join_well_data.py    # 井数据空间联接
    ├── MY/                        # 英买 2 示例 GeoJSON
    └── data/processed/            # 运行产出（如 gui_run_history.jsonl，视使用情况生成）
```

---

## 十一、GUI 向导与异步任务

- 左侧「流程向导（网格 → 训练 → 解释）」状态区，实时显示三步是否完成。
- 「一键空间-拓扑融合」在后台线程执行，可点击「取消任务」中止，避免界面卡死。
- 运行后会更新「最近一次运行结果」与 `program/data/processed/gui_run_history.jsonl`。

---

## 十二、可信度与稳定性输出

- 训练结果同时给出：
  - 常规随机 CV 指标；
  - 空间 Block-CV 指标（避免空间邻近泄漏）；
  - split-conformal 预测区间与测试覆盖率。
- 预测导出（CSV/GPKG）新增字段：
  - `uncertainty_width`、`uncertainty_level`、`uncertainty_score`；
  - `processing_run_id`、`run_timestamp_utc`、`config_hash_sha256`。
- 每次运行会额外导出 `*_run_manifest.json` 记录工件路径和元信息。

---

## 十三、QGIS 样式与井数据联接

### 13.1 样式模板

仓库提供 `qgis_styles/`：

- `predictions_xgb.qml`：预测值分级色带；
- `uncertainty_score.qml`：不确定性分级色带；
- `clusters_unique.qml`：按 `cluster_id` 分类着色。

在 QGIS 中：图层属性 → 样式 → 加载样式（`.qml`）即可一键套用。

### 13.2 井数据联接脚本

```bash
python program/tools/join_well_data.py \
  --grid "program/model/xgboost_predictions.gpkg" --layer predictions_xgb \
  --wells "/path/to/wells.geojson" \
  --out "program/data/processed/wells_joined.gpkg"
```

脚本会把网格预测字段（如 `prediction_xgboost`、`cluster_id`、`risk_level`、`uncertainty_score`）空间连接到井点。

---

## 十四、工程治理与发布

- 启动时会校验 **`program/config.yaml`** 的关键字段范围与互斥关系，提前提示配置错误。
- 提供 CSV 契约测试（必备网格顶点列 + 至少一个特征列），降低升级依赖后的隐性破坏。
- 版本与变更见 **`VERSION`**、**`CHANGELOG.md`**。
- 支持 `pip install -e .`（见 **`pyproject.toml`**）。

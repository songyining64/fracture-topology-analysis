# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — 油气断裂网络连通性智能分析与预测系统
macOS / Windows 通用；在对应平台执行 pyinstaller build_app.spec 即可。
"""
import sys
import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

block_cipher = None

# ── 路径 ──────────────────────────────────────────────
SPEC_DIR = SPECPATH
PROGRAM_DIR = os.path.join(SPEC_DIR, "program")

# ── 数据文件 ──────────────────────────────────────────
# PyInstaller onedir 模式下，main.py 的 __file__ 在 _internal/ 根目录，
# 所以 config.yaml 等数据文件也放在 "."（即 _internal/ 根）。
data_files = [
    (os.path.join(PROGRAM_DIR, "config.yaml"), "."),
    (os.path.join(PROGRAM_DIR, "demo.ui"), "."),
]

# 示例数据（CSV / GeoJSON）
for csv in [
    "Yingmai 2 area in Tarim Basin.csv",
    "MY.csv",
    "THK.csv",
    "KB11.csv",
]:
    src = os.path.join(PROGRAM_DIR, csv)
    if os.path.isfile(src):
        data_files.append((src, "."))

for subdir in ["MY", "KB11", "THK"]:
    src = os.path.join(PROGRAM_DIR, subdir)
    if os.path.isdir(src):
        data_files.append((src, subdir))

# 预训练模型
model_dir = os.path.join(PROGRAM_DIR, "model")
if os.path.isdir(model_dir):
    data_files.append((model_dir, "model"))

# QGIS 样式
qgis_dir = os.path.join(SPEC_DIR, "qgis_styles")
if os.path.isdir(qgis_dir):
    data_files.append((qgis_dir, "qgis_styles"))

# ── 收集第三方包数据文件 ──────────────────────────────
datas_extra = []
for pkg in [
    "fractopo",
    "pyproj",
    "pyogrio",
    "geopandas",
    "shapely",
    "xgboost",
    "sklearn",
    "certifi",
]:
    try:
        datas_extra += collect_data_files(pkg)
    except Exception:
        pass

# pyogrio / pyproj / shapely 二进制（.so / .dylib / .dll）
binaries_extra = []
for pkg in ["pyogrio", "pyproj", "shapely"]:
    try:
        binaries_extra += collect_dynamic_libs(pkg)
    except Exception:
        pass

# ── Hidden Imports ────────────────────────────────────
hidden_imports = []

# scikit-learn 子模块（函数内导入，PyInstaller 经常遗漏）
hidden_imports += collect_submodules("sklearn")

# fractopo
hidden_imports += collect_submodules("fractopo")

# pyogrio（C 扩展 _geometry / _io / _ogr / _err / _vsi 等）
hidden_imports += collect_submodules("pyogrio")

# pyproj
hidden_imports += collect_submodules("pyproj")

# PyQt5
hidden_imports += [
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "PyQt5.sip",
]

# matplotlib Qt5 后端
hidden_imports += [
    "matplotlib",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_agg",
]

# 函数内延迟导入的包
hidden_imports += [
    "yaml",
    "scipy",
    "scipy.stats",
    "scipy.spatial",
    "scipy.ndimage",
    "numpy",
    "pandas",
    "geopandas",
    "shapely",
    "shapely.geometry",
    "pyproj",
    "networkx",
    "cv2",
    "PIL",
    "PIL.Image",
    "joblib",
    "cloudpickle",
]

# 可选重型包（torch / xgboost / shap / umap / optuna）
for pkg in [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch_geometric",
    "torch_geometric.data",
    "torch_geometric.nn",
    "xgboost",
    "shap",
    "umap",
    "optuna",
    "numba",
    "llvmlite",
]:
    hidden_imports.append(pkg)

# 本地模块（program/ 下）
hidden_imports += [
    "demo",
    "feature_engineering",
    "fusion_algorithm",
    "gnn_embeddings",
    "multiscale_features",
    "evaluation",
    "topology_fusion",
    "spatial_topology_framework",
    "export_grid_csv",
    "utils",
    "utils.config_loader",
    "utils.config_validation",
    "utils.crs_metric",
    "utils.matplotlib_chinese",
    "utils.logging_utils",
    "utils.export_utils",
    "utils.validation",
    "ml",
    "ml.train",
    "ml.explain",
    "ml.infer",
    "ml.tune",
]

# ── Analysis ──────────────────────────────────────────
a = Analysis(
    [os.path.join(PROGRAM_DIR, "main.py")],
    pathex=[PROGRAM_DIR],
    binaries=binaries_extra,
    datas=data_files + datas_extra,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPEC_DIR, "pyi_rth_multiprocessing.py")],
    excludes=[
        "tkinter",
        "IPython",
        "notebook",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── 使用 onedir 模式（体积更小、启动更快） ────────────
APP_NAME = "油气区断裂网络连通性智能分析与预测系统"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

# ── macOS .app 包（仅 macOS 生效） ────────────────────
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier="com.fracture.connectivity",
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleVersion": "1.2.0",
            "CFBundleShortVersionString": "1.2.0",
            "NSHighResolutionCapable": True,
        },
    )

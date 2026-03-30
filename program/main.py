# -*- coding: utf-8 -*-
import sys
import os
import math
import warnings

# 保证 program 目录在 path 中，便于从项目根或 program/ 运行时的导入
_PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

# 清除 fractopo 损坏的 joblib 缓存（EOFError），避免每次运行报错
for _cache in [
    os.path.join(_PROGRAM_DIR, ".cache", "fractopo"),
    os.path.join(os.path.dirname(_PROGRAM_DIR), ".cache", "fractopo"),
]:
    if os.path.isdir(_cache):
        try:
            import shutil
            shutil.rmtree(_cache)
        except Exception:
            pass

# Mac 上 PyQt5 找不到 cocoa 插件时（必须在 import PyQt5 之前设置）
if sys.platform == "darwin":
    for p in sys.path:
        qt_plugin_path = os.path.join(p, "PyQt5", "Qt5", "plugins", "platforms")
        if os.path.exists(qt_plugin_path):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path
            break

# 确保 matplotlib 缓存目录可写（必须在 import matplotlib 之前）。
# macOS 上 ~/.matplotlib 因 com.apple.provenance 扩展属性可能不可写，
# 导致每次启动重建临时缓存、字体扫描极慢、中文显示为方框。
# 优先用项目内 .cache/mplconfig（项目目录本身始终可写），
# 这样字体缓存持久保存在项目里，第二次启动直接命中缓存，中文字体即可正常显示。
if "MPLCONFIGDIR" not in os.environ:
    _mpl_project_cache = os.path.join(_PROGRAM_DIR, ".cache", "mplconfig")
    os.makedirs(_mpl_project_cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_project_cache

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm as mpl_cm

from utils.matplotlib_chinese import setup_matplotlib_chinese
from utils.crs_metric import unify_traces_area_crs, reproject_to_metric_crs

setup_matplotlib_chinese()

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QComboBox, QFrame, QHBoxLayout, QVBoxLayout, QLabel, \
    QPushButton, QInputDialog

from fractopo.branches_and_nodes import branches_and_nodes
from pprint import pprint
from matplotlib.lines import Line2D
from demo import Ui_MainWindow
from fractopo.general import (
    CC_branch,
    CI_branch,
    II_branch,
    X_node,
    Y_node,
    I_node,
    CONNECTION_COLUMN,
    CLASS_COLUMN,
)
from fractopo import Network
import geopandas as gpd
from scipy.stats import gaussian_kde
import numpy as np
from PyQt5.QtGui import QTextCursor

try:
    from topology_fusion import (
        run_fusion_pipeline,
        run_fusion_pipeline_ae,
        run_fusion_pipeline_umap,
        run_fusion_pipeline_vae,
    )
except ImportError:
    run_fusion_pipeline = None
    run_fusion_pipeline_ae = None
    run_fusion_pipeline_umap = None
    run_fusion_pipeline_vae = None

try:
    import torch as _torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import umap as _umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings("ignore", message=".*geographic CRS.*", category=UserWarning)

# 数据源配置：迹线、研究区、显示名、对应的网格 CSV（融合/ML 用）
# 当前仅保留塔里木盆地英买 2（MY）
DATA_SOURCES = [
    {
        "traces": "MY/11.geojson",
        "area": "MY/my_area1.geojson",
        "name": "塔里木盆地英买2",
        "csv": "Yingmai 2 area in Tarim Basin.csv",
    },
]

# 全局变量，由 load_data_source 更新
traces = None
area = None
name = None
rate = 1.0
width = 0.0
height = 0.0
left = 0.0
right = 0.0
down = 0.0
up = 0.0


def load_data_source(index: int):
    """按索引加载数据源，更新全局 traces/area/name/rate/width/height/left/right/down/up。"""
    global traces, area, name, rate, width, height, left, right, down, up
    if index < 0 or index >= len(DATA_SOURCES):
        return False
    cfg = DATA_SOURCES[index]
    base = _PROGRAM_DIR
    trace_path = os.path.join(base, cfg["traces"])
    area_path = os.path.join(base, cfg["area"])
    if not os.path.isfile(trace_path) or not os.path.isfile(area_path):
        return False
    traces = gpd.read_file(trace_path)
    area = gpd.read_file(area_path)
    traces, area = unify_traces_area_crs(traces, area)
    traces, area = reproject_to_metric_crs(traces, area)
    name = cfg["name"]
    traces.drop_duplicates(subset="geometry", inplace=True)
    traces.reset_index(drop=True, inplace=True)
    geometry = traces.geometry.tolist()
    left = math.inf
    right = -math.inf
    down = math.inf
    up = -math.inf
    for one in geometry:
        b = one.boundary.bounds
        left = min(left, b[0])
        right = max(right, b[2])
        down = min(down, b[1])
        up = max(up, b[3])
    rate = (up - down) / (right - left) if (right - left) > 0 else 1.0
    width = 0.01 * (right - left)
    height = 0.01 * (up - down)
    # 上述 left/right/down/up 已通过 global 写入模块全局
    return True


def load_first_available_data_source() -> int:
    """加载唯一数据源（英买 MY）；失败返回 -1。"""
    if load_data_source(0):
        return 0
    return -1


EMPTY_CROP_MSG = (
    "裁剪后迹线为空：迹线与当前研究区多边形无空间重叠，或数据/坐标系不匹配。\n\n"
    "请检查：① 迹线 GeoJSON 与研究区是否属同一工区；② 在 QGIS 中二者是否相交；③ 导出时统一坐标系。"
)


def try_network(*args, **kwargs):
    """
    构造 fractopo.Network，捕获裁剪后迹线为空等错误，避免未处理异常导致进程退出。
    返回 (network, None) 或 (None, 错误说明)。
    """
    try:
        return Network(*args, **kwargs), None
    except ValueError as e:
        if "Empty trace GeoDataFrame after crop" in str(e):
            return None, EMPTY_CROP_MSG
        return None, str(e)


def _style_ternary_plot(fig, tax):
    """去掉 python-ternary 默认灰色三角底色，并加深图中的虚线（理论曲线等）。"""
    try:
        tax.set_background_color(color="white", alpha=1.0, zorder=-1000)
    except Exception:
        pass

    def _line_is_dashed(line):
        ls = line.get_linestyle()
        if ls == "--":
            return True
        if isinstance(ls, tuple) and len(ls) >= 2:
            return True
        return False

    for ax in fig.axes:
        for line in ax.get_lines():
            if not _line_is_dashed(line):
                continue
            line.set_color("#1a1a1a")
            line.set_alpha(min(1.0, max(line.get_alpha() or 0.6, 0.6) + 0.32))
            lw = line.get_linewidth()
            line.set_linewidth(max(lw * 1.65, 1.2))


def _polish_fractopo_ternary_labels(fig):
    """
    fractopo 在 X/Y/I 顶点使用白字+粗描边；若再统一套 bbox 会像乱码方框且易被裁切。
    统计信息（多行）单独用圆角白底框；$C_B$ 阈值线用 DejaVu 避免数学符号缺字。
    """
    corners = frozenset({"X", "Y", "I", "I-C", "C-C", "I-I"})
    for ax in fig.axes:
        for txt in ax.texts:
            raw = txt.get_text().strip()
            compact = raw.replace(" ", "").replace("–", "-").replace("—", "-")
            if compact in corners:
                txt.set_bbox(None)
                try:
                    txt.set_path_effects([])
                except Exception:
                    pass
                txt.set_color("#111111")
                txt.set_fontfamily("DejaVu Sans")
                txt.set_fontweight("bold")
                txt.set_fontsize(19)
            elif raw.startswith("$"):
                txt.set_fontfamily("DejaVu Sans")
                txt.set_color("#1a1a1a")
                try:
                    txt.set_path_effects([])
                except Exception:
                    pass
            elif "\n" in raw:
                txt.set_bbox(
                    dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        edgecolor="#9CA3AF",
                        alpha=0.95,
                    )
                )
                zh_fonts = plt.rcParams.get("font.sans-serif", [])
                if isinstance(zh_fonts, (list, tuple)) and zh_fonts:
                    txt.set_fontfamily(zh_fonts[0])


# 启动时加载第一个可用的数据源（见 load_first_available_data_source）
START_DATA_SOURCE_INDEX = load_first_available_data_source()


def assign_colors(feature_type: str):
    if feature_type in (CC_branch, X_node):
        return "green"
    if feature_type in (CI_branch, Y_node):
        return "blue"
    if feature_type in (II_branch, I_node):
        return "black"
    return "red"


# 控制台输出重定向（错误信息放行，便于排查；仅过滤已知无害的系统噪音）
class StreamRedirector(QtCore.QObject):
    text_written = QtCore.pyqtSignal(str)

    def __init__(self, is_error=False):
        super().__init__()
        self.is_error = is_error

    def write(self, text):
        text = str(text)
        if not text.strip():
            return

        if self.is_error:
            # 只过滤已知无害的系统噪音，保留真实错误供排查
            skip_patterns = (
                "building the font cache",
                "PasteBoard:",
                "Connection Invalid",
                "Failure on line",
                "no screens available",
                "id scheduleApplicationNotification",
                "MemorizedFunc",
                "Exception while loading results",
                "joblib/memory.py",
                ".cache/fractopo",
            )
            if any(p in text for p in skip_patterns):
                return
            self.text_written.emit(text)
        else:
            self.text_written.emit(text)

    def flush(self):
        pass


def _make_latent_fusion_cmap_norm(n_k: int):
    """潜空间聚类图：柔和离散色 + BoundaryNorm；配色与分区底图、散点一致。"""
    base = np.array(
        [
            [0.40, 0.62, 0.86],
            [0.94, 0.58, 0.48],
            [0.52, 0.74, 0.54],
            [0.72, 0.56, 0.82],
            [0.96, 0.78, 0.40],
            [0.42, 0.74, 0.80],
            [0.86, 0.48, 0.56],
            [0.58, 0.62, 0.72],
        ],
        dtype=np.float64,
    )
    n_k = max(int(n_k), 1)
    reps = int(np.ceil(n_k / len(base)))
    colors = np.vstack([base] * reps)[:n_k]
    cmap = ListedColormap(colors)
    bounds = np.arange(n_k + 1, dtype=np.float64) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# 主窗口逻辑
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # 1. 开启终端输出智能重定向（带降噪滤镜）
        self.stdout_redirector = StreamRedirector(is_error=False)
        self.stdout_redirector.text_written.connect(self.append_text)
        sys.stdout = self.stdout_redirector

        self.stderr_redirector = StreamRedirector(is_error=True)
        self.stderr_redirector.text_written.connect(self.append_text)
        sys.stderr = self.stderr_redirector

        # 1.5 数据源下拉与已加载数据一致（仅英买 MY）
        self.combo_data_source.blockSignals(True)
        if START_DATA_SOURCE_INDEX >= 0:
            self.combo_data_source.setCurrentIndex(START_DATA_SOURCE_INDEX)
        self.combo_data_source.blockSignals(False)
        if START_DATA_SOURCE_INDEX < 0:
            self.append_text(
                "【提示】未找到英买 2 数据。请在 program/MY 下放置 11.geojson、my_area1.geojson，"
                "并准备好网格 CSV（见 README / 运行说明）。\n"
            )

        # 1.6 绑定数据源切换
        self.combo_data_source.currentIndexChanged.connect(self._on_data_source_changed)
        self._refresh_shap_feature_combo()

        # 2. 绑定第一排：基础地质与拓扑绘图
        self.btn_yuantu.clicked.connect(self.run_yuantu)
        self.btn_fenleihou.clicked.connect(self.run_fenleihou)
        self.btn_relitu.clicked.connect(self.run_relitu)
        self.btn_azimuth.clicked.connect(self.run_azimuth)
        self.btn_meiguitu.clicked.connect(self.run_meiguitu)
        self.btn_sanyuantu.clicked.connect(self.run_sanyuantu)
        self.btn_guanxi.clicked.connect(self.run_guanxi)
        self.btn_b.clicked.connect(self.b)
        self.btn_a.clicked.connect(self.a)

        # 3. 绑定第二排：视图与参数提取
        self.combo_topo.currentIndexChanged.connect(self.onIndexChanged_2)
        self.combo_params.currentIndexChanged.connect(self.onIndexChanged)
        self.btn_tuopushuxing.clicked.connect(self.run_tuopushuxing)

        # 4. 绑定第三排：机器学习与属性融合
        self.combo_fusion.currentIndexChanged.connect(self._set_ronghe_combo_tooltip)
        self.btn_ronghe.clicked.connect(self.run_ronghe)
        self.btn_guoji_weighted.clicked.connect(self.run_guoji_weighted_fusion)
        self.btn_guoji_compare.clicked.connect(self.run_guoji_fusion_compare)
        self.btn_guoji_train.clicked.connect(self.run_guoji_train)
        self.btn_guoji_shap.clicked.connect(self.run_guoji_shap)
        self.btn_spatial.clicked.connect(self.run_spatial_topology_framework)

        self.opt = 0
        print("系统初始化完成：")


    def append_text(self, text):
        scrollbar = self.text_browser.verticalScrollBar()
        is_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

        cursor = self.text_browser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)

        if is_at_bottom:
            scrollbar.setValue(scrollbar.maximum())


    def embed_figure(self, figs, *, description=None, descriptions=None):
        """
        在右侧画布区嵌入图像。可选：
        description — 单段说明（多图时每张共用）；
        descriptions — 与 figs 等长的说明列表（翻页时随当前图切换）。
        说明显示在图表下方灰色区域。
        """
        if not isinstance(figs, list):
            figs = [figs]
        n = len(figs)
        if descriptions is not None and len(descriptions) == n:
            self._fig_captions = [str(s).strip() for s in descriptions]
        elif description:
            d = str(description).strip()
            self._fig_captions = [d] * n
        else:
            self._fig_captions = [""] * n

        while self.canvas_layout.count():
            item = self.canvas_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout() is not None:
                layout = item.layout()
                while layout.count():
                    sub_item = layout.takeAt(0)
                    sub_widget = sub_item.widget()
                    if sub_widget is not None:
                        sub_widget.setParent(None)
                        sub_widget.deleteLater()
                layout.deleteLater()

        self.current_figs = figs
        self.current_fig_sizes = [tuple(fig.get_size_inches()) for fig in figs]
        self.current_fig_dpis = [fig.dpi for fig in figs]
        self.current_fig_idx = 0

        if len(figs) > 1:
            self.gallery_control_layout = QtWidgets.QHBoxLayout()
            self.gallery_control_layout.setContentsMargins(10, 5, 10, 5)

            self.btn_prev_fig = QtWidgets.QPushButton("◀ 上一张")
            self.btn_prev_fig.setStyleSheet(
                    "background-color: #34495e; color: white; padding: 5px 15px; border-radius: 4px;")
            self.btn_prev_fig.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

            self.lbl_fig_status = QtWidgets.QLabel(f"1 / {len(figs)}")
            self.lbl_fig_status.setAlignment(QtCore.Qt.AlignCenter)
            self.lbl_fig_status.setStyleSheet("font-weight: bold; font-size: 14px;")

            self.btn_next_fig = QtWidgets.QPushButton("下一张 ▶")
            self.btn_next_fig.setStyleSheet(
                "background-color: #34495e; color: white; padding: 5px 15px; border-radius: 4px;")
            self.btn_next_fig.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

            self.btn_prev_fig.clicked.connect(self.show_prev_figure)
            self.btn_next_fig.clicked.connect(self.show_next_figure)

            self.gallery_control_layout.addStretch()
            self.gallery_control_layout.addWidget(self.btn_prev_fig)
            self.gallery_control_layout.addWidget(self.lbl_fig_status)
            self.gallery_control_layout.addWidget(self.btn_next_fig)
            self.gallery_control_layout.addStretch()

            self.canvas_layout.addLayout(self.gallery_control_layout)

        self.canvas_display_layout = QtWidgets.QVBoxLayout()
        self.canvas_display_layout.setContentsMargins(8, 8, 8, 8)
        self.canvas_layout.addLayout(self.canvas_display_layout)

        self._render_current_figure()

    def show_prev_figure(self):
        if self.current_fig_idx > 0:
            self.current_fig_idx -= 1
            self._render_current_figure()

    def show_next_figure(self):
        if self.current_fig_idx < len(self.current_figs) - 1:
            self.current_fig_idx += 1
            self._render_current_figure()

    def _render_current_figure(self):
        while self.canvas_display_layout.count():
            item = self.canvas_display_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        fig = self.current_figs[self.current_fig_idx]
        if hasattr(self, "current_fig_sizes") and self.current_fig_idx < len(self.current_fig_sizes):
            width, height = self.current_fig_sizes[self.current_fig_idx]
            fig.set_size_inches(width, height, forward=False)
        if hasattr(self, "current_fig_dpis") and self.current_fig_idx < len(self.current_fig_dpis):
            fig.set_dpi(self.current_fig_dpis[self.current_fig_idx])

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas_display_layout.addWidget(canvas)
        canvas.draw()

        cap_text = ""
        if hasattr(self, "_fig_captions") and self.current_fig_idx < len(self._fig_captions):
            cap_text = self._fig_captions[self.current_fig_idx]
        cap_lbl = QtWidgets.QLabel()
        cap_lbl.setObjectName("figureCaptionLabel")
        cap_lbl.setWordWrap(True)
        cap_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        cap_lbl.setOpenExternalLinks(False)
        cap_lbl.setStyleSheet(
            "QLabel#figureCaptionLabel {"
            " background-color: #f1f3f5; color: #212529; padding: 10px 12px;"
            " border-radius: 6px; font-size: 13px; border: 1px solid #dee2e6;"
            "}"
        )
        if cap_text:
            cap_lbl.setText("【图说明】" + cap_text)
            cap_lbl.show()
        else:
            cap_lbl.hide()
        self.canvas_display_layout.addWidget(cap_lbl)

        if len(self.current_figs) > 1:
            self.lbl_fig_status.setText(f"第 {self.current_fig_idx + 1} 张 / 共 {len(self.current_figs)} 张")
            self.btn_prev_fig.setEnabled(self.current_fig_idx > 0)
            self.btn_next_fig.setEnabled(self.current_fig_idx < len(self.current_figs) - 1)

        QtWidgets.QApplication.processEvents()

    def _plot_latent_fusion_kmeans_regions(
        self,
        ax,
        df_out,
        x_col: str,
        y_col: str,
        kmeans,
        n_clusters: int,
    ):
        """
        潜空间平面细网格上 KMeans.predict：Voronoi 式分区。
        网格坐标强制 float64，避免 sklearn 1.6+/Py3.13 下 predict 报 buffer dtype mismatch。
        返回 (是否画了簇中心, ListedColormap|None, BoundaryNorm|None) 供散点与 colorbar 同配色。
        """
        xy = df_out[[x_col, y_col]].to_numpy(dtype=np.float64)
        cc = getattr(kmeans, "cluster_centers_", None)
        n_k = int(getattr(kmeans, "n_clusters", n_clusters))
        if cc is None:
            return False, None, None
        cc = np.asarray(cc, dtype=np.float64)
        if cc.shape != (n_k, 2):
            return False, None, None
        x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
        y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
        dx, dy = x_max - x_min, y_max - y_min
        pad_x = max(dx * 0.14, 0.08)
        pad_y = max(dy * 0.14, 0.08)
        if dx < 1e-12:
            pad_x = max(pad_x, 0.5)
        if dy < 1e-12:
            pad_y = max(pad_y, 0.5)
        gx = np.linspace(x_min - pad_x, x_max + pad_x, 360, dtype=np.float64)
        gy = np.linspace(y_min - pad_y, y_max + pad_y, 360, dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        # 与 KMeans(euclidean) 的 predict 等价：最近簇中心标号。
        # 不用 kmeans.predict：部分环境 sklearn Cython 要求 X 为 C 连续 double，易与 float32 网格冲突报错。
        grid_xy = np.ascontiguousarray(
            np.column_stack([xx.ravel(), yy.ravel()]),
            dtype=np.float64,
        )
        cc_c = np.ascontiguousarray(cc, dtype=np.float64)
        d2 = np.sum((grid_xy[:, np.newaxis, :] - cc_c[np.newaxis, :, :]) ** 2, axis=2)
        Z = np.argmin(d2, axis=1).reshape(xx.shape)
        cmap_soft, _ = _make_latent_fusion_cmap_norm(n_k)
        lev = np.arange(n_k + 1, dtype=np.float64) - 0.5
        ax.contourf(
            xx,
            yy,
            Z,
            levels=lev,
            cmap=cmap_soft,
            alpha=0.58,
            antialiased=True,
            zorder=0,
        )
        if n_k > 1:
            ax.contour(
                xx,
                yy,
                Z,
                levels=np.arange(n_k - 1, dtype=np.float64) + 0.5,
                colors="#ffffff",
                linewidths=0.85,
                alpha=0.92,
                zorder=1,
            )
        ax.scatter(
            cc[:, 0],
            cc[:, 1],
            marker="X",
            s=100,
            c="#2a2a2a",
            zorder=5,
            linewidths=1.0,
            edgecolors="white",
            label="KMeans 簇中心",
        )
        cmap_full, norm_full = _make_latent_fusion_cmap_norm(n_k)
        return True, cmap_full, norm_full

    def _plot_spatial_cluster_grid(self, ax, df_out, n_clusters: int, method_name: str) -> bool:
        """
        将网格 CSV 中的四边形单元按 cluster_id 着色，得到与「断裂密度/聚类」
        类似的空间绿–蓝填色图（投影坐标，单位 m）。
        """

        vtx_cols = [
            "vertex1_x",
            "vertex1_y",
            "vertex2_x",
            "vertex2_y",
            "vertex3_x",
            "vertex3_y",
            "vertex4_x",
            "vertex4_y",
        ]
        if not all(c in df_out.columns for c in vtx_cols):
            return False
        if len(df_out) == 0 or "cluster_id" not in df_out.columns:
            return False
        arr = df_out[vtx_cols].to_numpy(dtype=float)
        verts = arr.reshape(-1, 4, 2)
        z = df_out["cluster_id"].to_numpy(dtype=float)
        cmap = mpl_cm.get_cmap("GnBu", max(int(n_clusters), 1))
        bounds = np.arange(-0.5, float(n_clusters) + 0.5, 1.0)
        norm = BoundaryNorm(bounds, cmap.N)
        pc = PolyCollection(
            verts,
            array=z,
            cmap=cmap,
            norm=norm,
            edgecolors="0.82",
            linewidths=0.12,
        )
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        zh_fonts = plt.rcParams.get("font.sans-serif", [])
        font_family = (
            zh_fonts[0]
            if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0
            else None
        )
        ax.set_title(
            f"空间网格聚类分布（{method_name}）",
            fontsize=12,
            fontfamily=font_family,
        )
        ax.tick_params(axis="both", labelsize=9)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 8))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("聚类簇编号", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#1a1a1a")
            spine.set_linewidth(1.1)
        return True

    def _get_guoji_csv(self):
        """根据当前数据源获取对应的网格 CSV 路径。"""
        idx = self.combo_data_source.currentIndex()
        if 0 <= idx < len(DATA_SOURCES):
            csv_name = DATA_SOURCES[idx]["csv"]
        else:
            csv_name = "Yingmai 2 area in Tarim Basin.csv"
        csv_path = os.path.join(_PROGRAM_DIR, csv_name)
        if not os.path.isfile(csv_path):
            QMessageBox.warning(self, "未找到数据",
                                f"未找到：{csv_name}\n请先运行 export_grid_csv.py 为该区域生成网格 CSV。")
            return None
        return csv_path

    def _on_data_source_changed(self, index: int):
        """数据源切换时重新加载迹线与研究区。"""
        if load_data_source(index):
            print(f"已切换数据源：{name}")
            self._refresh_shap_feature_combo()
        else:
            QMessageBox.warning(self, "加载失败",
                                f"无法加载选中的数据源，请确认 {DATA_SOURCES[index]['traces']} 和 {DATA_SOURCES[index]['area']} 存在。")

    def _refresh_shap_feature_combo(self):
        """用当前数据源网格 CSV 刷新 SHAP 特征下拉列表（与训练时 feature_engineering 列一致）。"""
        idx = self.combo_data_source.currentIndex()
        if 0 <= idx < len(DATA_SOURCES):
            csv_name = DATA_SOURCES[idx]["csv"]
        else:
            csv_name = "Yingmai 2 area in Tarim Basin.csv"
        csv_path = os.path.join(_PROGRAM_DIR, csv_name)
        self.combo_shap_features.blockSignals(True)
        self.combo_shap_features.clear()
        self.combo_shap_features.addItem("全部（默认顺序）")
        if os.path.isfile(csv_path):
            try:
                from feature_engineering import build_feature_matrix

                r = build_feature_matrix(csv_path, out_processed_dir=None)
                for fname in r["feature_names"]:
                    self.combo_shap_features.addItem(fname)
            except Exception:
                pass
        self.combo_shap_features.blockSignals(False)

    def run_guoji_weighted_fusion(self):
        try:
            from fusion_algorithm import run_weighted_fusion_pipeline
        except ImportError:
            QMessageBox.warning(self, "模块未安装",
                                "请确保 fusion_algorithm、feature_engineering 可用，并安装 scikit-learn。")
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        warnings.filterwarnings("ignore")
        try:
            df = run_weighted_fusion_pipeline(csv_path)
            mean_s = df["weighted_fusion_score"].mean()
            std_s = df["weighted_fusion_score"].std()
            txt = "【加权融合结果】\n\n"
            txt += f"样本数：{len(df)}\n"
            txt += f"融合得分 均值：{mean_s:.4f}  标准差：{std_s:.4f}\n"
            n_at_min = int((df["weighted_fusion_score"] == 0).sum())
            txt += (
                f"\n说明：得分已按全工区 min-max 归一到 [0,1]，最低一档均为 0；"
                f"当前并列最低的网格数：{n_at_min}（多为无断裂或拓扑特征最弱的格子）。\n"
            )
            txt += "\n前 5 行得分：\n" + df["weighted_fusion_score"].head().to_string()
            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            QMessageBox.information(self, "完成", "加权融合已运行，结果已显示在左侧文本框。")
        except Exception as e:
            QMessageBox.critical(self, "运行出错", str(e))

    def run_guoji_fusion_compare(self):
        try:
            from fusion_algorithm import run_fusion_comparison_experiment
        except ImportError:
            QMessageBox.warning(self, "模块未安装",
                                "请确保 fusion_algorithm、feature_engineering 可用。GAT 需 pip install torch torch_geometric。")
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        warnings.filterwarnings("ignore")
        try:
            out_dir = os.path.join(os.path.dirname(csv_path), "data", "processed")
            res = run_fusion_comparison_experiment(csv_path, out_dir=out_dir, save_boxplot=True)
            txt = "【融合对比：加权 vs GAT】\n\n"
            txt += f"加权得分 均值：{res['weighted_scores'].mean():.4f}\n"
            txt += f"GAT 得分 均值：{res['gat_scores'].mean():.4f}\n"
            if res.get("gat_degraded") and res.get("gat_degraded_reason"):
                txt += "\n【说明】\n" + res["gat_degraded_reason"] + "\n"
                txt += "\n因此箱线图右侧「GAT 融合」常显示为一条贴在 0 的线，并非加权融合也有问题。\n"
            if res.get("boxplot_path") and os.path.isfile(res["boxplot_path"]):
                txt += f"\n箱线图已保存：{res['boxplot_path']}\n"
                plt.figure(figsize=(5, 4))
                img = plt.imread(res["boxplot_path"])
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                self.embed_figure(
                    plt.gcf(),
                    description=(
                        "箱线图对比「规则加权融合」与「GAT 图注意力融合」在每张网格上的得分分布；"
                        "箱体与须须表示分位与离散程度，若离群点多说明工区内差异大。"
                        "若 GAT 侧退化，左侧文本会说明原因。"
                    ),
                )
            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            if res.get("gat_degraded"):
                QMessageBox.warning(
                    self,
                    "融合对比完成（GAT 侧无效或退化）",
                    "左侧已说明原因。若 GAT 全为 0，多半是未安装 torch_geometric；"
                    "安装后仍是一条线则可能是 GAT 输出无方差。",
                )
            else:
                QMessageBox.information(self, "完成", "融合对比已运行，箱线图已弹出。")
        except Exception as e:
            QMessageBox.critical(self, "运行出错", str(e))

    def run_guoji_train(self):
        import matplotlib.pyplot as plt
        plt.close('all')  # 强制清空历史残留画板

        try:
            from feature_engineering import build_feature_matrix
            from ml.train import train_xgboost_regression, save_model_report
        except ImportError as e:
            QMessageBox.warning(self, "模块未安装", f"请确保环境可用。\n{e}")
            return

        csv_path = self._get_guoji_csv()
        if not csv_path:
            return

        import warnings
        warnings.filterwarnings("ignore")

        try:
            r = build_feature_matrix(csv_path, out_processed_dir=None)
            X, y = r["X"], r["y"]
            if y is None:
                target_col, ok = QInputDialog.getText(
                    self,
                    "请指定目标列",
                    "未检测到目标列。请输入要预测的列名\n（如 Fracture Intensity B21、Connections per Branch）：",
                    text="Fracture Intensity B21",
                )
                if not ok or not target_col.strip():
                    return
                target_col = target_col.strip()
                r = build_feature_matrix(csv_path, target_column=target_col, out_processed_dir=None)
                X, y = r["X"], r["y"]
                if y is None:
                    QMessageBox.warning(self, "列名无效", f"在 CSV 中未找到列「{target_col}」，请确认列名拼写。")
                    return

            res = train_xgboost_regression(X, y, n_splits=5, test_size=0.1)
            model_dir = os.path.join(os.path.dirname(csv_path), "model")
            save_model_report(res, model_dir, name="xgboost_reg", feature_names=r.get("feature_names"))

            # 抓取训练生成的散点图并嵌入
            fig = plt.gcf()
            self.embed_figure(
                [fig],
                description=(
                    "XGBoost 训练过程输出的诊断图（具体子图以程序为准）：通常含训练/交叉验证相关的"
                    "预测效果或残差示意，用于快速判断拟合是否正常；详细数值见左侧文本与 model 目录报告。"
                ),
            )
            plt.close('all')

            txt = "【XGBoost 训练结果】\n\n"
            txt += f"CV R² 均值：{res['cv_agg']['R2_mean']:.4f}\n"
            txt += f"测试集 R²：{res['test_metrics']['R2']:.4f}\n"
            txt += f"\n模型已保存：{model_dir}/xgboost_reg.json"

            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QtGui.QTextCursor.End)
            QMessageBox.information(self, "完成", "训练完成，图表已刷新。")

        except Exception as e:
            QMessageBox.critical(self, "运行出错", str(e))

    def run_guoji_shap(self):
        import matplotlib.pyplot as plt
        plt.close('all')

        try:
            from ml.explain import explain_xgboost
        except ImportError:
            QMessageBox.warning(self, "模块未安装", "请确保 ml.explain 可用，并安装 shap。")
            return

        csv_path = self._get_guoji_csv()
        if not csv_path:
            return

        basic_model_dir = os.path.join(os.path.dirname(csv_path), "model")
        out_dir = basic_model_dir
        model_path = os.path.join(basic_model_dir, "xgboost_reg.json")

        if not os.path.isfile(model_path):
            QMessageBox.warning(self, "请先训练", "未找到基础训练模型！\n请先点击「训练 XGBoost」。")
            return

        import warnings
        warnings.filterwarnings("ignore")

        try:
            emph = None
            if self.combo_shap_features.currentIndex() > 0:
                emph = [self.combo_shap_features.currentText().strip()]
            df_imp = explain_xgboost(
                model_path, csv_path, out_dir=out_dir, emphasize_first=emph
            )

            txt = "【XGBoost SHAP 特征贡献分析】\n\n"
            if emph:
                txt += f"关注特征（图中置顶）：{emph[0]}\n\n"
            txt += df_imp.head(10).to_string()
            txt += f"\n\n（summary 图已保存至 {out_dir}/shap_summary.png）"

            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QtGui.QTextCursor.End)

            shap_png = os.path.join(out_dir, "shap_summary.png")
            if os.path.isfile(shap_png):
                fig = plt.figure(figsize=(8, 6))
                img = plt.imread(shap_png)
                plt.imshow(img)
                plt.axis("off")
                fig.tight_layout()
                self.embed_figure(
                    [fig],
                    description=(
                        "SHAP 摘要图：每一行对应一个输入特征；横轴为该特征对模型输出的 SHAP 贡献（影响方向与幅度）；"
                        "点色表示该样本上特征取值高低。可据此判断哪些拓扑/融合属性最能驱动当前目标列预测。"
                    ),
                )

            plt.close('all')
            QMessageBox.information(self, "完成", "SHAP 分析已成功运行，特征图已在右侧显示！")

        except Exception as e:
            QMessageBox.critical(self, "SHAP 运行遇到小麻烦",
                                 f"底层分析失败。建议先点击【训练 XGBoost】重新对齐数据再试。\n报错详情：{str(e)}")

    def run_spatial_topology_framework(self):
        """一键运行空间-拓扑融合学习框架"""
        import matplotlib.pyplot as plt
        plt.close('all')

        try:
            from spatial_topology_framework import run_spatial_topology_fusion_pipeline
        except ImportError as e:
            QMessageBox.warning(self, "模块未安装", f"请确保相关模块可用。\n{e}")
            return

        csv_path = self._get_guoji_csv()
        if not csv_path:
            return

        target_column, ok = QInputDialog.getText(
            self,
            "目标列名",
            "请输入要预测的目标列（例如 Fracture Intensity B21）：",
            text="Fracture Intensity B21",
        )
        if not ok or not target_column.strip():
            return
        target_column = target_column.strip()

        import warnings
        warnings.filterwarnings("ignore")

        try:
            res = run_spatial_topology_fusion_pipeline(
                csv_path=csv_path,
                target_column=target_column,
            )

            xgb_res = res.get("xgb_result", {})
            cv_agg = xgb_res.get("cv_agg", {})
            test_metrics = xgb_res.get("test_metrics", {})
            shap_df = res.get("shap_importance")

            out_dir = os.path.join(os.path.dirname(csv_path), "data", "processed")
            shap_png = os.path.join(out_dir, "shap_summary.png")

            if os.path.isfile(shap_png):
                fig = plt.figure(figsize=(8, 6))
                img = plt.imread(shap_png)
                plt.imshow(img)
                plt.axis("off")
                fig.tight_layout()
                self.embed_figure(
                    [fig],
                    description=(
                        "「一键空间–拓扑融合流水线」结束后的 SHAP 汇总图（若已生成）："
                        "含义与单独 SHAP 按钮相同，特征重要性针对流水线中指定的目标列。"
                    ),
                )

            plt.close('all')


            txt = "【空间-拓扑融合分析】\n\n"
            txt += f"目标列：{target_column}\n"
            if cv_agg:
                txt += "\n交叉验证（CV）指标：\n"
                if "R2_mean" in cv_agg: txt += f"  R² 均值：{cv_agg['R2_mean']:.4f}\n"
                if "RMSE_mean" in cv_agg: txt += f"  RMSE 均值：{cv_agg['RMSE_mean']:.4f}\n"
            if test_metrics:
                txt += "\n测试集指标：\n"
                if "R2" in test_metrics: txt += f"  R²：{test_metrics['R2']:.4f}\n"
            if shap_df is not None and not shap_df.empty:
                txt += "\nTop 特征贡献（SHAP）：\n"
                txt += shap_df.head(8).to_string(index=False)
                txt += "\n\n（SHAP 分析图已嵌入右侧画板）"

            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QtGui.QTextCursor.End)
            QMessageBox.information(self, "完成", "空间-拓扑融合运行完毕")

        except Exception as e:
            import traceback
            err_detail = traceback.format_exc()
            QMessageBox.critical(self, "运行出错", f"{str(e)}\n\n详细报错：\n{err_detail[-500:]}")

    def _set_ronghe_combo_tooltip(self):
        lines = ["融合方式："]
        lines.append("• PCA：直接可用")
        lines.append("• 自编码器/VAE：需 pip install torch" if not HAS_TORCH else "• 自编码器/VAE：已安装 torch")
        lines.append("• UMAP：需 pip install umap-learn" if not HAS_UMAP else "• UMAP：已安装 umap-learn")
        self.combo_fusion.setToolTip("\n".join(lines))

    def onIndexChanged(self, index):
        self.opt = index
        self.run_lunkuo()

    def onIndexChanged_2(self, index):
        if index == 1:
            self.run_tuopuhou1()
        elif index == 2:
            self.run_tuopuhou2()

    def run_yuantu(self):
        warnings.filterwarnings("ignore")
        # 原图仅展示迹线 + 研究区边界，无需构造 Network（Network 会裁剪迹线，CRS 不一致或不相交时会崩溃）
        if traces is None or area is None or traces.empty:
            QMessageBox.warning(self, "无数据", "请先切换数据源并确保迹线、研究区文件存在且非空。")
            return
        fig, ax = plt.subplots(1, 1, figsize=(9, 9 * rate))
        traces.plot(ax=ax, color="blue")
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.set_aspect('equal')
        for s in ax.spines.values():
            s.set_color("#0d0d0d")
            s.set_linewidth(1.35)
        self.embed_figure(
            fig,
            description="原始断裂迹线图：蓝色线为输入迹线，坐标系见标题；用于检查数据范围、与研究区是否一致，未做拓扑分类。",
        )

    def run_fenleihou(self):
        warnings.filterwarnings("ignore")
        network, nw_err = try_network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        fig, ax = plt.subplots(figsize=(9, 9 * rate))
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        network.branch_gdf.plot(
            colors=[assign_colors(bt) for bt in network.branch_types],
            ax=ax,
            aspect="equal",
        )
        handles = [
            plt.Line2D([0], [0], color="green", lw=2, label="CC_branch / X_node"),
            plt.Line2D([0], [0], color="blue", lw=2, label="CI_branch / Y_node"),
            plt.Line2D([0], [0], color="black", lw=2, label="II_branch / I_node"),
            plt.Line2D([0], [0], color="red", lw=2, label="Other / Boundary"),
        ]
        ax.legend(handles=handles, loc='lower left')
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.set_aspect('equal')
        for s in ax.spines.values():
            s.set_color("#0d0d0d")
            s.set_linewidth(1.35)
        self.embed_figure(
            fig,
            description=(
                "分类后迹线图：按 fractopo 分支类型（CC/CI/II）对线段着色；图例中绿色/蓝色/黑色对应不同分支类，"
                "红色为无法归类或边界相关；反映拓扑划分后的空间模式。"
            ),
        )

    def run_tuopuhou1(self):
        warnings.filterwarnings("ignore")
        network, nw_err = try_network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        fig, ax = plt.subplots(figsize=(9, 9 * rate))
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        network.trace_gdf.plot(ax=ax, linewidth=0.5, aspect="equal")
        network.node_gdf.plot(
            c=[assign_colors(bt) for bt in network.node_types],
            ax=ax,
            markersize=10,
            aspect="equal",
        )
        area.boundary.plot(ax=ax, color="red", aspect="equal")
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="green", markersize=10, label="X_node"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="blue", markersize=10, label="Y_node"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="black", markersize=10, label="I_node"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="red", markersize=10, label="Other / Boundary"),
        ]
        ax.legend(handles=handles, loc='lower left')
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.set_aspect('equal')
        self.embed_figure(
            fig,
            description=(
                "拓扑化视图 1：浅色为迹线；节点按 X/Y/I 类型着色（见左下角图例示意），红线为研究区边界。"
                "用于核对节点识别是否落在迹线交点等位置。"
            ),
        )

    def run_tuopuhou2(self):
        warnings.filterwarnings("ignore")
        # Drop duplicates from the trace GeoDataFrame
        traces.drop_duplicates(subset="geometry", inplace=True)
        # Reset the index of the GeoDataFrame
        traces.reset_index(drop=True, inplace=True)
        network, nw_err = try_network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        # 定义节点类型到颜色的映射
        type_to_color = {
            'E': 'red',  # 假设E类型节点用红色表示
            'I': 'green',  # 假设I类型节点用绿色表示
            'X': 'blue',  # 假设X类型节点用蓝色表示
            'Y': 'yellow',  # 假设Y类型节点用黄色表示
        }
        # 定义节点类型到形状的映射
        type_to_shape = {
            'E': 'o',
            'I': 'o',
            'X': '^',
            'Y': '*',
        }
        # 开始绘图
        fig, ax = plt.subplots(figsize=(9, 9 * rate))

        # 按分支类型分组绘制（branch_gdf）；fractopo≥0.9 用 Connection（C - C 等），旧版曾用 Class（CC/CI/II）
        bg = network.branch_gdf
        if CONNECTION_COLUMN in bg.columns:
            branch_draw = [
                (CC_branch, "red", "CC"),
                (CI_branch, "green", "CI"),
                (II_branch, "blue", "II"),
            ]
            for conn_val, color, leg in branch_draw:
                subset = bg[bg[CONNECTION_COLUMN] == conn_val]
                if not subset.empty:
                    subset.plot(
                        ax=ax,
                        color=color,
                        linewidth=1,
                        label=leg,
                        aspect="equal",
                    )
        elif "Class" in bg.columns:
            for branch_type, color in (("CC", "red"), ("CI", "green"), ("II", "blue")):
                subset = bg[bg["Class"] == branch_type]
                if not subset.empty:
                    subset.plot(
                        ax=ax,
                        color=color,
                        linewidth=1,
                        label=branch_type,
                        aspect="equal",
                    )
        else:
            QMessageBox.warning(
                self,
                "无法绘制分支",
                "branch_gdf 中未找到「Connection」或「Class」列，可能与当前 fractopo 版本不兼容。",
            )
            return

        # 遍历每个节点类型，绘制对应类型的节点
        for node_type in type_to_color.keys():
            nodes = network.node_gdf[network.node_gdf[CLASS_COLUMN] == node_type]
            if not nodes.empty:
                ax.scatter(nodes.geometry.x, nodes.geometry.y, s=50,
                           c=type_to_color[node_type], marker=type_to_shape[node_type], label=node_type, zorder=5)
        area.boundary.plot(ax=ax, color="red", aspect="equal")
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.legend(title=' Type')
        ax.set_aspect('equal')
        self.embed_figure(
            fig,
            description=(
                "拓扑化视图 2：彩色线段表示分支连接类型（C-C / C-I / I-I 等 fractopo Connection 记号），"
                "节点散点为 X/Y/I/E 类型；与视图 1 互补，侧重「线段-节点」联合展示。"
            ),
        )

    def run_tuopushuxing(self):
        warnings.filterwarnings("ignore")
        network, nw_err = try_network(
            traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
            circular_target_area=False, snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        parameters = 'parameters'.ljust(40, ' ') + 'values' + "\n"
        for key, value in network.parameters.items():
            parameters = parameters + str(key).ljust(40, ' ') + str(value) + "\n"
        self.text_browser.clear()
        self.text_browser.insertPlainText(parameters)
        self.text_browser.moveCursor(QTextCursor.End)

    def run_azimuth(self):
        setup_matplotlib_chinese()
        network, nw_err = try_network(
            name=name,
            trace_gdf=traces,
            area_gdf=area,
            truncate_traces=True,
            circular_target_area=False,
            determine_branches_nodes=True,
            snap_threshold=0.001,
            azimuth_set_names=("N-S", "E-W"),
            azimuth_set_ranges=((135, 45), (45, 135)),
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        pprint((network.azimuth_set_names, network.azimuth_set_ranges))
        pprint(network.trace_azimuth_set_counts)
        fig, ax = plt.subplots(figsize=(9, 9 * rate))
        colors = ("red", "blue")
        assert len(colors) == len(network.azimuth_set_names)
        for azimuth_set, set_range, color in zip(network.azimuth_set_names, network.azimuth_set_ranges, colors):
            trace_gdf_set = network.trace_gdf.loc[network.trace_gdf["azimuth_set"] == azimuth_set]
            trace_gdf_set.plot(color=color, label=f"{azimuth_set} - {set_range}", ax=ax)
        zh_fonts = plt.rcParams.get("font.sans-serif", [])
        font_family = zh_fonts[0] if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0 else "Microsoft YaHei"
        ax.set_title(f"方位角集图 - {name}", fontsize=14, fontfamily=font_family)
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.set_aspect('equal')
        for s in ax.spines.values():
            s.set_color("#0d0d0d")
            s.set_linewidth(1.35)
        plt.legend()
        self.embed_figure(
            fig,
            description=(
                "方位角集图：不同颜色对应不同方位组（如 N-S 与 E-W 及角度范围）；"
                "用于查看各走向迹线在工区内的分布是否分组明显。"
            ),
        )

    def run_relitu(self):
        warnings.filterwarnings("ignore")
        # 热力图 为每条线段生成一系列的点
        points = []
        for geom in traces.geometry:
            points.extend(list(geom.interpolate(distance=5, normalized=True).coords))
        x, y = np.array(points).T
        kde = gaussian_kde(np.vstack([x, y]))
        kde_values = kde(np.vstack([x, y]))
        fig, ax = plt.subplots(figsize=(9, 9 * rate))
        scatter = ax.scatter(x, y, c=kde_values, cmap="Reds", s=10, alpha=0.5)
        plt.title("Fracture density heatmap " + name)
        plt.axis("equal")
        ax.set_aspect('equal')
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        for s in ax.spines.values():
            s.set_color("#0d0d0d")
            s.set_linewidth(1.35)
        self.embed_figure(
            fig,
            description=(
                "断裂密度热力图：沿迹线加密采样点后做核密度估计，颜色越暖表示该处线密度越高；"
                "反映断裂在平面上的聚集带，并非网格 CSV 中的属性。"
            ),
        )

    def a(self):
        warnings.filterwarnings("ignore")
        network, nw_err = try_network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        fit, fig1, ax = network.plot_trace_lengths()
        # ax.set_aspect('equal')
        fit, fig2, ax = network.plot_branch_lengths()
        # ax.set_aspect('equal')

        # 长度分布拟合线配色：主模型/对比模型，并降低饱和感（alpha）
        fit_line_colors = {
            "power": "#d62728",   # Power-law
            "lognormal": "#1f77b4",
            "exponential": "#2ca02c",
        }
        for fig in (fig1, fig2):
            for one_ax in fig.axes:
                for line in one_ax.get_lines():
                    label = (line.get_label() or "").lower()
                    if "power" in label:
                        line.set_color(fit_line_colors["power"])
                        line.set_alpha(0.7)
                    elif "lognormal" in label:
                        line.set_color(fit_line_colors["lognormal"])
                        line.set_alpha(0.7)
                    elif "exponential" in label:
                        line.set_color(fit_line_colors["exponential"])
                        line.set_alpha(0.7)
                # 同步图例与注释文字颜色
                legend = one_ax.get_legend()
                if legend is not None:
                    handles = getattr(legend, "legend_handles", None)
                    if handles is None:
                        handles = getattr(legend, "legendHandles", [])
                    for handle, txt in zip(handles, legend.get_texts()):
                        tlabel = (txt.get_text() or "").lower()
                        if "power" in tlabel:
                            color = fit_line_colors["power"]
                        elif "lognormal" in tlabel:
                            color = fit_line_colors["lognormal"]
                        elif "exponential" in tlabel:
                            color = fit_line_colors["exponential"]
                        else:
                            continue
                        if hasattr(handle, "set_color"):
                            handle.set_color(color)
                        if hasattr(handle, "set_alpha"):
                            handle.set_alpha(0.7)
                        txt.set_color(color)
                for txt in one_ax.texts:
                    tlabel = (txt.get_text() or "").lower()
                    if "power" in tlabel:
                        txt.set_color(fit_line_colors["power"])
                    elif "lognormal" in tlabel:
                        txt.set_color(fit_line_colors["lognormal"])
                    elif "exponential" in tlabel:
                        txt.set_color(fit_line_colors["exponential"])

        self.embed_figure(
            [fig1, fig2],
            descriptions=[
                "迹线长度分布直方图及幂律、对数正态、指数等典型拟合曲线；用于判断标度律与共守分布形态。",
                "分支长度分布及同样拟合对比；分支由迹线拓扑分解得到，长度统计与迹线层可对照阅读。",
            ],
        )

    def run_meiguitu(self):
        warnings.filterwarnings("ignore")
        setup_matplotlib_chinese()
        network, nw_err = try_network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        azimuth_bin_dict, fig1, ax = network.plot_trace_azimuth()
        azimuth_bin_dict, fig2, ax = network.plot_branch_azimuth()
        # 仅覆盖玫瑰图配色，不改动其他绘图逻辑
        for one_ax in fig1.axes:
            for patch in one_ax.patches:
                patch.set_facecolor("#2C3E50")  # 迹线玫瑰图填充
                patch.set_edgecolor("black")    # 迹线玫瑰图边框
                patch.set_alpha(0.65)           # 降低深蓝灰不透明度
        for one_ax in fig2.axes:
            for patch in one_ax.patches:
                patch.set_facecolor("#AED6F1")  # 分支玫瑰图填充
                patch.set_edgecolor("#2E86C1")  # 分支玫瑰图边框（更深）
                patch.set_alpha(0.65)           # 与迹线玫瑰图透明度一致
        zh_fonts = plt.rcParams.get("font.sans-serif", [])
        font_family = zh_fonts[0] if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0 else "Microsoft YaHei"
        # 覆盖 fractopo 默认标题字体，确保数据源中文名正常显示
        for fig, title_text in (
            (fig1, f"迹线玫瑰图 - {name}"),
            (fig2, f"分支玫瑰图 - {name}"),
        ):
            for one_ax in fig.axes:
                one_ax.set_title(title_text, fontfamily=font_family, fontsize=14)

        self.embed_figure(
            [fig1, fig2],
            descriptions=[
                "迹线方位玫瑰图：极坐标下各走向区间频数，峰值方向即优势构造走向。",
                "分支方位玫瑰图：对拓扑分支线段统计走向，可与迹线玫瑰图对比构造与分解后差异。",
            ],
        )

    def run_sanyuantu(self):
        warnings.filterwarnings("ignore")

        setup_matplotlib_chinese()

        network, nw_err = try_network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return

        fig1, ax1, tax1 = network.plot_xyi()
        ax1.axis('off')
        fig1.set_size_inches(10, 10)
        # 三角与顶点标签留出边距，避免 X/Y/I 贴边被裁切
        ax1.set_position([0.13, 0.13, 0.62, 0.58])
        fig1.subplots_adjust(left=0.06, right=0.94, bottom=0.07, top=0.90)

        fig2, ax2, tax2 = network.plot_branch()
        ax2.axis('off')
        fig2.set_size_inches(10, 10)
        ax2.set_position([0.13, 0.13, 0.62, 0.58])
        fig2.subplots_adjust(left=0.06, right=0.94, bottom=0.07, top=0.90)

        _style_ternary_plot(fig1, tax1)
        _style_ternary_plot(fig2, tax2)
        _polish_fractopo_ternary_labels(fig1)
        _polish_fractopo_ternary_labels(fig2)

        # 顶部中文标题 + 图例中数据源名称（fractopo 默认 DejaVu Sans 会导致中文成方框）
        zh_fonts = plt.rcParams.get("font.sans-serif", [])
        font_family = zh_fonts[0] if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0 else "Microsoft YaHei"
        for fig, ttl in (
            (fig1, f"节点类型三元图（XYI）- {name}"),
            (fig2, f"分支类型三元图（CC/CI/II）- {name}"),
        ):
            fig.suptitle(ttl, fontsize=14, fontfamily=font_family, y=0.96)
            for ax in fig.axes:
                leg = ax.get_legend()
                if leg is not None:
                    for t in leg.get_texts():
                        t.set_fontfamily(font_family)
                    title = leg.get_title()
                    if title is not None:
                        title.set_fontfamily(font_family)
            for leg in getattr(fig, "legends", []):
                for t in leg.get_texts():
                    t.set_fontfamily(font_family)

        self.embed_figure(
            [fig1, fig2],
            descriptions=[
                "节点类型三元图（XYI）：三角形顶点为 X、Y、I 三类节点占比，落在三角形内的点云表示样本整体组成。",
                "分支类型三元图（CC、CI、II）：三端元为三类分支在数量或长度加权下的比例（定义见 fractopo）。",
            ],
        )

    def run_guanxi(self):
        warnings.filterwarnings("ignore")
        network, nw_err = try_network(
            traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
            circular_target_area=False, snap_threshold=0.001,
        )
        if nw_err:
            QMessageBox.warning(self, "无法构建断裂网络", nw_err)
            return
        print(f"Azimuth set names: {network.azimuth_set_names}")
        print(f"Azimuth set ranges: {network.azimuth_set_ranges}")
        figs, fig_axes = network.plot_azimuth_crosscut_abutting_relationships()
        # 覆盖 fractopo 默认配色：cross-cut / A→B / B→A
        relationship_colors = ("#4A5568", "#2B6CB0", "#63B3ED")
        for fig in figs:
            for ax in fig.axes:
                if not hasattr(ax, "containers"):
                    continue
                for container in ax.containers:
                    # 该图每个子图仅有一组3根柱子
                    if len(container) >= 3:
                        for patch, color in zip(container[:3], relationship_colors):
                            patch.set_facecolor(color)
                            patch.set_edgecolor("black")
                        break
                legend = ax.get_legend()
                if legend is not None:
                    # 显式同步图例色块，避免与柱体颜色不一致
                    handles = getattr(legend, "legend_handles", None)
                    if handles is None:
                        handles = getattr(legend, "legendHandles", [])
                    for handle, color in zip(handles, relationship_colors):
                        if hasattr(handle, "set_facecolor"):
                            handle.set_facecolor(color)
                        if hasattr(handle, "set_edgecolor"):
                            handle.set_edgecolor("black")
                    leg_txts = legend.get_texts()
                    if leg_txts:
                        ax.set_title(
                            leg_txts[0].get_text(),
                            fontsize=11,
                            fontweight="bold",
                            pad=14,
                            fontfamily="DejaVu Sans",
                        )
                    legend.set_loc("upper left")
                # 仅给右侧「trace count」侧栏加框，避免整块轴域文字误加框导致拥挤
                for txt in ax.texts:
                    if "trace count" in txt.get_text():
                        txt.set_bbox(
                            dict(
                                boxstyle="round,pad=0.35",
                                facecolor="white",
                                edgecolor="#9CA3AF",
                                alpha=0.95,
                            )
                        )
                        txt.set_clip_on(False)
        for fig in figs:
            # 标题显示当前数据源名称，并在柱形图上端居中
            if hasattr(fig, "_suptitle") and fig._suptitle is not None:
                # 保持中文标题，并沿用全局 matplotlib 中文字体配置
                fig._suptitle.set_text(str(name))
                zh_fonts = plt.rcParams.get("font.sans-serif", [])
                if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0:
                    fig._suptitle.set_fontfamily(zh_fonts[0])
                fig._suptitle.set_fontsize(15)
                fig._suptitle.set_bbox(
                    dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#9CA3AF", alpha=0.95)
                )
                fig._suptitle.set_x(0.5)
                fig._suptitle.set_y(0.96)
                fig._suptitle.set_ha("center")
                fig._suptitle.set_va("top")
            fig.subplots_adjust(left=0.07, right=0.78, top=0.82, bottom=0.14, wspace=0.34)
            fig.set_size_inches(15, 7.8)

        if figs:
            _cap_rel = (
                "交叉与相邻关系图：各子图表示两方位集之间交切（cross-cut）与不同方向邻接（abutting）的计数统计；"
                "柱色对应图例中关系类型；侧栏为 trace count。翻页可浏览不同方位集组合。"
            )
            self.embed_figure(figs, descriptions=[_cap_rel] * len(figs))

    def b(self):
        branches, nodes = branches_and_nodes(traces, area, snap_threshold=0.001)
        # 左右并排，避免上下叠图时标题与坐标轴标签互相遮挡
        h_in = max(7.5, 8.0 * float(rate))
        fig, axes = plt.subplots(1, 2, figsize=(17, h_in), sharex=True, sharey=True)
        ax0, ax1 = axes[0], axes[1]
        traces.plot(ax=ax0, color="blue", label="Traces")
        area.boundary.plot(ax=ax0, color="black", label="Target Area", linestyle="dashed")
        ax0.set_title("Traces & Target Area", fontsize=12, pad=12)
        nodes.plot(ax=ax1, column="Class", zorder=10, legend=False, categorical=True, markersize=7)
        ax1.set_title("Branches & Nodes & Area", fontsize=12, pad=12)
        area.boundary.plot(ax=ax1, color="black", linestyle="dashed")
        for ax in (ax0, ax1):
            ax.set_xlim(left - width, right + width)
            ax.set_ylim(down - height, up + height)
            area.boundary.plot(ax=ax, color="red")
            ax.set_aspect("equal")
            xa0, xa1 = ax.get_xlim()
            ya0, ya1 = ax.get_ylim()
            mx = max(abs(xa0), abs(xa1), abs(ya0), abs(ya1))
            if mx >= 1e5:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
                ax.set_xlabel("Easting (×10⁶ m)")
                if ax is ax0:
                    ax.set_ylabel("Northing (×10⁶ m)")
            else:
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
                ax.set_xlabel("X")
                if ax is ax0:
                    ax.set_ylabel("Y")
            ax.tick_params(axis="both", labelsize=9)

        class_order = [c for c in ("X", "Y", "I", "E") if c in nodes["Class"].dropna().unique()]
        if class_order:
            cmap = plt.get_cmap("tab10")
            handles = [
                Line2D([0], [0], marker="o", linestyle="", markersize=7,
                       markerfacecolor=cmap(i), markeredgecolor="black", label=cls)
                for i, cls in enumerate(class_order)
            ]
            legend = ax1.legend(
                handles=handles,
                title="Node Type",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
            )
            leg_handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
            for handle in leg_handles:
                if hasattr(handle, "_sizes"):
                    handle._sizes = [20]
        fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.14, wspace=0.18)
        self.embed_figure(
            fig,
            description=(
                "左：原始迹线与研究区边界；右：节点类型（X/Y/I 等）在平面上的位置及研究区；"
                "两图共用坐标比例，便于与拓扑化视图对照检查识别结果。"
            ),
        )

    def _plot_contour_safe(self, network, sampled_grid, parameters):
        import pandas as pd
        import numpy as np

        if isinstance(parameters, str):
            parameters = [parameters]

        for param in parameters:
            if param not in sampled_grid.columns:
                continue

            grid_plot = sampled_grid.copy()
            grid_plot = grid_plot[grid_plot.geometry.notna()]
            grid_plot = grid_plot[~grid_plot.geometry.is_empty]
            grid_plot = grid_plot[grid_plot.geometry.is_valid]

            if grid_plot.empty:
                print(f"⚠️ 参数 {param} 的有效网格为空，跳过绘制。")
                continue

            grid_plot[param] = pd.to_numeric(grid_plot[param], errors='coerce')
            grid_plot[param] = grid_plot[param].replace([np.inf, -np.inf], np.nan).fillna(0)

            try:
                fig, ax = plt.subplots(figsize=(9, 8))
                ax.grid(False)

                centroids = grid_plot.geometry.centroid
                x = centroids.x.values
                y = centroids.y.values
                z = grid_plot[param].values

                try:
                    network.trace_gdf.plot(ax=ax, color='black', linewidth=0.5, alpha=0.3)
                except:
                    pass

                contour = ax.tricontourf(x, y, z, levels=50, cmap='plasma', alpha=0.85)

                cbar = fig.colorbar(contour, ax=ax)
                cbar.set_label(param)

                ax.set_aspect("equal")
                ax.set_title(f"平滑热力图: {param}", fontsize=14, pad=15)
                plt.tight_layout()

                _cap = (
                    f"本图为网格拓扑参数「{param}」的平滑填色：对网格中心做三角剖分后插值填色；"
                    f"色条为该指标量纲；浅底为迹线叠置。坐标为当前投影平面。"
                )
                try:
                    self.embed_figure([fig], description=_cap)
                except TypeError:
                    self.embed_figure(fig, description=_cap)


            except Exception as e:
                print(f"平滑渲染失败: {param}。原因: {str(e)}")
                try:
                    plt.close(fig)
                except:
                    pass

    def run_lunkuo(self):
        warnings.filterwarnings("ignore")
        print(f"当前选择的绘图选项: {self.opt}")


        progress = QtWidgets.QProgressDialog("正在进行空间计算与高清渲染，请耐心等待...", None, 0, 0, self)
        progress.setWindowTitle("系统运算中")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

        try:
            traces_local = traces  # 初始化，预处理块内会重新赋值为副本

            try:
                traces_local = traces.copy()
                bounds = traces_local.total_bounds

                if area is not None:
                    minx, miny, maxx, maxy = area.total_bounds
                    spatial_index = traces_local.sindex
                    possible_matches_index = list(spatial_index.intersection((minx, miny, maxx, maxy)))
                    traces_local = traces_local.iloc[possible_matches_index]

                map_span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                dp_tolerance = map_span * 0.002

                if dp_tolerance > 0:
                    traces_local = traces_local.copy()
                    traces_local.geometry = traces_local.geometry.simplify(tolerance=dp_tolerance, preserve_topology=True)

            except Exception as e:
                print(f"算法预处理跳过。原因: {e}")
                traces_local = traces.copy()

            bounds = traces_local.total_bounds
            dynamic_width = (bounds[2] - bounds[0]) / 20.0

            if dynamic_width <= 0:
                dynamic_width = 100.0

            print(f"正在执行空间拓扑计算 (网格大小: {dynamic_width:.4f})...")
            QtWidgets.QApplication.processEvents()

            network, nw_err = try_network(
                traces_local,
                area,
                name=name,
                determine_branches_nodes=True,
                truncate_traces=True,
                circular_target_area=False,
                snap_threshold=0.001,
            )
            if nw_err:
                print(f"❌ {nw_err}")
                QMessageBox.warning(self, "无法构建断裂网络", nw_err)
                return

            sampled_grid = network.contour_grid(cell_width=dynamic_width)
            print("拓扑网格计算完成，准备渲染！")

            progress.setLabelText("计算完成，正在生成高清平滑图像...")
            QtWidgets.QApplication.processEvents()

            # --- 路由分发绘图请求 ---
            if self.opt == 1:
                self._plot_contour_safe(network, sampled_grid, ["Fracture Intensity B21", "Fracture Intensity P21"])
            elif self.opt == 2:
                self._plot_contour_safe(network, sampled_grid,
                                        ["Trace Min Length", "Trace Max Length", "Trace Mean Length"])
            elif self.opt == 3:
                self._plot_contour_safe(network, sampled_grid,
                                        ["Dimensionless Intensity B22", "Dimensionless Intensity P22"])
            elif self.opt == 4:
                self._plot_contour_safe(network, sampled_grid, "Number of Traces (Real)")
            elif self.opt == 5:
                self._plot_contour_safe(network, sampled_grid,
                                        ["Branch Min Length", "Branch Max Length", "Branch Mean Length"])
            elif self.opt == 6:
                self._plot_contour_safe(network, sampled_grid, ["Areal Frequency B20", "Areal Frequency P20"])
            elif self.opt == 7:
                self._plot_contour_safe(network, sampled_grid, ["Connections per Trace", "Connections per Branch"])
            elif self.opt == 8:
                self._plot_contour_safe(network, sampled_grid, "Connection Frequency")

        except Exception as e:
            print(f"❌ 运行报错: {str(e)}")

        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            progress.close()

    def run_ronghe(self):
        if run_fusion_pipeline is None:
            QMessageBox.warning(
                self, "模块未安装",
                "请确保 topology_fusion 模块可用（与 main.py 同目录），并已安装 scikit-learn。",
            )
            return
        method = self.combo_fusion.currentText().replace("融合方法: ", "").strip()
        if method in ("自编码器", "VAE") and not HAS_TORCH:
            QMessageBox.warning(
                self, "需要安装 PyTorch",
                f"「{method}」依赖 PyTorch。请执行：pip install torch\n或先选择 PCA/UMAP。",
            )
            return
        if method == "UMAP" and not HAS_UMAP:
            QMessageBox.warning(
                self, "需要安装 umap-learn",
                "「UMAP」依赖 umap-learn。请执行：pip install umap-learn\n或先选择 PCA。",
            )
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        warnings.filterwarnings("ignore")
        try:
            if method == "自编码器":
                df_out, scaler, kmeans, cluster_means = run_fusion_pipeline_ae(
                    csv_path, n_latent=2, n_clusters=4, ae_epochs=100
                )
                x_col, y_col = "Z1", "Z2"
                method_name = "自编码器"
            elif method == "UMAP":
                df_out, scaler, _, kmeans, cluster_means = run_fusion_pipeline_umap(
                    csv_path, n_components=2, n_clusters=4, n_neighbors=15
                )
                x_col, y_col = "U1", "U2"
                method_name = "UMAP"
            elif method == "VAE":
                df_out, scaler, kmeans, cluster_means = run_fusion_pipeline_vae(
                    csv_path, n_latent=2, n_clusters=4, vae_epochs=150
                )
                x_col, y_col = "Z1", "Z2"
                method_name = "VAE"
            else:
                df_out, scaler, pca, kmeans, cluster_means = run_fusion_pipeline(
                    csv_path, n_components=2, n_clusters=4
                )
                x_col, y_col = "PC1", "PC2"
                method_name = "PCA"
        except Exception as e:
            QMessageBox.critical(self, "运行出错", f"属性融合或聚类时出错：\n{str(e)}")
            return
        n_samples = len(df_out)
        n_clusters = int(df_out["cluster_id"].max()) + 1
        spatial_ok = all(
            c in df_out.columns
            for c in (
                "vertex1_x",
                "vertex1_y",
                "vertex2_x",
                "vertex2_y",
                "vertex3_x",
                "vertex3_y",
                "vertex4_x",
                "vertex4_y",
            )
        )
        if spatial_ok:
            # 左右排列；尺寸略小于原先的 13.5×5.8，减轻占屏
            fig1, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(11.6, 5.25),
                gridspec_kw={"wspace": 0.26},
            )
        else:
            fig1, ax1 = plt.subplots(figsize=(7, 6))
            ax2 = None
        ax1.set_axisbelow(True)
        ax1.grid(True, alpha=0.42, linestyle="-", linewidth=0.55, color="0.75", zorder=0.3)
        ax1.set_facecolor("#e8ebf2")
        has_center_legend, cmap_latent, norm_latent = self._plot_latent_fusion_kmeans_regions(
            ax1, df_out, x_col, y_col, kmeans, n_clusters
        )
        _skw = dict(
            s=19,
            alpha=0.9,
            zorder=3,
            edgecolors="white",
            linewidths=0.32,
        )
        if cmap_latent is not None and norm_latent is not None:
            scatter = ax1.scatter(
                df_out[x_col],
                df_out[y_col],
                c=df_out["cluster_id"],
                cmap=cmap_latent,
                norm=norm_latent,
                **_skw,
            )
        else:
            scatter = ax1.scatter(
                df_out[x_col],
                df_out[y_col],
                c=df_out["cluster_id"],
                cmap="tab10",
                **_skw,
            )
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title(
            f"拓扑属性融合（{method_name}）：{x_col}–{y_col}"
            f"\n（柔和底色=KMeans 分区｜散点=网格单元）",
            fontsize=11,
        )
        plt.colorbar(scatter, ax=ax1, label="cluster_id")
        if has_center_legend:
            leg = ax1.legend(loc="best", fontsize=8, framealpha=0.92)
            if leg is not None:
                leg.set_zorder(6)
        ax1.set_aspect("equal", adjustable="datalim")
        if ax2 is not None:
            if not self._plot_spatial_cluster_grid(ax2, df_out, n_clusters, method_name):
                ax2.text(0.5, 0.5, "无法绘制空间网格（缺少顶点列）", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_axis_off()
        plt.tight_layout()
        self.embed_figure(
            fig1,
            description=(
                "属性融合与聚类：左图为多拓扑指标降维后的潜空间（散点为网格、底色为 KMeans 分区），"
                "右图为同一聚类编号在平面网格上的空间分布（绿–蓝为簇编号）。"
                "用于观察属性相似簇是否在空间上成片出现。"
            ),
        )
        summary_lines = [
            "【智能拓扑分析结果】",
            "",
            f"融合方式：{method_name}",
            f"数据：{os.path.basename(csv_path)}",
            f"有效网格数：{n_samples}",
            f"聚类数：{n_clusters}",
            f"新属性：{x_col}, {y_col}, cluster_id",
            "",
            "各簇在部分拓扑属性上的均值：",
            "",
            cluster_means.head(4).to_string(),
        ]
        self.text_browser.clear()
        self.text_browser.insertPlainText("\n".join(summary_lines))
        self.text_browser.moveCursor(QTextCursor.End)
        QMessageBox.information(
            self, "运行完成",
            f"已用 {method_name} 生成 {x_col}、{y_col} 与 {n_clusters} 类聚类结果。",
        )


import traceback


def exception_hook(exctype, value, tb):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("\n" + "=" * 50)
    print("系统崩溃：")
    print("".join(traceback.format_exception(exctype, value, tb)))
    print("=" * 50 + "\n")
    sys.exit(1)


if __name__ == "__main__":
    sys.excepthook = exception_hook  # 挂载防崩溃

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
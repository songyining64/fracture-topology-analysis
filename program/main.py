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

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 使用微软雅黑或黑体
plt.rcParams['axes.unicode_minus'] = False

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QComboBox, QFrame, QHBoxLayout, QVBoxLayout, QLabel, \
    QPushButton, QInputDialog

from fractopo.branches_and_nodes import branches_and_nodes
from pprint import pprint
from matplotlib.lines import Line2D
from demo import Ui_MainWindow
from fractopo.general import CC_branch, CI_branch, II_branch, X_node, Y_node, I_node
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

# 数据源：取消注释要使用的区块，保证 name 与 traces/area 一致
trace_data_url = "KB11/KB11_traces.geojson"
area_data_url = "KB11/KB11_area.geojson"
traces = gpd.read_file(trace_data_url)
area = gpd.read_file(area_data_url)
name = "KB11"

# trace_data_url = "THK/thkceshi-landmark1.geojson"
# area_data_url = "THK/my_area.geojson"
# traces = gpd.read_file(trace_data_url)
# area = gpd.read_file(area_data_url)
# name = "Yingmai 2 area in Tarim Basin"

# trace_data_url = "MY/11.geojson"
# area_data_url = "MY/my_area1.geojson"
# traces = gpd.read_file(trace_data_url)
# area = gpd.read_file(area_data_url)
# name = "MY"

traces.drop_duplicates(subset="geometry", inplace=True)
traces.reset_index(drop=True, inplace=True)

geometry = traces.geometry.tolist()
left, right, down, up = math.inf, -math.inf, math.inf, -math.inf
for one in geometry:
    left = min(left, one.boundary.bounds[0])
    right = max(right, one.boundary.bounds[2])
    down = min(down, one.boundary.bounds[1])
    up = max(up, one.boundary.bounds[3])
rate = (up - down) / (right - left)
width, height = 0.01 * (right - left), 0.01 * (up - down)


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


    def embed_figure(self, figs):
        if not isinstance(figs, list):
            figs = [figs]

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

        if len(self.current_figs) > 1:
            self.lbl_fig_status.setText(f"第 {self.current_fig_idx + 1} 张 / 共 {len(self.current_figs)} 张")
            self.btn_prev_fig.setEnabled(self.current_fig_idx > 0)
            self.btn_next_fig.setEnabled(self.current_fig_idx < len(self.current_figs) - 1)

        QtWidgets.QApplication.processEvents()


    def _get_guoji_csv(self):
        csv_name = "Yingmai 2 area in Tarim Basin.csv"
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
        if not os.path.isfile(csv_path):
            QMessageBox.warning(self, "未找到数据",
                                f"未找到：{csv_name}\n请先运行 data export.py 或将该文件放在 program 目录。")
            return None
        return csv_path

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
            if res.get("boxplot_path") and os.path.isfile(res["boxplot_path"]):
                txt += f"\n箱线图已保存：{res['boxplot_path']}\n"
                plt.figure(figsize=(5, 4))
                img = plt.imread(res["boxplot_path"])
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                self.embed_figure(plt.gcf())
            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            QMessageBox.information(self, "完成", "融合对比已运行，箱线图已弹出。")
        except Exception as e:
            QMessageBox.critical(self, "运行出错", str(e))

    def run_guoji_train(self):
        try:
            from feature_engineering import build_feature_matrix
            from ml.train import train_xgboost_regression, save_model_report
        except ImportError as e:
            QMessageBox.warning(self, "模块未安装", f"请确保 feature_engineering、ml.train 可用，并安装 xgboost。\n{e}")
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        warnings.filterwarnings("ignore")
        try:
            r = build_feature_matrix(csv_path, out_processed_dir=None)
            X, y = r["X"], r["y"]
            if y is None:
                y = r["X"][:, 0]
            if len(X) < 10:
                QMessageBox.warning(
                    self, "样本过少",
                    f"特征工程后有效样本仅 {len(X)} 个，至少需要约 10 个才能做 5 折 CV。请检查 CSV 或先运行 data export 生成完整网格数据。",
                )
                return
            res = train_xgboost_regression(X, y, n_splits=5, test_size=0.1)
            model_dir = os.path.join(os.path.dirname(csv_path), "model")
            save_model_report(res, model_dir, name="xgboost_reg", feature_names=r.get("feature_names"))
            txt = "【XGBoost 训练结果】\n\n"
            txt += f"CV R² 均值：{res['cv_agg']['R2_mean']:.4f}\n"
            txt += f"CV MAE 均值：{res['cv_agg']['MAE_mean']:.4f}\n"
            txt += f"测试集 R²：{res['test_metrics']['R2']:.4f}\n"
            txt += f"测试集 MAE：{res['test_metrics']['MAE']:.4f}\n"
            txt += f"\n模型已保存：{model_dir}/xgboost_reg.json"
            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            QMessageBox.information(self, "完成", "训练完成，模型已保存至 model/。")
        except Exception as e:
            import traceback
            err_detail = traceback.format_exc()
            QMessageBox.critical(
                self, "运行出错",
                f"{str(e)}\n\n详细报错：\n{err_detail[-800:]}",
            )

    def run_guoji_shap(self):
        try:
            from ml.explain import explain_xgboost
        except ImportError:
            QMessageBox.warning(self, "模块未安装", "请确保 ml.explain 可用，并安装 shap。")
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        model_path = os.path.join(os.path.dirname(csv_path), "model", "xgboost_reg.json")
        if not os.path.isfile(model_path):
            QMessageBox.warning(self, "请先训练", "未找到 model/xgboost_reg.json，请先点击「训练 XGBoost」。")
            return
        warnings.filterwarnings("ignore")
        try:
            out_dir = os.path.join(os.path.dirname(csv_path), "model")
            df_imp = explain_xgboost(model_path, csv_path, out_dir=out_dir)
            txt = "【SHAP 特征贡献占比】\n\n"
            txt += df_imp.head(10).to_string()
            txt += "\n\n（summary 图已保存至 model/shap_summary.png）"
            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            shap_png = os.path.join(out_dir, "shap_summary.png")
            if os.path.isfile(shap_png):
                plt.figure(figsize=(7, 5))
                img = plt.imread(shap_png)
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                self.embed_figure(plt.gcf())
            QMessageBox.information(self, "完成", "SHAP 可解释已运行，特征贡献与图已显示。")
        except Exception as e:
            QMessageBox.critical(self, "运行出错", str(e))

    def run_spatial_topology_framework(self):
        """一键运行空间-拓扑融合学习框架：特征工程 + 融合 + XGBoost + SHAP。"""
        try:
            from spatial_topology_framework import run_spatial_topology_fusion_pipeline
        except ImportError as e:
            QMessageBox.warning(
                self,
                "模块未安装",
                f"请确保 spatial_topology_framework.py、feature_engineering、fusion_algorithm、ml 可用。\n{e}",
            )
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        # 先读取 CSV 列名，便于校验和提示
        try:
            import pandas as _pd
            _df_preview = _pd.read_csv(csv_path, nrows=1)
            _available_cols = list(_df_preview.columns)
        except Exception:
            _available_cols = []
        target_column, ok = QInputDialog.getText(
            self,
            "目标列名",
            "请输入要预测的目标列（例如 Area）：",
            text="Area",
        )
        if not ok or not target_column or not target_column.strip():
            return
        target_column = target_column.strip()
        if _available_cols and target_column not in _available_cols:
            QMessageBox.warning(
                self,
                "目标列不存在",
                f"CSV 中未找到列「{target_column}」。\n可用列：{', '.join(_available_cols[:15])}{'...' if len(_available_cols) > 15 else ''}",
            )
            return
        import warnings as _warnings

        _warnings.filterwarnings("ignore")
        try:
            res = run_spatial_topology_fusion_pipeline(
                csv_path=csv_path,
                target_column=target_column,
            )
            xgb_res = res.get("xgb_result", {})
            cv_agg = xgb_res.get("cv_agg", {})
            test_metrics = xgb_res.get("test_metrics", {})
            shap_df = res.get("shap_importance")
            txt = "【空间-拓扑融合分析】\n\n"
            txt += f"目标列：{target_column}\n"
            if cv_agg:
                txt += "\n交叉验证（CV）指标：\n"
                if "R2_mean" in cv_agg:
                    txt += f"  R² 均值：{cv_agg['R2_mean']:.4f}\n"
                if "MAE_mean" in cv_agg:
                    txt += f"  MAE 均值：{cv_agg['MAE_mean']:.4f}\n"
                if "RMSE_mean" in cv_agg:
                    txt += f"  RMSE 均值：{cv_agg['RMSE_mean']:.4f}\n"
            if test_metrics:
                txt += "\n测试集指标：\n"
                if "R2" in test_metrics:
                    txt += f"  R²：{test_metrics['R2']:.4f}\n"
                if "MAE" in test_metrics:
                    txt += f"  MAE：{test_metrics['MAE']:.4f}\n"
                if "RMSE" in test_metrics:
                    txt += f"  RMSE：{test_metrics['RMSE']:.4f}\n"
            if shap_df is not None and not shap_df.empty:
                top_k = shap_df.head(8)
                txt += "\nTop 特征贡献（SHAP）：\n"
                txt += top_k.to_string(index=False)
                txt += "\n\n（完整 SHAP 图已保存至 data/processed/shap_summary.png 或 config 中指定目录）"
            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            QMessageBox.information(self, "完成", "空间-拓扑融合分析已运行，结果已显示在左侧文本框。")
        except Exception as e:
            import traceback as _tb

            err_detail = _tb.format_exc()
            QMessageBox.critical(
                self,
                "运行出错",
                f"{str(e)}\n\n详细报错：\n{err_detail[-800:]}",
            )

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
        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        fig, ax = plt.subplots(1, 1, figsize=(9, 9 * rate))
        traces.plot(ax=ax, color="blue")
        area.boundary.plot(ax=ax, color="red")
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.set_aspect('equal')
        self.embed_figure(fig)

    def run_fenleihou(self):
        warnings.filterwarnings("ignore")
        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        fig, ax = plt.subplots(figsize=(9, 9 * rate))
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        network.branch_gdf.plot(colors=[assign_colors(bt) for bt in network.branch_types], ax=ax)
        area.boundary.plot(ax=ax, color="red")
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
        self.embed_figure(fig)

    def run_tuopuhou1(self):
        warnings.filterwarnings("ignore")
        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        fig, ax = plt.subplots(figsize=(9, 9 * rate))
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        network.trace_gdf.plot(ax=ax, linewidth=0.5)
        network.node_gdf.plot(c=[assign_colors(bt) for bt in network.node_types], ax=ax, markersize=10)
        area.boundary.plot(ax=ax, color="red")
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
        self.embed_figure(fig)

    def run_tuopuhou2(self):
        warnings.filterwarnings("ignore")
        # Drop duplicates from the trace GeoDataFrame
        traces.drop_duplicates(subset="geometry", inplace=True)
        # Reset the index of the GeoDataFrame
        traces.reset_index(drop=True, inplace=True)
        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        # 定义节点类型到颜色的映射
        type_to_color = {
            'E': 'red',  # 假设E类型节点用红色表示
            'I': 'green',  # 假设I类型节点用绿色表示
            'X': 'blue',  # 假设X类型节点用蓝色表示
            'Y': 'yellow',  # 假设Y类型节点用黄色表示
        }
        type_to_color2 = {'CC': 'red', 'CI': 'green', 'II': 'blue', }
        # 定义节点类型到形状的映射
        type_to_shape = {
            'E': 'o',  # 假设E类型节点用圆圈表示
            'I': 'o',  # 假设I类型节点用正方形表示
            'X': '^',  # 假设X类型节点用三角形表示
            'Y': '*'  # 假设Y类型节点用星号表示
        }
        # 开始绘图
        fig, ax = plt.subplots(figsize=(9, 9 * rate))

        # 遍历每个节点类型，绘制对应类型的节点
        for node_type in type_to_color.keys():
            # 选取当前节点类型的节点 Traces
            nodes = network.node_gdf[network.node_gdf['Class'] == node_type]
            # 使用相应的颜色和形状绘制节点
            ax.scatter(nodes.geometry.x, nodes.geometry.y, s=50,
                       c=type_to_color[node_type], marker=type_to_shape[node_type], label=node_type)
        for branch_type in type_to_color2.keys():
            network.trace_gdf.plot(colors=[assign_colors(bt) for bt in network.branch_types], ax=ax, linewidth=1,
                                   label=branch_type)
        area.boundary.plot(ax=ax, color="red")
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.legend(title=' Type')
        ax.set_aspect('equal')
        self.embed_figure(fig)

    def run_tuopushuxing(self):
        warnings.filterwarnings("ignore")
        network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
                          circular_target_area=False, snap_threshold=0.001, )
        parameters = 'parameters'.ljust(40, ' ') + 'values' + "\n"
        for key, value in network.parameters.items():
            parameters = parameters + str(key).ljust(40, ' ') + str(value) + "\n"
        self.text_browser.clear()
        self.text_browser.insertPlainText(parameters)
        self.text_browser.moveCursor(QTextCursor.End)

    def run_azimuth(self):
        network = Network(
            name="KB11",
            trace_gdf=traces,
            area_gdf=area,
            truncate_traces=True,
            circular_target_area=False,
            determine_branches_nodes=True,
            snap_threshold=0.001,
            azimuth_set_names=("N-S", "E-W"),
            azimuth_set_ranges=((135, 45), (45, 135)),
        )
        pprint((network.azimuth_set_names, network.azimuth_set_ranges))
        pprint(network.trace_azimuth_set_counts)
        fig, ax = plt.subplots(figsize=(9, 9 * rate))
        colors = ("red", "blue")
        assert len(colors) == len(network.azimuth_set_names)
        for azimuth_set, set_range, color in zip(network.azimuth_set_names, network.azimuth_set_ranges, colors):
            trace_gdf_set = network.trace_gdf.loc[network.trace_gdf["azimuth_set"] == azimuth_set]
            trace_gdf_set.plot(color=color, label=f"{azimuth_set} - {set_range}", ax=ax)
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        area.boundary.plot(ax=ax, color="red")
        ax.set_aspect('equal')
        plt.legend()
        self.embed_figure(fig)

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
        area.boundary.plot(ax=ax, color="red")
        ax.set_aspect('equal')
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        self.embed_figure(fig)

    def a(self):
        warnings.filterwarnings("ignore")
        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        fit, fig1, ax = network.plot_trace_lengths()
        # ax.set_aspect('equal')
        fit, fig2, ax = network.plot_branch_lengths()
        # ax.set_aspect('equal')

        self.embed_figure([fig1, fig2])

    def run_meiguitu(self):
        warnings.filterwarnings("ignore")
        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        azimuth_bin_dict, fig1, ax = network.plot_trace_azimuth()
        azimuth_bin_dict, fig2, ax = network.plot_branch_azimuth()

        self.embed_figure([fig1, fig2])

    def run_sanyuantu(self):
        warnings.filterwarnings("ignore")

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )


        fig1, ax1, tax1 = network.plot_xyi()
        ax1.axis('off')
        fig1.set_size_inches(9, 9)
        # fractopo 的三元图顶点标签和边角文字容易贴近画布边缘，统一向内收缩显示区域
        ax1.set_position([0.18, 0.16, 0.64, 0.64])
        fig1.subplots_adjust(left=0.12, right=0.90, bottom=0.12, top=0.90)

        fig2, ax2, tax2 = network.plot_branch()
        ax2.axis('off')
        fig2.set_size_inches(9, 9)
        ax2.set_position([0.18, 0.16, 0.64, 0.64])
        fig2.subplots_adjust(left=0.12, right=0.90, bottom=0.12, top=0.90)

        self.embed_figure([fig1, fig2])

    def run_guanxi(self):
        warnings.filterwarnings("ignore")
        network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
                          circular_target_area=False, snap_threshold=0.001, )
        print(f"Azimuth set names: {network.azimuth_set_names}")
        print(f"Azimuth set ranges: {network.azimuth_set_ranges}")
        figs, fig_axes = network.plot_azimuth_crosscut_abutting_relationships()
        for fig in figs:
            fig.set_size_inches(15, 7)

        if figs:
            self.embed_figure(figs)

    def b(self):
        branches, nodes = branches_and_nodes(traces, area, snap_threshold=0.001)
        fig, axes = plt.subplots(1, 2, figsize=(9, 9 * rate))
        traces.plot(ax=axes[0], color="blue", label="Traces")
        area.boundary.plot(ax=axes[0], color="black", label="Target Area", linestyle="dashed")
        axes[0].set_title("Traces & Target Area")
        nodes.plot(ax=axes[1], column="Class", zorder=10, legend=True, categorical=True, markersize=7)
        axes[1].set_title("Branches & Nodes & Area")
        area.boundary.plot(ax=axes[1], color="black", linestyle="dashed")
        axes[1].set_xlim(*axes[0].get_xlim())
        axes[1].set_ylim(*axes[0].get_ylim())
        for ax in axes:
            ax.set_xlim(left - width, right + width)
            ax.set_ylim(down - height, up + height)
            area.boundary.plot(ax=ax, color="red")
            ax.set_aspect('equal')
        legend = axes[1].get_legend()
        for handle in legend.legend_handles:
            handle._sizes = [20]
        legend.set_bbox_to_anchor((1, 0.5))
        plt.tight_layout()
        self.embed_figure(fig)

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
                cbar.set_label(param, fontname='Microsoft YaHei')

                ax.set_aspect("equal")
                ax.set_title(f"平滑热力图: {param}", fontname='Microsoft YaHei', fontsize=14, pad=15)
                plt.tight_layout()

                try:
                    self.embed_figure([fig])
                except TypeError:
                    self.embed_figure(fig)


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
            global traces
            global area

            try:
                bounds = traces.total_bounds

                if area is not None:
                    minx, miny, maxx, maxy = area.total_bounds
                    spatial_index = traces.sindex
                    possible_matches_index = list(spatial_index.intersection((minx, miny, maxx, maxy)))
                    original_len = len(traces)
                    traces = traces.iloc[possible_matches_index]

                map_span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                dp_tolerance = map_span * 0.002

                if dp_tolerance > 0:
                    traces.geometry = traces.geometry.simplify(tolerance=dp_tolerance, preserve_topology=True)

            except Exception as e:
                print(f"算法预处理跳过。原因: {e}")

            bounds = traces.total_bounds
            dynamic_width = (bounds[2] - bounds[0]) / 20.0

            if dynamic_width <= 0:
                dynamic_width = 100.0

            print(f"正在执行空间拓扑计算 (网格大小: {dynamic_width:.4f})...")
            QtWidgets.QApplication.processEvents()

            network = Network(
                traces,
                area,
                name=name,
                determine_branches_nodes=True,
                truncate_traces=True,
                circular_target_area=False,
                snap_threshold=0.001,
            )

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
        method = self.combo_fusion.currentText()
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
        csv_name = "Yingmai 2 area in Tarim Basin.csv"
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
        if not os.path.isfile(csv_path):
            QMessageBox.warning(
                self, "未找到数据",
                f"未找到：{csv_name}\n请先运行 data export.py 或将该文件放在 program 目录下。",
            )
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
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        ax1.grid(False)
        scatter = ax1.scatter(
            df_out[x_col], df_out[y_col],
            c=df_out["cluster_id"], cmap="tab10", s=15, alpha=0.8,
        )
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title(f"拓扑属性融合（{method_name}）：{x_col}–{y_col}（颜色=聚类类型）")
        plt.colorbar(scatter, ax=ax1, label="cluster_id")
        ax1.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        self.embed_figure(fig1)
        summary_lines = [
            "【智能拓扑分析结果】",
            "",
            f"融合方式：{method_name}",
            f"数据：{csv_name}",
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
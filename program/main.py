import math
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from fractopo.branches_and_nodes import branches_and_nodes

from pprint import pprint

from matplotlib.lines import Line2D

from demo import Ui_MainWindow
from fractopo.general import CC_branch, CI_branch, II_branch, X_node, Y_node, I_node
from fractopo import Network
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from PyQt5.QtGui import QTextCursor

# trace_data_url = "KB11/KB11_traces.geojson"
# area_data_url = "KB11/KB11_area.geojson"
# traces = gpd.read_file(trace_data_url)
# area = gpd.read_file(area_data_url)
# name = "KB11"

trace_data_url = "THK/thkceshi-landmark1.geojson"
area_data_url = "THK/my_area.geojson"
traces = gpd.read_file(trace_data_url)
area = gpd.read_file(area_data_url)
traces.drop_duplicates(subset="geometry", inplace=True)
traces.reset_index(drop=True, inplace=True)
name = "Yingmai 2 area in Tarim Basin"

# trace_data_url = "MY/11.geojson"
# area_data_url = "MY/my_area1.geojson"
# traces = gpd.read_file(trace_data_url)
# area = gpd.read_file(area_data_url)
# name = "MY"

geometry = traces.geometry.tolist()
left, right, down, up = math.inf, -math.inf, math.inf, -math.inf
for one in geometry:
    left = min(left, one.boundary.bounds[0])
    right = max(right, one.boundary.bounds[2])
    down = min(down, one.boundary.bounds[1])
    up = max(up, one.boundary.bounds[3])
rate = (up - down) / (right - left)
print(rate)
width, height = 0.01 * (right - left), 0.01 * (up - down)


def assign_colors(feature_type: str):
    if feature_type in (CC_branch, X_node):
        return "green"
    if feature_type in (CI_branch, Y_node):
        return "blue"
    if feature_type in (II_branch, I_node):
        return "black"
    return "red"


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_3.clicked.connect(self.run_yuantu)
        self.pushButton_2.clicked.connect(self.run_fenleihou)
        self.pushButton_8.clicked.connect(self.run_relitu)
        self.pushButton_7.clicked.connect(self.run_meiguitu)
        self.pushButton_4.clicked.connect(self.run_sanyuantu)
        self.pushButton_6.clicked.connect(self.run_guanxi)
        self.pushButton_9.clicked.connect(self.run_tuopushuxing)
        self.pushButton_10.clicked.connect(self.run_azimuth)
        self.comboBox.currentIndexChanged.connect(self.onIndexChanged)
        self.comboBox_2.currentIndexChanged.connect(self.onIndexChanged_2)
        self.pushButton.clicked.connect(self.a)
        self.pushButton_5.clicked.connect(self.b)
        self.opt = 0

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
        plt.show()

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
        fix, ax = plt.subplots(figsize=(9, 9 * rate))
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
        plt.show()

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
        fix, ax = plt.subplots(figsize=(9, 9 * rate))
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        network.trace_gdf.plot(ax=ax, linewidth=0.5)
        network.node_gdf.plot(c=[assign_colors(bt) for bt in network.node_types], ax=ax, markersize=10)
        area.boundary.plot(ax=ax, color="red")
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="green", markersize=10, label="X_node"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="blue", markersize=10, label="Y_node"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="black", markersize=10, label="I_node"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="red", markersize=10, label="Other / Boundary"),
            # plt.Line2D([0], [0], color="red", lw=2, label="Other / Boundary"),
        ]
        ax.legend(handles=handles, loc='lower left')
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.set_aspect('equal')
        plt.savefig('plot.pdf')
        plt.show()

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
        type_to_size = {
            'E': '10',  # 假设E类型节点用圆圈表示
            'I': '3',  # 假设I类型节点用正方形表示
            'X': '10',  # 假设X类型节点用三角形表示
            'Y': '10'
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
            network.trace_gdf.plot(colors=[assign_colors(bt) for bt in network.branch_types], ax=ax, linewidth=1, label=branch_type)
        area.boundary.plot(ax=ax, color="red")
        plt.xlim((left - width, right + width))
        plt.ylim((down - height, up + height))
        ax.legend(title=' Type')
        ax.set_aspect('equal')
        plt.savefig('plot1.pdf')
        plt.show()

    def run_tuopushuxing(self):
        warnings.filterwarnings("ignore")
        network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001, )
        parameters = 'parameters'.ljust(40, ' ') + 'values' + "\n"
        for key, value in network.parameters.items():
            parameters = parameters + str(key).ljust(40, ' ') + str(value) + "\n"
        self.textBrowser.clear()
        self.textBrowser.insertPlainText(parameters)
        self.textBrowser.moveCursor(QTextCursor.End)

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
        plt.savefig('plot1.pdf')
        plt.show()

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
        plt.show()

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
        fit, fig, ax = network.plot_trace_lengths()
        # ax.set_aspect('equal')
        fit, fig, ax = network.plot_branch_lengths()
        # ax.set_aspect('equal')
        plt.show()

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
        azimuth_bin_dict, fig, ax = network.plot_trace_azimuth()
        azimuth_bin_dict, fig, ax = network.plot_branch_azimuth()
        plt.show()

    def run_sanyuantu(self):
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
        fig1, ax1, tax1 = network.plot_xyi()
        fig1.set_size_inches(9, 9)
        fig1.tight_layout()
        fig2, ax2, tax2 = network.plot_branch()
        fig2.set_size_inches(9, 9)
        fig2.tight_layout()
        plt.show()

    def run_guanxi(self):
        warnings.filterwarnings("ignore")
        network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001, )
        print(f"Azimuth set names: {network.azimuth_set_names}")
        print(f"Azimuth set ranges: {network.azimuth_set_ranges}")
        figs, fig_axes = network.plot_azimuth_crosscut_abutting_relationships()
        for fig in figs:
            fig.set_size_inches(15, 7)
        plt.show()

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
        for handle in legend.legendHandles:
            handle._sizes = [20]
        legend.set_bbox_to_anchor((1, 0.5))
        plt.tight_layout()
        plt.savefig('plot1.pdf')
        plt.show()

    def run_lunkuo(self):
        warnings.filterwarnings("ignore")
        print(self.opt)
        network = Network(
            traces,
            area,
            name=name,
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        sampled_grid = network.contour_grid(cell_width=1000)
        if self.opt == 1:
            network.plot_contour(parameter="Fracture Intensity B21", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Fracture Intensity P21", sampled_grid=sampled_grid)
        elif self.opt == 2:
            network.plot_contour(parameter="Trace Min Length", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Trace Max Length", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Trace Mean Length", sampled_grid=sampled_grid)
        elif self.opt == 3:
            network.plot_contour(parameter="Dimensionless Intensity B22", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Dimensionless Intensity P22", sampled_grid=sampled_grid)
        elif self.opt == 4:
            network.plot_contour(parameter="Number of Traces (Real)", sampled_grid=sampled_grid)
        elif self.opt == 5:
            network.plot_contour(parameter="Branch Min Length", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Branch Max Length", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Branch Mean Length", sampled_grid=sampled_grid)
        elif self.opt == 6:
            network.plot_contour(parameter="Areal Frequency B20", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Areal Frequency P20", sampled_grid=sampled_grid)
        elif self.opt == 7:
            network.plot_contour(parameter="Connections per Trace", sampled_grid=sampled_grid)
            network.plot_contour(parameter="Connections per Branch", sampled_grid=sampled_grid)
        elif self.opt == 8:
            network.plot_contour(parameter="Connection Frequency", sampled_grid=sampled_grid)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

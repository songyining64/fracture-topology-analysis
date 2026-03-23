"""单张「分类+拓扑」图：分支/节点按类型着色，保存 plot13.pdf"""
import math
from fractopo.general import CC_branch, CI_branch, II_branch, X_node, Y_node, I_node
from fractopo import Network
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()


def assign_colors(feature_type: str):
    if feature_type in (CC_branch, X_node):
        return "green"
    if feature_type in (CI_branch, Y_node):
        return "blue"
    if feature_type in (II_branch, I_node):
        return "black"
    return "red"


trace_data_url = "THK/thkceshi-landmark1.geojson"
area_data_url = "THK/my_area.geojson"
traces = gpd.read_file(trace_data_url)
area = gpd.read_file(area_data_url)
name = "MY"

geometry = traces.geometry.tolist()
left, right, down, up = math.inf, -math.inf, math.inf, -math.inf
for one in geometry:
    left = min(left, one.boundary.bounds[0])
    right = max(right, one.boundary.bounds[2])
    down = min(down, one.boundary.bounds[1])
    up = max(up, one.boundary.bounds[3])
rate = (up - down) / (right - left)
width, height = 0.01 * (right - left), 0.01 * (up - down)

warnings.filterwarnings("ignore")
network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001, )
fig, ax = plt.subplots(figsize=(9, 9 * rate))
network.branch_gdf.plot(colors=[assign_colors(bt) for bt in network.branch_types], ax=ax)
network.trace_gdf.plot(ax=ax, linewidth=0.5)
network.node_gdf.plot(
    c=[assign_colors(bt) for bt in network.node_types], ax=ax, markersize=10
)
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
plt.savefig('plot13.pdf')
plt.show()

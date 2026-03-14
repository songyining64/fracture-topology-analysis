import math
import warnings
import geopandas as gpd
from fractopo import Network
from matplotlib import pyplot as plt
import time

time_start = time.time()  #程序开始时间
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
data_option = 2
if data_option == 1:
    traces = gpd.read_file("KB11/KB11_traces.geojson")
    area = gpd.read_file("KB11/KB11_area.geojson")
    name = "KB11"
elif data_option == 2:
    traces = gpd.read_file("THK/thkceshi-landmark1.geojson")
    area = gpd.read_file("THK/my_area.geojson")
    name = "THK"
elif data_option == 3:
    traces = gpd.read_file("MY/11.geojson")
    area = gpd.read_file("MY/my_area1.geojson")
    name = "MY"

network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
                  circular_target_area=False, snap_threshold=0.001, )
width = 1000
geometry = traces.geometry.tolist()
left, right, down, up = math.inf, -math.inf, math.inf, -math.inf
for one in geometry:
    left = min(left, one.boundary.bounds[0])
    right = max(right, one.boundary.bounds[2])
    down = min(down, one.boundary.bounds[1])
    up = max(up, one.boundary.bounds[3])
print('左边界:', left, '右边界:', right, '上边界:', up, '下边界:', down)
print('X轴范围:', right - left, 'Y轴范围:', up - down)
print('网格数:', int((right - left) / width), '*', int((up - down) / width), '=',
      int((right - left) / width * (up - down) / width))
sampled_grid = network.contour_grid(cell_width=width)
# parameter = "Fracture Intensity B21"
# parameter = "Fracture Intensity P21"
parameter = "Number of Traces (Real)"
# parameter = "Number of Traces"
# parameter = "Number of Branches"
# parameter = "Number of Branches (Real)"
# parameter = "Trace Min Length"
# parameter = "Trace Max Length"
# parameter = "Trace Mean Length"
# parameter = "Connections per Branch"
# parameter = "Connection Frequency"
# parameter = "Dimensionless Intensity P22"
# parameter = "Branch Max Length"
# parameter = "Areal Frequency B20"
# parameter = "Areal Frequency P20"
# parameter = "Areal Frequency B20"
# parameter = "Areal Frequency P20"
# 没什么效果的
# parameter = "Branch Mean Length"
# parameter = "Branch Min Length"
# parameter = "Dimensionless Intensity B22"
# parameter = "Connections per Trace"
network.plot_contour(parameter=parameter, sampled_grid=sampled_grid)
plt.savefig(name + '-' + str(width) + '-' + parameter + '.pdf')
plt.show()
print('用时:', time.time() - time_start, '秒')  #当前时间-程序开始时间

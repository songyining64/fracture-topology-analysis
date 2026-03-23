import warnings
import geopandas as gpd
from fractopo import Network
from fractopo.analysis.parameters import plot_parameters_plot
from matplotlib import pyplot as plt

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()
warnings.filterwarnings("ignore")

traces = gpd.read_file("KB11/KB11_traces.geojson")
area = gpd.read_file("KB11/KB11_area.geojson")
name = "KB11"
KB11 = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001, )

traces = gpd.read_file("THK/thkceshi-landmark1.geojson")
area = gpd.read_file("THK/my_area.geojson")
name = "MY"
MY = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001, )

b22 = "Dimensionless Intensity B22"
cpb = "Connections per Branch"
selected = {b22, cpb}
kb11_network_selected_params = {param: value for param, value in KB11.parameters.items() if param in selected}
kb7_network_selected_params = {param: value for param, value in MY.parameters.items() if param in selected}
figs, axes = plot_parameters_plot(topology_parameters_list=[kb11_network_selected_params, kb7_network_selected_params, ], labels=["KB11", "MY"], colors=["red", "blue"], )
plt.savefig('plot13.pdf')
plt.show()

import warnings
import matplotlib as mpl
import geopandas as gpd
import joblib
import shutil
from fractopo import Network
from fractopo.analysis import length_distributions
from fractopo import general
from matplotlib import pyplot as plt

# 清除 fractopo 的缓存
shutil.rmtree('.cache/fractopo', ignore_errors=True)

warnings.filterwarnings("ignore")
traces = gpd.read_file("KB11/KB11_traces.geojson")
area = gpd.read_file("KB11/KB11_area.geojson")
name = "KB11"
KB11 = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001)

traces = gpd.read_file("MY/11.geojson")
area = gpd.read_file("MY/my_area1.geojson")
name = "MY"
MY = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001)

traces = gpd.read_file("THK/thkceshi-landmark1.geojson")
area = gpd.read_file("THK/my_area.geojson")
name = "THK"
THK = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001)

mpl.rcParams["figure.figsize"] = (5, 5)
mpl.rcParams["font.size"] = 8

networks = [KB11, MY, THK]
distributions = [netw.trace_length_distribution(azimuth_set=None) for netw in networks]
mld = length_distributions.MultiLengthDistribution(distributions=distributions, using_branches=False, fitter=length_distributions.scikit_linear_regression)

shgo_kwargs = dict(sampling_method="sobol")
scorer = general.r2_scorer
opt_result, opt_mld = mld.optimize_cut_offs(scorer=scorer)

polyfit, fig, ax = opt_mld.plot_multi_length_distributions(automatic_cut_offs=False, scorer=scorer, plot_truncated_data=True)
print(f""" Optimized cut-offs: {opt_result.optimize_result.x}
Resulting power-law exponent: {opt_result.polyfit.m_value}
Resulting {scorer.__name__} score: {opt_result.polyfit.score} """)

plt.tight_layout()
plt.savefig('plot14.pdf')
plt.show()

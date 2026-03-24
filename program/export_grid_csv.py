"""
为指定区域生成网格拓扑 CSV，供融合/ML 使用。
用法：python export_grid_csv.py [THK|KB11|MY]  默认 THK
"""
import sys
import os
import geopandas as gpd
import pandas as pd
from fractopo import Network

# 区域配置（与 main.py DATA_SOURCES 对应）
# THK=准噶尔盆地车莫古隆起, MY=塔里木盆地英买2, KB11=柯坪断隆KB11
REGIONS = {
    "THK": {"traces": "THK/thkceshi-landmark1.geojson", "area": "THK/my_area.geojson", "name": "准噶尔盆地车莫古隆起", "csv": "THK.csv"},
    "KB11": {"traces": "KB11/KB11_traces.geojson", "area": "KB11/my_area1.geojson", "name": "柯坪断隆KB11", "csv": "KB11.csv"},
    "MY": {"traces": "MY/11.geojson", "area": "MY/my_area1.geojson", "name": "塔里木盆地英买2", "csv": "Yingmai 2 area in Tarim Basin.csv"},
}

region_key = (sys.argv[1] if len(sys.argv) > 1 else "THK").upper()
if region_key not in REGIONS:
    print(f"未知区域，请用: THK | KB11 | MY")
    sys.exit(1)

cfg = REGIONS[region_key]
base = os.path.dirname(os.path.abspath(__file__))
trace_data_url = os.path.join(base, cfg["traces"])
area_data_url = os.path.join(base, cfg["area"])
name = cfg["name"]
out_csv_name = cfg.get("csv", name + ".csv")


if not os.path.isfile(trace_data_url) or not os.path.isfile(area_data_url):
    print(f"未找到数据文件: {trace_data_url} 或 {area_data_url}")
    sys.exit(1)

traces = gpd.read_file(trace_data_url)
area = gpd.read_file(area_data_url)
traces.drop_duplicates(subset="geometry", inplace=True)
traces.reset_index(drop=True, inplace=True)

network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001)
sampled_grid = network.contour_grid(cell_width=750.0)
data = []
for index, row in sampled_grid.iterrows():
    coords = list(row['geometry'].exterior.coords)[:4]
    entry = {
        'vertex1_x': coords[0][0],
        'vertex1_y': coords[0][1],
        'vertex2_x': coords[1][0],
        'vertex2_y': coords[1][1],
        'vertex3_x': coords[2][0],
        'vertex3_y': coords[2][1],
        'vertex4_x': coords[3][0],
        'vertex4_y': coords[3][1],
    }
    entry.update(row.drop('geometry').to_dict())
    data.append(entry)

df = pd.DataFrame(data)
out_csv = os.path.join(base, out_csv_name)
df.to_csv(out_csv, index=False)
print(f"已生成: {out_csv}  (网格数: {len(df)})")

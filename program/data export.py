import geopandas as gpd
import pandas as pd
from fractopo import Network

trace_data_url = "THK/thkceshi-landmark1.geojson"
area_data_url = "THK/my_area.geojson"
traces = gpd.read_file(trace_data_url)
area = gpd.read_file(area_data_url)
traces.drop_duplicates(subset="geometry", inplace=True)
traces.reset_index(drop=True, inplace=True)
name = "Yingmai 2 area in Tarim Basin"




network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=False, snap_threshold=0.001, )
sampled_grid = network.contour_grid(cell_width=750)
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
df.to_csv(name + '.csv', index=False)

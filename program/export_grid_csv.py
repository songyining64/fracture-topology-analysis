"""
为指定区域生成网格拓扑 CSV，供融合/ML 使用。
当前仅支持英买 2：python export_grid_csv.py  或  python export_grid_csv.py MY
"""
import os
import sys
from typing import Dict, Optional

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import geopandas as gpd
import pandas as pd
from fractopo import Network

from utils.config_loader import load_config
from utils.crs_metric import unify_traces_area_crs, reproject_to_metric_crs
from utils.export_utils import export_spatial_dataframe
from utils.logging_utils import get_logger

REGIONS = {
    "MY": {
        "traces": "MY/11.geojson",
        "area": "MY/my_area1.geojson",
        "name": "塔里木盆地英买2",
        "csv": "Yingmai 2 area in Tarim Basin.csv",
    },
}


def grid_to_vertex_dataframe(sampled_grid: gpd.GeoDataFrame) -> pd.DataFrame:
    data = []
    for _, row in sampled_grid.iterrows():
        coords = list(row["geometry"].exterior.coords)[:4]
        entry = {
            "vertex1_x": coords[0][0],
            "vertex1_y": coords[0][1],
            "vertex2_x": coords[1][0],
            "vertex2_y": coords[1][1],
            "vertex3_x": coords[2][0],
            "vertex3_y": coords[2][1],
            "vertex4_x": coords[3][0],
            "vertex4_y": coords[3][1],
        }
        entry.update(row.drop("geometry").to_dict())
        data.append(entry)
    return pd.DataFrame(data)


def generate_grid_csv(region_key: str = "MY", cell_width: Optional[float] = None) -> Dict[str, Optional[str]]:
    cfg_all = load_config()
    export_cfg = (cfg_all.get("export_grid") or {}) if isinstance(cfg_all, dict) else {}
    log_cfg = (cfg_all.get("logging") or {}) if isinstance(cfg_all, dict) else {}
    logger = get_logger(
        "export_grid_csv",
        level=log_cfg.get("level", "INFO"),
        log_file=os.path.join(_BASE, log_cfg.get("file", "logs/pipeline.log")),
    )

    region_key = region_key.upper()
    if region_key not in REGIONS:
        raise ValueError("未知区域，当前仅支持: MY（英买2）")

    cfg = REGIONS[region_key]
    trace_data_url = os.path.join(_BASE, cfg["traces"])
    area_data_url = os.path.join(_BASE, cfg["area"])
    name = cfg["name"]
    out_csv_name = cfg.get("csv", name + ".csv")
    cell_width = float(cell_width or export_cfg.get("cell_width", 750.0))

    if not os.path.isfile(trace_data_url) or not os.path.isfile(area_data_url):
        raise FileNotFoundError(f"未找到数据文件: {trace_data_url} 或 {area_data_url}")

    logger.info("读取 GeoJSON：%s / %s", trace_data_url, area_data_url)
    traces = gpd.read_file(trace_data_url)
    area = gpd.read_file(area_data_url)
    traces, area = unify_traces_area_crs(traces, area)
    traces, area = reproject_to_metric_crs(traces, area)
    traces.drop_duplicates(subset="geometry", inplace=True)
    traces.reset_index(drop=True, inplace=True)

    logger.info("构建 fractopo.Network，cell_width=%s", cell_width)
    network = Network(
        traces,
        area,
        name=name,
        determine_branches_nodes=True,
        truncate_traces=True,
        circular_target_area=False,
        snap_threshold=0.001,
    )
    sampled_grid = network.contour_grid(cell_width=cell_width)
    df = grid_to_vertex_dataframe(sampled_grid)
    out_paths = export_spatial_dataframe(
        df,
        _BASE,
        os.path.splitext(out_csv_name)[0],
        export_csv=bool(export_cfg.get("out_csv", True)),
        export_gpkg=bool(export_cfg.get("out_gpkg", True)),
        layer_name=str(export_cfg.get("gpkg_layer", "grid_topology")),
    )
    logger.info("网格导出完成：rows=%s csv=%s gpkg=%s", len(df), out_paths.get("csv"), out_paths.get("gpkg"))
    return {"dataframe": df, **out_paths}


def main() -> None:
    region_key = (sys.argv[1] if len(sys.argv) > 1 else "MY").upper()
    try:
        result = generate_grid_csv(region_key)
        print(
            f"已生成网格结果：CSV={result.get('csv')} | GPKG={result.get('gpkg')} "
            f"(网格数: {len(result['dataframe'])})"
        )
    except Exception as e:
        print(f"导出失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

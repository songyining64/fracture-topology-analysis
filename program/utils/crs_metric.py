# -*- coding: utf-8 -*-
"""迹线/研究区 CRS：对齐同一坐标系，并在地理 CRS 下投到 UTM，避免 fractopo 内 .length 在度单位下告警与偏差。"""

import geopandas as gpd


def unify_traces_area_crs(traces: gpd.GeoDataFrame, area: gpd.GeoDataFrame):
    """
    将迹线与研究区对齐到同一 CRS。
    fractopo 在 crop 前不会把两套不同 CRS 自动变换到同一空间，易导致裁剪为空。
    """
    from pyproj import CRS

    tc, ac = traces.crs, area.crs
    if tc is not None and ac is not None:
        if not CRS.from_user_input(tc).equals(CRS.from_user_input(ac)):
            traces = traces.to_crs(ac)
    elif ac is not None and tc is None:
        traces = traces.set_crs(ac)
    elif tc is not None and ac is None:
        area = area.set_crs(tc)
    return traces, area


def reproject_to_metric_crs(traces: gpd.GeoDataFrame, area: gpd.GeoDataFrame):
    """
    地理坐标系（如 EPSG:4326）下 geometry.length 单位为度，geopandas/fractopo 会大量报警告且阈值含义不对。
    自动估计 UTM 并重投影（与网格 cell_width 等单位一致为米）。
    """
    try:
        if traces.crs is None:
            traces = traces.set_crs(4326)
        if area.crs is None:
            area = area.set_crs(traces.crs)
        elif not traces.crs.equals(area.crs):
            area = area.to_crs(traces.crs)
        crs = traces.crs
        if crs is not None and getattr(crs, "is_geographic", False):
            utm = traces.estimate_utm_crs()
            traces = traces.to_crs(utm)
            area = area.to_crs(utm)
    except Exception:
        pass
    return traces, area

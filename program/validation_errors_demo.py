"""拓扑验证错误示意图：展示 fractopo 会报的几何错误类型，保存 validation_errors.png"""
import logging
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from shapely.geometry import LineString, Point

fig: Figure
fig, axes = plt.subplots(2, 3, figsize=(7, 7))
fig.tight_layout(h_pad=1.5)

axes_flat: Sequence[Axes] = axes.flatten()

for ax in axes_flat:
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis("off")

default_text_kwargs = dict(ha="center", fontdict=dict(size="small"))

# Axis 1: MULTI JUNCTION
axis_1 = axes_flat[0]
traces_1 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-5, 0), (5, 0)]),
        LineString([(0, 5), (0, -5)]),
        LineString([(-2.5, 2.5), (2.5, -2.5)]),
    ]
)
errors_1 = gpd.GeoDataFrame(geometry=[Point(0, 0)])
traces_1.plot(ax=axis_1, color="black")
errors_1.plot(ax=axis_1, marker="X", color="red", zorder=10)
axis_1.text(x=0, y=-7, s="More than two traces intersect\non the same point.", **default_text_kwargs)

# Axis 2: MULTI JUNCTION
axis_2 = axes_flat[1]
traces_2 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-2.5, 2.5), (0.5, -0.5)]),
        LineString([(-5, -5), (5, 5)]),
    ]
)
traces_2.plot(ax=axis_2, color="black")
axis_2.annotate("Overlap distance higher\n than defined snap threshold.", xy=(0.5, -0.5), xytext=(-1.0, -4),
                arrowprops=dict(arrowstyle="->", color="red"), fontstyle="italic", fontsize="small")
axis_2.text(x=0, y=-7, s="Trace overlaps another trace.", **default_text_kwargs)

# Axis 3: V NODE
axis_3 = axes_flat[2]
traces_3 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-2.5, 2.5), (0, 0)]),
        LineString([(-2.5, -5), (0, 0)]),
    ]
)
errors_3 = gpd.GeoDataFrame(geometry=[Point(0, 0)])
traces_3.plot(ax=axis_3, color="black")
errors_3.plot(ax=axis_3, marker="X", color="red", zorder=10)
axis_3.text(x=0, y=-7, s="Two traces end in a\nV-node formation.", **default_text_kwargs)

# Axis 4: MULTIPLE CROSSCUTS
axis_4 = axes_flat[3]
traces_4 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-5, 0), (5, 0)]),
        LineString([(-5, 1), (-2, -1), (1, 1), (5, -1)]),
    ]
)
intersections = traces_4.geometry.values[0].intersection(traces_4.geometry.values[1])
errors_4 = gpd.GeoDataFrame(geometry=list(intersections.geoms))
traces_4.plot(ax=axis_4, color="black")
errors_4.plot(ax=axis_4, marker="X", color="red", zorder=10)
axis_4.text(x=0, y=-7, s="Two traces cross each\nother more than two times.", **default_text_kwargs)

# Axis 5: OVERLAPPING
axis_5 = axes_flat[4]
traces_5 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-5, 0), (5, 0)]),
        LineString([(-5, 1), (-1, 0), (1, 0), (5, -1)]),
    ]
)
intersections = traces_5.geometry.values[0].intersection(traces_5.geometry.values[1])
assert isinstance(intersections, LineString)
errors_5 = gpd.GeoDataFrame(geometry=[intersections])
traces_5.plot(ax=axis_5, color="black")
errors_5.plot(ax=axis_5, color="red", zorder=10)
axis_5.text(x=0, y=-7, s="Two traces overlap.", **default_text_kwargs)
axis_5.annotate("Trace continues\n along the other trace.", xy=(0.0, 0.0), xytext=(-3, 3),
                arrowprops=dict(arrowstyle="->", color="red"), fontstyle="italic", fontsize="small")

# Axis 6: OVERLAPPING
axis_6 = axes_flat[5]
traces_6 = gpd.GeoDataFrame(geometry=[LineString([(-5, -1), (-1, -1), (-2, 1), (5, 1)])])
traces_6.plot(ax=axis_6, color="red")
axis_6.text(x=0, y=-7, s="Trace is not sub-linear.", **default_text_kwargs)

plt.subplots_adjust(wspace=0.11, hspace=-0.31)

if __name__ == "__main__":
    output_name = "validation_errors.png"
    try:
        output_path = Path(__file__).parent / output_name
        fig.savefig(output_path, bbox_inches="tight")
        print(f"已保存: {output_path}")
    except Exception as e:
        logging.info(f"Failed to save {output_name} plot.", exc_info=True)
    plt.show()

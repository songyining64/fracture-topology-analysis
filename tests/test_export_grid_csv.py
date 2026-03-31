import os
import sys
import unittest

import geopandas as gpd
from shapely.geometry import Polygon

PROGRAM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "program")
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

from export_grid_csv import grid_to_vertex_dataframe


class ExportGridCsvTest(unittest.TestCase):
    def test_grid_to_vertex_dataframe_extracts_polygon_vertices(self):
        gdf = gpd.GeoDataFrame(
            {
                "Connection Frequency": [0.5],
                "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            },
            geometry="geometry",
            crs=None,
        )
        df = grid_to_vertex_dataframe(gdf)
        self.assertIn("vertex1_x", df.columns)
        self.assertIn("vertex4_y", df.columns)
        self.assertEqual(df.loc[0, "vertex1_x"], 0)
        self.assertEqual(df.loc[0, "vertex3_y"], 1)
        self.assertEqual(df.loc[0, "Connection Frequency"], 0.5)


if __name__ == "__main__":
    unittest.main()

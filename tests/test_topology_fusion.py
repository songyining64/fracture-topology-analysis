import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROGRAM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "program")
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

from topology_fusion import load_and_prepare, interpret_clusters


class TopologyFusionTest(unittest.TestCase):
    def test_load_and_prepare_and_interpret_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "fusion.csv")
            df = pd.DataFrame(
                {
                    "Fracture Intensity B21": [1.0, 1.1, 3.0, 3.1, 5.0, np.nan],
                    "Connections per Branch": [0.1, 0.2, 0.8, 0.75, 1.2, 1.3],
                    "Connections per Trace": [0.2, 0.25, 0.9, 0.85, 1.1, 1.15],
                }
            )
            df.to_csv(csv_path, index=False)
            df_out, X, used_cols = load_and_prepare(
                csv_path,
                feature_columns=[
                    "Fracture Intensity B21",
                    "Connections per Branch",
                    "Connections per Trace",
                ],
            )
            self.assertEqual(len(df_out), 6)
            self.assertEqual(X.shape, (6, 3))
            self.assertIn("Fracture Intensity B21", used_cols)
            labels = np.array([0, 0, 1, 1, 1, 1])
            cluster_means = interpret_clusters(df_out, used_cols, labels, n_clusters=2)
            self.assertEqual(cluster_means.shape[0], 2)
            self.assertIn("Connections per Trace", cluster_means.columns)


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest

import pandas as pd

PROGRAM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "program")
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS


class FeatureEngineeringTest(unittest.TestCase):
    def test_build_feature_matrix_filters_target_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sample.csv")
            data = {col: [1.0, 2.0, 3.0, 4.0] for col in DEFAULT_FEATURE_COLUMNS[:6]}
            data["Connections per Branch"] = [0.1, 0.2, 0.3, 0.4]
            data["Fracture Intensity B21"] = [10.0, 11.0, 12.0, 13.0]
            pd.DataFrame(data).to_csv(csv_path, index=False)
            result = build_feature_matrix(
                csv_path,
                target_column="Fracture Intensity B21",
                n_select_mi=None,
                out_processed_dir=None,
            )
            self.assertEqual(result["X"].shape[0], 4)
            self.assertGreaterEqual(result["X"].shape[1], 1)
            self.assertNotIn("Fracture Intensity B21", result["feature_names"])


if __name__ == "__main__":
    unittest.main()

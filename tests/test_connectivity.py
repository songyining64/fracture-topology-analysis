import os
import sys
import tempfile
import unittest

import pandas as pd

PROGRAM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "program")
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

from feature_engineering import (
    CONNECTIVITY_FEATURE_COLUMNS,
    is_connectivity_feature,
    suggest_regression_target_columns,
)
from ml.explain import connectivity_shap_breakdown


class ConnectivityHelpersTest(unittest.TestCase):
    def test_connectivity_feature_columns_match_high_value_triplet(self):
        self.assertEqual(
            set(CONNECTIVITY_FEATURE_COLUMNS),
            {
                "Connections per Branch",
                "Connections per Trace",
                "Connection Frequency",
            },
        )

    def test_is_connectivity_feature(self):
        self.assertTrue(is_connectivity_feature("Connections per Branch"))
        self.assertFalse(is_connectivity_feature("Fracture Intensity B21"))

    def test_connectivity_shap_breakdown(self):
        df = pd.DataFrame(
            {
                "feature": ["Connections per Branch", "Foo"],
                "importance": [0.6, 0.4],
                "contribution_pct": [60.0, 40.0],
            }
        )
        sub, cum = connectivity_shap_breakdown(df)
        self.assertEqual(len(sub), 1)
        self.assertAlmostEqual(cum, 60.0)

    def test_suggest_regression_target_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "g.csv")
            pd.DataFrame(
                {
                    "vertex1_x": [0, 1],
                    "Fracture Intensity B21": [1.0, 2.0],
                    "const": [1.0, 1.0],
                    "good": [3.0, 5.0],
                }
            ).to_csv(path, index=False)
            s = suggest_regression_target_columns(path, max_suggestions=10)
            self.assertIn("good", s)
            self.assertNotIn("vertex1_x", s)


if __name__ == "__main__":
    unittest.main()

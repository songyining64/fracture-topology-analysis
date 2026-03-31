import os
import sys
import unittest

PROGRAM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "program")
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

from utils.config_validation import validate_config


class ConfigValidationTest(unittest.TestCase):
    def test_validate_config_ok(self):
        errs = validate_config(
            {
                "export_grid": {"cell_width": 750.0},
                "clustering": {"n_clusters": 4, "k_search_min": 2, "k_search_max": 12},
                "train": {
                    "test_size": 0.1,
                    "conformal_alpha": 0.1,
                    "spatial_cv_blocks": 9,
                    "target_column": "Fracture Intensity B21",
                    "stability_seeds": [1, 2, 3],
                },
            }
        )
        self.assertEqual(errs, [])

    def test_validate_config_bad(self):
        errs = validate_config(
            {
                "export_grid": {"cell_width": -1},
                "clustering": {"n_clusters": 1, "k_search_min": 6, "k_search_max": 4},
                "train": {"test_size": 0.8, "conformal_alpha": 2, "spatial_cv_blocks": 1},
            }
        )
        self.assertGreaterEqual(len(errs), 3)


if __name__ == "__main__":
    unittest.main()


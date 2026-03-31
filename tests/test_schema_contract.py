import os
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
PROGRAM_DIR = os.path.join(ROOT, "program")
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

from feature_engineering import DEFAULT_FEATURE_COLUMNS
from utils.validation import check_csv_has_columns
from utils.export_utils import VERTEX_COLUMNS


class SchemaContractTest(unittest.TestCase):
    def test_grid_csv_contract_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "grid.csv")
            cols = list(DEFAULT_FEATURE_COLUMNS[:4]) + list(VERTEX_COLUMNS)
            pd.DataFrame([{c: 1.0 for c in cols}]).to_csv(csv_path, index=False)
            df = check_csv_has_columns(
                csv_path,
                required=list(VERTEX_COLUMNS),
                at_least_one=list(DEFAULT_FEATURE_COLUMNS),
            )
            self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()


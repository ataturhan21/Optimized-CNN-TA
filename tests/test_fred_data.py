"""Check FRED post-processing offline with a fake reader."""
import ast
from pathlib import Path
import types
import unittest
import warnings

import numpy as np
import pandas as pd


def fred_function(frame):
    path = Path(__file__).resolve().parents[1] / "src/create_data.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name == "fetch_fred_data")
             or (isinstance(node, ast.Assign)
                 and any(getattr(t, "id", None) == "fred_codes" for t in node.targets))]
    reader = lambda symbols, start, end: types.SimpleNamespace(read=lambda: frame.copy())
    namespace = {"pd": pd, "pdr": types.SimpleNamespace(fred=types.SimpleNamespace(FredReader=reader))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["fetch_fred_data"], namespace["fred_codes"]


class FredDataTests(unittest.TestCase):
    def test_missing_values_are_forward_filled_without_deprecated_api(self):
        index = pd.date_range("2020-01-01", periods=3)
        _, codes = fred_function(pd.DataFrame())
        frame = pd.DataFrame({code: [1.0, np.nan, 3.0] for code in codes.values()}, index=index)
        frame["DGS10"] = [4.0, np.nan, 5.0]
        frame["DGS2"] = [1.0, 2.0, np.nan]
        fetch, codes = fred_function(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = fetch("2020-01-01", "2020-01-03")
        names = {v: k for k, v in codes.items()}
        np.testing.assert_allclose(result[names["DGS10"]], [4.0, 4.0, 5.0])
        np.testing.assert_allclose(result[names["DGS2"]], [1.0, 2.0, 2.0])
        np.testing.assert_allclose(result["10-2 Year Yield Difference"], [3.0, 2.0, 3.0])
        self.assertFalse(result.isna().any().any())


if __name__ == "__main__":
    unittest.main()

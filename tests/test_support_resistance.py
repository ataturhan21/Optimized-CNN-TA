"""Run support/resistance clustering without importing TA-Lib or TensorFlow."""
import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def strategy_function(name):
    path = Path(__file__).resolve().parents[1] / "src/create_strategy.py"
    node = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np, "pd": pd, "AgglomerativeClustering": AgglomerativeClustering,
                 "set_random_seed": lambda seed: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class SupportResistanceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1, 200))
        self.ohlcv = pd.DataFrame(
            {"Open": close, "High": close + 1, "Low": close - 1, "Close": close},
            index=pd.Index(pd.date_range("2020-01-01", periods=200), name="Date"),
        )

    def test_levels_are_clustered_with_current_scikit_learn(self):
        calculate = strategy_function("calculate_support_resistance")
        original = self.ohlcv.copy(deep=True)
        supports = calculate(self.ohlcv, 20, 4, "support")
        resistances = calculate(self.ohlcv, 20, 4, "resistance")
        self.assertEqual(len(supports), 4)
        self.assertEqual(len(resistances), 4)
        self.assertTrue(np.all(np.isfinite(supports)))
        self.assertGreater(resistances.max(), supports.min())
        pd.testing.assert_frame_equal(self.ohlcv, original)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from src.feaure_engineering import scaling


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {"price": [1.0, 2.0, 6.0], "Label": [0, 1, 2], "volume": [2, 4, 8]},
            index=pd.date_range("2023-01-01", periods=3),
        )

    def test_training_without_sell_class_preserves_all_labels(self):
        result = scaling(self.data, split_index=2)
        pd.testing.assert_series_equal(result["Label"], self.data["Label"])

    def test_holdout_does_not_set_scaling_parameters(self):
        result = scaling(self.data, split_index=2)
        np.testing.assert_allclose(result["price"], [0.5, 1.0, 3.0])
        np.testing.assert_allclose(result["volume"], [0.5, 1.0, 2.0])

    def test_input_index_and_column_order_are_preserved(self):
        original = self.data.copy(deep=True)
        result = scaling(self.data, split_index=2)
        pd.testing.assert_frame_equal(self.data, original)
        self.assertTrue(result.index.equals(original.index))
        self.assertEqual(result.columns.tolist(), original.columns.tolist())

    def test_unknown_method_has_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported scaling method"):
            scaling(self.data, split_index=2, method="unknown")


if __name__ == "__main__":
    unittest.main()

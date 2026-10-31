"""Test data-window construction without importing optional market/ML integrations."""
import ast
from pathlib import Path
import types
import unittest

import numpy as np
import pandas as pd


def window_function():
    path = Path(__file__).resolve().parents[1] / "src/create_data.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "create_2d_data")
    namespace = {"np": np, "tf": types.SimpleNamespace(keras=types.SimpleNamespace(
        utils=types.SimpleNamespace(to_categorical=lambda labels, num_classes:
                                   np.eye(num_classes)[np.asarray(labels, dtype=int)])))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["create_2d_data"]


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.create = window_function()
        self.frame = pd.DataFrame({"price": [10, 20, 30], "Label": [2, 0, 1],
                                   "volume": [100, 200, 300]}, index=[7, 9, 12])

    def test_middle_label_column_is_not_replaced_by_last_feature(self):
        original = self.frame.copy(deep=True)
        x_train, y_train, x_test, y_test = self.create([(self.frame, self.frame)], 2)[0]
        self.assertEqual(x_train.shape, (2, 2, 2, 1))
        np.testing.assert_array_equal(x_train[0, :, :, 0], [[10, 100], [20, 200]])
        np.testing.assert_array_equal(y_train, [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(y_test, [0, 1])
        np.testing.assert_array_equal(x_test, x_train)
        pd.testing.assert_frame_equal(self.frame, original)

    def test_each_window_uses_its_last_rows_label(self):
        frame = self.frame[["Label", "price", "volume"]]
        _, y_train, _, y_test = self.create([(frame, frame)], 2)[0]
        np.testing.assert_array_equal(y_train.argmax(axis=1), frame["Label"].iloc[1:])
        np.testing.assert_array_equal(y_test, frame["Label"].iloc[1:])

    def test_mismatched_features_and_short_splits_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "same order"):
            self.create([(self.frame, self.frame[["volume", "price", "Label"]])], 2)
        with self.assertRaisesRegex(ValueError, "at least length rows"):
            self.create([(self.frame.iloc[:1], self.frame)], 2)
        with self.assertRaisesRegex(ValueError, "length feature columns"):
            self.create([(self.frame, self.frame)], 3)

    def test_invalid_window_length_and_labels_fail_explicitly(self):
        for length in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.create([(self.frame, self.frame)], length)
        self.frame.loc[9, "Label"] = 3
        with self.assertRaisesRegex(ValueError, "Labels must use"):
            self.create([(self.frame, self.frame)], 2)


if __name__ == "__main__":
    unittest.main()

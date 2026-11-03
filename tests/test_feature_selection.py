import unittest

import numpy as np
import pandas as pd

from src.feaure_engineering import feature_selection


class FeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        labels = np.array([0, 1, 2] * 10)
        self.data = pd.DataFrame({
            "signal": labels * 10.0 + np.tile([0.0, 0.1, 0.2], 10),
            "noise": np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 6),
            "Label": labels,
        })

    def test_anova_keeps_the_most_informative_features(self):
        self.assertEqual(feature_selection(self.data, selected_feature_count=1), ["signal"])

    def test_unknown_method_has_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported feature selection method"):
            feature_selection(self.data, selected_feature_count=1, method="unknown")


if __name__ == "__main__":
    unittest.main()

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MaxAbsScaler


def feature_selection(
    data, selected_feature_count: int = 30, method: str = "ANOVA", seed=42
):
    np.random.seed(seed)
    data_after_selection = data.copy(deep=True)
    if method == "ANOVA":
        select = SelectKBest(score_func=f_classif, k=selected_feature_count)
    fitted = select.fit(
        data_after_selection.drop(columns=["Label"], axis=1),
        data_after_selection["Label"],
    )

    selected_features_boolean = select.get_support()
    features = list(data_after_selection.drop(columns=["Label"]).columns)
    selected_features = [
        features[j] for j in range(len(features)) if selected_features_boolean[j]
    ]
    return selected_features


"""
def transform():
    fitted = select.fit(
        data_after_selection.drop(columns=["Label"]),
        data_after_selection["Label"],
    )
    train_features = fitted.transform(data_after_selection[i][0].iloc[:, :-1])
    test_features = fitted.transform(data_after_selection[i][1].iloc[:, :-1])
        train_label = data_after_selection[i][0].Label
    test_label = data_after_selection[i][1].Label

    data_after_selection[i][0] = pd.DataFrame(
        data=train_features.astype("float32"),
        columns=selected_features,
        index=data_after_selection[i][0].index,
    )
    data_after_selection[i][0]["Label"] = train_label
    data_after_selection[i][1] = pd.DataFrame(
        data=test_features.astype("float32"),
        columns=selected_features,
        index=data_after_selection[i][1].index,
    )
    data_after_selection[i][1]["Label"] = test_label
"""


def scaling(data, split_index: int, method: str = "MaxAbs", seed=42):
    """Fit feature scaling on the training period while preserving class labels."""
    np.random.seed(seed)
    if method != "MaxAbs":
        raise ValueError(f"Unsupported scaling method: {method}")

    features = data.drop(columns=["Label"])
    scaler = MaxAbsScaler()
    scaler.fit(features.iloc[:split_index])
    scaled = pd.DataFrame(
        scaler.transform(features), columns=features.columns, index=data.index
    )
    scaled["Label"] = data["Label"]
    return scaled.reindex(columns=data.columns)


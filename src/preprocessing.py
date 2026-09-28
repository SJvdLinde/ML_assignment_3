"""
Loads data and does data cleaning and pre processing
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import COVERTYPE_SUBSAMPLE_SIZE, RAW_DATA_DIR, SEED


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    continuous_cols: list
    class_names: list

    @property
    def n_classes(self):
        return len(self.class_names)


def encode_labels(raw_labels):

    original_labels = sorted(pd.unique(raw_labels))
    label_to_index = {label: i for i, label in enumerate(original_labels)}
    y = np.array([label_to_index[label] for label in raw_labels], dtype=int)
    return y, [str(label) for label in original_labels]


def load_wall_robot():
    robot_df = pd.read_csv(RAW_DATA_DIR / "wall_robot.csv")
    sensor_readings = robot_df.drop(columns="class")

    # nothing to clean. all 24 sensors are distances in metres
    all_cols = list(range(sensor_readings.shape[1]))
    return sensor_readings, robot_df["class"], all_cols


def load_cardiotocography():
    ctg_df = pd.read_csv(RAW_DATA_DIR / "cardiotocography.csv")

    leaky_cols = [f"V{i}" for i in range(26, 36)]

    not_useful_cols = ["V1", "V2", "V3", "V5"]
    ctg_df = ctg_df.drop(columns=leaky_cols + not_useful_cols)

    # remove duplicates
    rows_before = len(ctg_df)
    ctg_df = ctg_df.drop_duplicates().reset_index(drop=True)
    assert rows_before - len(ctg_df) == 13, "expected exactly 13 duplicate rows"

    ctg_features = ctg_df.drop(columns="class")

    # heavy right skew on these counts
    for skewed_col in ["V7", "V15", "V20", "V24"]:
        ctg_features[skewed_col] = np.log1p(ctg_features[skewed_col])

    # all 21 are measurements so all get standardised
    all_cols = list(range(ctg_features.shape[1]))
    return ctg_features, ctg_df["class"], all_cols


def load_covertype():
    full_cov_df = pd.read_csv(RAW_DATA_DIR / "covertype.csv.gz")

    # take a 20k subsample
    cov_df, _ = train_test_split(
        full_cov_df,
        train_size=COVERTYPE_SUBSAMPLE_SIZE,
        stratify=full_cov_df["class"],
        random_state=SEED,
    )
    cov_df = cov_df.reset_index(drop=True)
    cov_features = cov_df.drop(columns="class")

    constant_cols = [col for col in cov_features.columns if cov_features[col].nunique() == 1]
    cov_features = cov_features.drop(columns=constant_cols)

    continuous_names = list(full_cov_df.columns[:10])
    continuous_cols = [cov_features.columns.get_loc(col) for col in continuous_names]
    return cov_features, cov_df["class"], continuous_cols


LOADERS = {
    "wall_robot": load_wall_robot,
    "cardiotocography": load_cardiotocography,
    "covertype": load_covertype,
}


def load_dataset(name):
    features_df, raw_labels, continuous_cols = LOADERS[name]()
    y, class_names = encode_labels(raw_labels.to_numpy())

    return Dataset(
        name=name,
        X=features_df.to_numpy(dtype=float),
        y=y,
        feature_names=list(features_df.columns),
        continuous_cols=continuous_cols,
        class_names=class_names,
    )

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from config import N_FOLDS, N_REPEATS, SEED, VALIDATION_FRACTION


@dataclass
class Split:
    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def run_name(self):

        return f"r{self.repeat + 1}_f{self.fold + 1}"

    def make_rng(self):

        return np.random.default_rng([SEED, self.repeat, self.fold])


def make_splits(y, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=SEED):
    outer_cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    all_splits = []

    for split_number, (training_part, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(split_number, n_folds)

        train_idx, val_idx = train_test_split(
            training_part,
            test_size=VALIDATION_FRACTION,
            stratify=y[training_part],
            random_state=seed + 1000 * repeat + fold,
        )
        all_splits.append(Split(repeat, fold, np.sort(train_idx), np.sort(val_idx), test_idx))

    return all_splits


def scale_split(dataset, split):
    """Standardise the continuous columns, fitting the scaler on the train rows only."""
    X_train = dataset.X[split.train_idx].copy()
    X_val = dataset.X[split.val_idx].copy()
    X_test = dataset.X[split.test_idx].copy()

    cols = dataset.continuous_cols
    scaler = StandardScaler().fit(X_train[:, cols])
    X_train[:, cols] = scaler.transform(X_train[:, cols])
    X_val[:, cols] = scaler.transform(X_val[:, cols])
    X_test[:, cols] = scaler.transform(X_test[:, cols])

    return (X_train, dataset.y[split.train_idx],
            X_val, dataset.y[split.val_idx],
            X_test, dataset.y[split.test_idx])

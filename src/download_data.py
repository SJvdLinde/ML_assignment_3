"""
Run this from the project root to download the raw data sets
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def save(name, X, y, compress=False):
    df = pd.DataFrame(X).copy()
    df["class"] = pd.Series(y).to_numpy()
    path = RAW_DIR / (f"{name}.csv.gz" if compress else f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"{name}: {df.shape[0]} rows, {df.shape[1] - 1} features -> {path.name}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    robot = fetch_openml(data_id=1497, as_frame=True, parser="auto")
    save("wall_robot", robot.data, robot.target)

    ctg = fetch_openml(data_id=1466, as_frame=True, parser="auto")
    save("cardiotocography", ctg.data, ctg.target)

    # compressed to keep the file enough
    cov = fetch_covtype(as_frame=True)
    save("covertype", cov.data, cov.target, compress=True)


if __name__ == "__main__":
    main()

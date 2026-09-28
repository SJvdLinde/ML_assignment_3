import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import config
from network import FeedForwardNet
from preprocessing import load_dataset
from splits import make_splits, scale_split
from training import macro_f1, train_with_early_stopping

TUNING_DIR = config.RESULTS_DIR / "tuning"

def train_one_size(dataset, split, n_hidden):
    X_train, y_train, X_val, y_val, _, _ = scale_split(dataset, split)
    rng = split.make_rng()
    net = FeedForwardNet(X_train.shape[1], n_hidden, dataset.n_classes, rng, config.ACTIVATION)
    history, run_info = train_with_early_stopping(net, X_train, y_train, X_val, y_val, rng)

    labels = np.arange(dataset.n_classes)
    return {
        "n_hidden": n_hidden,
        "fold": split.fold + 1,
        "train_macro_f1": macro_f1(y_train, net.predict(X_train), labels),
        "val_macro_f1": macro_f1(y_val, net.predict(X_val), labels),
        **run_info,
    }

def pick_size_one_se_rule(per_size):
    # per_size has columns n_hidden, val_mean, val_se
    best_row = per_size.loc[per_size["val_mean"].idxmax()]
    good_enough = best_row["val_mean"] - best_row["val_se"]
    candidates = per_size[per_size["val_mean"] >= good_enough]
    return int(candidates["n_hidden"].min()), int(best_row["n_hidden"])

def plot_validation_curve(dataset_name, per_size, chosen_size):
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    ax.errorbar(per_size["n_hidden"], per_size["train_mean"], yerr=per_size["train_se"],
                label="Train", color="#3b6ea5", linestyle="--", marker="s", markersize=3, capsize=2)
    ax.errorbar(per_size["n_hidden"], per_size["val_mean"], yerr=per_size["val_se"],
                label="Validation", color="#c0504d", marker="o", markersize=3, capsize=2)
    ax.axvline(chosen_size, color="grey", linestyle=":", linewidth=1)

    ax.set_xscale("log", base=2)
    ax.set_xticks(per_size["n_hidden"])
    ax.set_xticklabels(per_size["n_hidden"])
    ax.set_xlabel("Hidden neurons")
    ax.set_ylabel("Macro F1")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # 300 dpi so it stays sharp in the report
    fig.savefig(config.FIGURES_DIR / f"validation_curve_{dataset_name}.png", dpi=300)
    plt.close(fig)


def main(skip_training=False):
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    chosen_sizes = {}

    for dataset_name in config.DATASET_NAMES:
        tuning_csv = TUNING_DIR / f"{dataset_name}_tuning.csv"

        if skip_training:

            tuning_df = pd.read_csv(tuning_csv)
            print(f"\n{dataset_name}: using saved {tuning_csv.name}")
        else:
            dataset = load_dataset(dataset_name)
            repeat_one_splits = [s for s in make_splits(dataset.y) if s.repeat == 0]
            print(f"\n{dataset_name}: {len(config.HIDDEN_SIZES_TO_TRY)} sizes x {len(repeat_one_splits)} folds")

            jobs = [delayed(train_one_size)(dataset, split, h)
                    for h in config.HIDDEN_SIZES_TO_TRY for split in repeat_one_splits]
            fold_results = Parallel(n_jobs=config.N_WORKERS)(jobs)

            tuning_df = pd.DataFrame(fold_results).sort_values(["n_hidden", "fold"])
            tuning_df.to_csv(tuning_csv, index=False)

        # mean and standard error per size
        grouped = tuning_df.groupby("n_hidden")
        n_folds = grouped.size()
        per_size = pd.DataFrame({
            "train_mean": grouped["train_macro_f1"].mean(),
            "train_se": grouped["train_macro_f1"].std() / np.sqrt(n_folds),
            "val_mean": grouped["val_macro_f1"].mean(),
            "val_se": grouped["val_macro_f1"].std() / np.sqrt(n_folds),
            "mean_epochs": grouped["epochs_trained"].mean(),
        }).reset_index()
        print(per_size.round(3).to_string(index=False))

        rule_pick, best = pick_size_one_se_rule(per_size)
        chosen = config.HIDDEN_SIZE_OVERRIDES.get(dataset_name, rule_pick)
        chosen_sizes[dataset_name] = {
            "chosen_n_hidden": chosen,
            "one_se_rule_n_hidden": rule_pick,
            "overridden": chosen != rule_pick,
            "best_mean_n_hidden": best,
            "chosen_val_macro_f1": float(per_size.loc[per_size["n_hidden"] == chosen, "val_mean"].iloc[0]),
            "best_val_macro_f1": float(per_size["val_mean"].max()),
        }
        print(f"-> best mean at h={best}, one-SE rule picks h={rule_pick}, using h={chosen}")

        plot_validation_curve(dataset_name, per_size, chosen)

    with open(TUNING_DIR / "chosen_hidden_sizes.json", "w") as f:
        json.dump(chosen_sizes, f, indent=2)


if __name__ == "__main__":
    import sys
    main(skip_training="--skip-training" in sys.argv)

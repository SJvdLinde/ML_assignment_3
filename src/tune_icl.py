import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from icl import predict_proba_real_order, train_icl
from preprocessing import load_dataset
from splits import make_splits, scale_split
from training import macro_f1

TUNING_DIR = config.RESULTS_DIR / "tuning"
CHOSEN_K_FILE = TUNING_DIR / "chosen_icl_k.json"


def run_one_k(dataset, split, k):
    X_train, y_train, X_val, y_val, _, _ = scale_split(dataset, split)
    rng = split.make_rng()

    start = time.perf_counter()
    net, class_order, history, _ = train_icl(X_train, y_train, X_val, y_val, dataset.n_classes, rng, k)
    seconds = time.perf_counter() - start

    labels = np.arange(dataset.n_classes)
    train_preds = predict_proba_real_order(net, class_order, X_train).argmax(axis=1)
    val_preds = predict_proba_real_order(net, class_order, X_val).argmax(axis=1)
    return {
        "k": k,
        "fold": split.fold + 1,
        "train_macro_f1": macro_f1(y_train, train_preds, labels),
        "val_macro_f1": macro_f1(y_val, val_preds, labels),
        "n_hidden": net.n_hidden,
        "epochs_trained": len(history),
        "train_seconds": seconds,
    }


def plot_k_curve(dataset_name, per_k, chosen_k):
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    ax.errorbar(per_k["k"], per_k["train_mean"], yerr=per_k["train_se"], label="Train",
                color="#3b6ea5", linestyle="--", marker="s", markersize=3, capsize=2)
    ax.errorbar(per_k["k"], per_k["val_mean"], yerr=per_k["val_se"], label="Validation",
                color="#c0504d", marker="o", markersize=3, capsize=2)
    ax.axvline(chosen_k, color="grey", linestyle=":", linewidth=1)

    ax.set_xticks(per_k["k"])
    ax.set_xlabel("Neurons allowed without improvement")
    ax.set_ylabel("Macro F1")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"icl_k_curve_{dataset_name}.png", dpi=300)
    plt.close(fig)

def main(dataset_names):
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    # keep choices for datasets
    chosen = json.loads(CHOSEN_K_FILE.read_text()) if CHOSEN_K_FILE.exists() else {}

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name)
        repeat_one_splits = [s for s in make_splits(dataset.y) if s.repeat == 0]
        tuning_csv = TUNING_DIR / f"{dataset_name}_icl_tuning.csv"

        earlier_results = pd.read_csv(tuning_csv) if tuning_csv.exists() else pd.DataFrame(columns=["k", "fold"])
        already_done = set(zip(earlier_results["k"], earlier_results["fold"]))
        jobs = [delayed(run_one_k)(dataset, split, k)
                for k in config.ICL_FAILED_ADDITIONS_TO_TRY for split in repeat_one_splits
                if (k, split.fold + 1) not in already_done]
        print(f"\n{dataset_name}: {len(jobs)} new ICL runs ({len(already_done)} reused)", flush=True)

        start = time.perf_counter()
        fold_results = []
        for row in Parallel(n_jobs=config.N_WORKERS, return_as="generator_unordered")(jobs):
            fold_results.append(row)
            print(f"  [{len(fold_results):2d}/{len(jobs)}] {(time.perf_counter() - start) / 60:5.1f} min | "
                  f"k={row['k']} fold {row['fold']}: val F1={row['val_macro_f1']:.3f}, h={row['n_hidden']}, "
                  f"{row['train_seconds'] / 60:.1f} min", flush=True)

        tuning_df = pd.concat([earlier_results, pd.DataFrame(fold_results)], ignore_index=True)
        tuning_df = tuning_df[tuning_df["k"].isin(config.ICL_FAILED_ADDITIONS_TO_TRY)].sort_values(["k", "fold"])
        tuning_df.to_csv(tuning_csv, index=False)

        grouped = tuning_df.groupby("k")
        n_folds = grouped.size()
        per_k = pd.DataFrame({
            "train_mean": grouped["train_macro_f1"].mean(),
            "train_se": grouped["train_macro_f1"].std() / np.sqrt(n_folds),
            "val_mean": grouped["val_macro_f1"].mean(),
            "val_se": grouped["val_macro_f1"].std() / np.sqrt(n_folds),
            "mean_hidden": grouped["n_hidden"].mean(),
            "mean_minutes": grouped["train_seconds"].mean() / 60,
        }).reset_index()
        print(per_k.round(3).to_string(index=False))

        best_row = per_k.loc[per_k["val_mean"].idxmax()]
        good_enough = best_row["val_mean"] - best_row["val_se"]
        chosen_k = int(per_k.loc[per_k["val_mean"] >= good_enough, "k"].min())
        chosen[dataset_name] = {
            "chosen_k": chosen_k,
            "best_mean_k": int(best_row["k"]),
            "chosen_val_macro_f1": float(per_k.loc[per_k["k"] == chosen_k, "val_mean"].iloc[0]),
            "best_val_macro_f1": float(best_row["val_mean"]),
        }
        print(f"-> best mean at k={int(best_row['k'])}, one-SE rule picks k={chosen_k}", flush=True)
        plot_k_curve(dataset_name, per_k, chosen_k)

        CHOSEN_K_FILE.write_text(json.dumps(chosen, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:] or config.DATASET_NAMES)

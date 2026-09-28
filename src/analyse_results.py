import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score

import config
from preprocessing import load_dataset
from training import macro_f1

ANALYSIS_DIR = config.RESULTS_DIR / "analysis"
METHODS = ["baseline", "icl"]
METHOD_LABELS = {"baseline": "Baseline", "icl": "ICL"}
METHOD_COLOURS = {"baseline": "#3b6ea5", "icl": "#c0504d"}
DATASET_LABELS = {"wall_robot": "Wall robot", "cardiotocography": "Cardiotocography", "covertype": "Covertype"}
ALPHA = 0.05

plt.rcParams.update({"font.size": 8, "font.family": "serif"})

def load_all_runs(dataset):
    labels = np.arange(dataset.n_classes)
    # rarest half of the classes
    counts = np.bincount(dataset.y, minlength=dataset.n_classes)
    rarest_first = np.argsort(counts, kind="stable")
    minority_classes = rarest_first[:dataset.n_classes // 2]

    run_rows = []
    confusion_totals = {method: np.zeros((dataset.n_classes, dataset.n_classes), dtype=int) for method in METHODS}

    for method in METHODS:
        run_dirs = sorted((config.RESULTS_DIR / dataset.name / method).glob("r*_f*"))
        assert len(run_dirs) == config.N_FOLDS * config.N_REPEATS, f"missing runs for {dataset.name}/{method}"

        for run_dir in run_dirs:
            summary = json.loads((run_dir / "summary.json").read_text())
            preds = pd.read_csv(run_dir / "predictions.csv")
            y_true, y_pred = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()

            per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            row = {
                "method": method,
                "run": run_dir.name,
                "macro_f1": macro_f1(y_true, y_pred, labels),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "accuracy": accuracy_score(y_true, y_pred),
                "minority_recall": per_class_recall[minority_classes].mean(),
                "train_macro_f1": summary["train_macro_f1"],
                "n_hidden": summary["n_hidden"],
                "n_weights": summary["n_weights"],
                "epochs": summary["epochs_trained"],
                "seconds": summary["train_seconds"],
            }
            for c in labels:
                row[f"recall_{dataset.class_names[c]}"] = per_class_recall[c]
            run_rows.append(row)

            confusion_totals[method] += confusion_matrix(y_true, y_pred, labels=labels)

    return pd.DataFrame(run_rows), confusion_totals, rarest_first, minority_classes, counts

# stats

def holm_correction(p_values):
    p_values = np.asarray(p_values)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running_max
    return adjusted

def paired_test(runs_df, measure):
    paired = runs_df.pivot(index="run", columns="method", values=measure)
    difference = paired["icl"] - paired["baseline"]
    stat, p = wilcoxon(paired["icl"], paired["baseline"], alternative="two-sided")
    return {
        "measure": measure,
        "baseline_mean": paired["baseline"].mean(),
        "icl_mean": paired["icl"].mean(),
        "mean_difference": difference.mean(),
        "icl_wins": int((difference > 0).sum()),
        "ties": int((difference == 0).sum()),
        "W": stat,
        "p": p,
    }


# figures

def save_figure(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=300)
    plt.close(fig)

def style_axes(ax):
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

def plot_per_class_recall(dataset, runs_df, rarest_first, counts):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    positions = np.arange(dataset.n_classes)
    bar_width = 0.38

    for offset, method in zip([-bar_width / 2, bar_width / 2], METHODS):
        method_runs = runs_df[runs_df["method"] == method]
        means = [method_runs[f"recall_{dataset.class_names[c]}"].mean() for c in rarest_first]
        stds = [method_runs[f"recall_{dataset.class_names[c]}"].std() for c in rarest_first]
        ax.bar(positions + offset, means, bar_width * 0.92, yerr=stds, capsize=1.5,
               color=METHOD_COLOURS[method], label=METHOD_LABELS[method], error_kw={"linewidth": 0.6})

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{dataset.class_names[c]}\n({counts[c]})" for c in rarest_first], fontsize=6.5)
    ax.set_xlabel("Class (instances), rarest to most common")
    ax.set_ylabel("Test recall")
    ax.set_ylim(0, 1.05)
    style_axes(ax)

    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save_figure(fig, f"per_class_recall_{dataset.name}.png")

def plot_confusion_matrices(dataset, confusion_totals, rarest_first):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    names = [dataset.class_names[c] for c in rarest_first]

    for ax, method in zip(axes, METHODS):
        # reorder rows/cols rarest first, normalise each row so it shows recall per true class
        matrix = confusion_totals[method][np.ix_(rarest_first, rarest_first)].astype(float)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

        for i in range(len(names)):
            for j in range(len(names)):
                text_colour = "white" if matrix[i, j] > 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=5 if len(names) > 7 else 6.5, color=text_colour)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(METHOD_LABELS[method])

    fig.tight_layout()
    save_figure(fig, f"confusion_{dataset.name}.png")

def plot_icl_growth(dataset, run_name="r1_f1"):
    run_dir = config.RESULTS_DIR / dataset.name / "icl" / run_name
    history = pd.read_csv(run_dir / "history.csv")
    events = pd.read_csv(run_dir / "events.csv")
    class_additions = events.loc[events["event"] == "add_class", "epoch"]

    fig, (ax_f1, ax_hidden) = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True,
                                            gridspec_kw={"height_ratios": [1.3, 1]})
    ax_f1.plot(history["epoch"], history["train_macro_f1"], color=METHOD_COLOURS["baseline"],
               linestyle="--", linewidth=0.8, label="Train")
    ax_f1.plot(history["epoch"], history["val_macro_f1"], color=METHOD_COLOURS["icl"],
               linewidth=0.8, label="Validation")
    ax_f1.set_ylabel("Macro F1\n(active classes)")
    ax_f1.legend(frameon=False, loc="lower right")

    # the last removal of useless neurons happens after the final epoch,
    # so tack the final size on the end or the plot overstates it
    final_hidden = json.loads((run_dir / "summary.json").read_text())["n_hidden"]
    epochs = list(history["epoch"]) + [history["epoch"].iloc[-1]]
    hidden = list(history["n_hidden"]) + [final_hidden]
    ax_hidden.step(epochs, hidden, where="post", color="black", linewidth=0.9)
    ax_hidden.set_ylabel("Hidden neurons")
    ax_hidden.set_xlabel("Epoch")

    for ax in (ax_f1, ax_hidden):
        style_axes(ax)
        for epoch in class_additions:
            ax.axvline(epoch, color="grey", linestyle=":", linewidth=0.8)

    # label each stage with how many classes are active
    stage_starts = [0] + list(class_additions)
    stage_ends = list(class_additions) + [history["epoch"].max()]
    total_epochs = history["epoch"].max()
    for n_classes, (start, end) in enumerate(zip(stage_starts, stage_ends), start=2):
        
        if end - start < 0.05 * total_epochs:
            continue
        ax_f1.text((start + end) / 2, 1.02, str(n_classes), ha="center", va="bottom",
                   fontsize=6.5, transform=ax_f1.get_xaxis_transform())

    fig.tight_layout()
    save_figure(fig, f"icl_growth_{dataset.name}.png")

def plot_baseline_learning_curve(dataset, run_name="r1_f1"):
    run_dir = config.RESULTS_DIR / dataset.name / "baseline" / run_name
    history = pd.read_csv(run_dir / "history.csv")
    best_epoch = json.loads((run_dir / "summary.json").read_text())["best_epoch"]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(history["epoch"], history["train_macro_f1"], color=METHOD_COLOURS["baseline"],
            linestyle="--", linewidth=0.9, label="Train")
    ax.plot(history["epoch"], history["val_macro_f1"], color=METHOD_COLOURS["icl"], linewidth=0.9, label="Validation")
    ax.axvline(best_epoch, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, f"baseline_learning_curve_{dataset.name}.png")

def mean_std(series, digits=3):
    return f"{series.mean():.{digits}f} ± {series.std():.{digits}f}"


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = []
    test_rows = []
    per_class_rows = []

    for dataset_name in config.DATASET_NAMES:
        dataset = load_dataset(dataset_name)
        runs_df, confusion_totals, rarest_first, minority_classes, counts = load_all_runs(dataset)
        runs_df.insert(0, "dataset", dataset_name)
        all_runs.append(runs_df)

        for measure in ["macro_f1", "minority_recall"]:
            test_rows.append({"dataset": dataset_name, **paired_test(runs_df, measure)})

        for c in rarest_first:
            col = f"recall_{dataset.class_names[c]}"
            per_class_rows.append({
                "dataset": dataset_name,
                "class": dataset.class_names[c],
                "instances": int(counts[c]),
                "minority": c in minority_classes,
                "baseline_recall": runs_df.loc[runs_df["method"] == "baseline", col].mean(),
                "icl_recall": runs_df.loc[runs_df["method"] == "icl", col].mean(),
            })

        plot_per_class_recall(dataset, runs_df, rarest_first, counts)
        plot_confusion_matrices(dataset, confusion_totals, rarest_first)
        plot_icl_growth(dataset)
        plot_baseline_learning_curve(dataset)

    runs_df = pd.concat(all_runs, ignore_index=True)
    runs_df.to_csv(ANALYSIS_DIR / "all_runs.csv", index=False)

    # summary table

    summary_measures = ["macro_f1", "balanced_accuracy", "minority_recall", "accuracy", "train_macro_f1",
                        "n_hidden", "n_weights", "epochs", "seconds"]
    summary = runs_df.groupby(["dataset", "method"])[summary_measures].agg(["mean", "std"])
    summary.to_csv(ANALYSIS_DIR / "summary.csv")

    # statistical tests

    tests_df = pd.DataFrame(test_rows)
    tests_df["p_holm"] = holm_correction(tests_df["p"])
    tests_df["significant"] = tests_df["p_holm"] < ALPHA
    tests_df.to_csv(ANALYSIS_DIR / "wilcoxon_tests.csv", index=False)

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df["difference"] = per_class_df["icl_recall"] - per_class_df["baseline_recall"]
    per_class_df.to_csv(ANALYSIS_DIR / "per_class_recall.csv", index=False)

    # create tables
    results_table = []
    cost_table = []
    for dataset_name in config.DATASET_NAMES:
        for method in METHODS:
            r = runs_df[(runs_df["dataset"] == dataset_name) & (runs_df["method"] == method)]
            results_table.append({
                "Dataset": DATASET_LABELS[dataset_name],
                "Method": METHOD_LABELS[method],
                "Macro F1": mean_std(r["macro_f1"]),
                "Balanced accuracy": mean_std(r["balanced_accuracy"]),
                "Minority recall": mean_std(r["minority_recall"]),
                "Accuracy": mean_std(r["accuracy"]),
            })
            cost_table.append({
                "Dataset": DATASET_LABELS[dataset_name],
                "Method": METHOD_LABELS[method],
                "Hidden neurons": round(r["n_hidden"].mean(), 1),
                "Weights": round(r["n_weights"].mean()),
                "Epochs": round(r["epochs"].mean()),
                "Seconds": round(r["seconds"].mean(), 1),
            })
    pd.DataFrame(results_table).to_csv(ANALYSIS_DIR / "results_table.csv", index=False)
    pd.DataFrame(cost_table).to_csv(ANALYSIS_DIR / "cost_table.csv", index=False)

    tests_table = pd.DataFrame({
        "Dataset": tests_df["dataset"].map(DATASET_LABELS),
        "Measure": tests_df["measure"].map({"macro_f1": "Macro F1", "minority_recall": "Minority recall"}),
        "Mean difference (ICL - baseline)": tests_df["mean_difference"].round(3),
        "ICL wins": tests_df["icl_wins"].astype(str) + "/15",
        "W": tests_df["W"],
        "p": tests_df["p"].round(4),
        "p (Holm)": tests_df["p_holm"].round(4),
        "Significant": tests_df["significant"],
    })
    tests_table.to_csv(ANALYSIS_DIR / "tests_table.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.xs("mean", axis=1, level=1).round(3).to_string())
    print("\n", tests_df.drop(columns=["baseline_mean", "icl_mean"]).round(4).to_string(index=False))
    print("\n", per_class_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import config
from network import FeedForwardNet
from splits import scale_split
from training import macro_f1, train_with_early_stopping

def save_run_outputs(run_dir, history, y_test, test_probs, class_names, summary, events=None):
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if events is not None:
        pd.DataFrame(events).to_csv(run_dir / "events.csv", index=False)

    predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": test_probs.argmax(axis=1)})
    for class_index, class_name in enumerate(class_names):
        predictions_df[f"prob_{class_name}"] = test_probs[:, class_index]
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def run_baseline(dataset, split, n_hidden, results_dir=config.RESULTS_DIR):
    X_train, y_train, X_val, y_val, X_test, y_test = scale_split(dataset, split)
    rng = split.make_rng()

    net = FeedForwardNet(X_train.shape[1], n_hidden, dataset.n_classes, rng, config.ACTIVATION)
    history, run_info = train_with_early_stopping(net, X_train, y_train, X_val, y_val, rng)

    all_labels = np.arange(dataset.n_classes)
    test_probs = net.predict_proba(X_test)
    test_preds = test_probs.argmax(axis=1)

    summary = {
        "dataset": dataset.name,
        "method": "baseline",
        "repeat": split.repeat + 1,
        "fold": split.fold + 1,
        "n_hidden": net.n_hidden,
        "n_weights": net.n_weights(),
        "activation": config.ACTIVATION,
        "optimiser": config.OPTIMISER,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "patience": config.PATIENCE,
        **run_info,
        "train_macro_f1": macro_f1(y_train, net.predict(X_train), all_labels),
        "val_macro_f1": macro_f1(y_val, net.predict(X_val), all_labels),
        "test_macro_f1": macro_f1(y_test, test_preds, all_labels),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, test_preds),
    }

    run_dir = results_dir / dataset.name / "baseline" / split.run_name
    save_run_outputs(run_dir, history, y_test, test_probs, dataset.class_names, summary)
    return summary

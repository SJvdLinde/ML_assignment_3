import copy
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score
import config
from experiment import save_run_outputs
from network import FeedForwardNet
from optimisers import make_optimiser
from splits import scale_split
from training import macro_f1, train_with_early_stopping

def rarest_first_order(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes)

    return np.argsort(counts, kind="stable")

def train_icl(X_train, y_train, X_val, y_val, n_classes, rng, failed_additions_allowed):
    
    class_order = rarest_first_order(y_train, n_classes)
    position_of_class = np.empty(n_classes, dtype=int)
    position_of_class[class_order] = np.arange(n_classes)
    train_pos = position_of_class[y_train]
    val_pos = position_of_class[y_val]

    n_active = 2
    net = FeedForwardNet(X_train.shape[1], 0, n_active, rng, config.ACTIVATION, direct=True)
    optimiser = make_optimiser(config.OPTIMISER, config.LEARNING_RATE)

    history = []
    events = []
    epochs_so_far = 0

    def log_event(event, reason, val_f1):
        events.append({
            "epoch": epochs_so_far, "event": event, "reason": reason,
            "val_macro_f1_before": val_f1,
            "n_hidden_after": net.n_hidden, "n_classes_after": net.n_classes,
        })

    while True:
        # one stage = one fixed set of active classes
        train_rows = train_pos < n_active
        val_rows = val_pos < n_active
        X_tr_active, y_tr_active = X_train[train_rows], train_pos[train_rows]
        X_val_active, y_val_active = X_val[val_rows], val_pos[val_rows]
        active_labels = np.arange(n_active)

        best_val_f1 = -1.0
        train_f1_at_best = -1.0
        best_snapshot = None
        failed_additions = 0
        best_train_f1_since_best = -1.0

        while True:
            phase_history, _ = train_with_early_stopping(
                net, X_tr_active, y_tr_active, X_val_active, y_val_active, rng, optimiser=optimiser)
            for row in phase_history:
                row["epoch"] += epochs_so_far
            history.extend(phase_history)
            epochs_so_far += len(phase_history)

            val_f1 = macro_f1(y_val_active, net.predict(X_val_active), active_labels)
            train_f1 = macro_f1(y_tr_active, net.predict(X_tr_active), active_labels)

            if val_f1 > best_val_f1 + config.ICL_MIN_IMPROVEMENT or best_snapshot is None:
                best_val_f1 = val_f1
                train_f1_at_best = train_f1
                best_snapshot = (net.get_weights(), copy.deepcopy(optimiser))
                failed_additions = 0
                best_train_f1_since_best = -1.0

            else:
                failed_additions += 1
                best_train_f1_since_best = max(best_train_f1_since_best, train_f1)

            stop_growing_reason = None
            if best_val_f1 >= config.ICL_ACCEPTABLE_F1:
                stop_growing_reason = "acceptable"
            elif failed_additions >= failed_additions_allowed:
                train_still_improving = best_train_f1_since_best > train_f1_at_best + config.ICL_MIN_IMPROVEMENT
                stop_growing_reason = "overfit" if train_still_improving else "saturated"
            elif net.n_hidden >= config.ICL_MAX_HIDDEN:
                stop_growing_reason = "cap"

            if stop_growing_reason is None:
                net.add_hidden_neuron()
                log_event("add_hidden", "underfit", best_val_f1)
                continue

            neurons_before_undo = net.n_hidden
            saved_weights, saved_optimiser = best_snapshot
            net.set_weights(saved_weights)
            optimiser = saved_optimiser
            if net.n_hidden < neurons_before_undo:
                log_event("remove_hidden", stop_growing_reason, best_val_f1)
            break

        if n_active == n_classes:
            log_event("finish", stop_growing_reason, best_val_f1)
            break

        net.add_output_neuron()
        n_active += 1
        log_event("add_class", stop_growing_reason, best_val_f1)

    return net, class_order, history, events

def predict_proba_real_order(net, class_order, X):
    probs_by_position = net.predict_proba(X)
    probs = np.empty_like(probs_by_position)
    probs[:, class_order] = probs_by_position
    
    return probs

def run_icl(dataset, split, failed_additions_allowed, results_dir=config.RESULTS_DIR):
    X_train, y_train, X_val, y_val, X_test, y_test = scale_split(dataset, split)
    rng = split.make_rng()

    start = time.perf_counter()
    net, class_order, history, events = train_icl(X_train, y_train, X_val, y_val, dataset.n_classes, rng,
                                                  failed_additions_allowed)
    train_seconds = time.perf_counter() - start

    all_labels = np.arange(dataset.n_classes)
    test_probs = predict_proba_real_order(net, class_order, X_test)
    test_preds = test_probs.argmax(axis=1)
    train_preds = predict_proba_real_order(net, class_order, X_train).argmax(axis=1)
    val_preds = predict_proba_real_order(net, class_order, X_val).argmax(axis=1)

    summary = {
        "dataset": dataset.name,
        "method": "icl",
        "repeat": split.repeat + 1,
        "fold": split.fold + 1,
        "class_order": [dataset.class_names[c] for c in class_order],
        "n_hidden": net.n_hidden,
        "n_weights": net.n_weights(),
        "activation": config.ACTIVATION,
        "optimiser": config.OPTIMISER,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "patience": config.PATIENCE,
        "acceptable_f1": config.ICL_ACCEPTABLE_F1,
        "min_improvement": config.ICL_MIN_IMPROVEMENT,
        "failed_additions_allowed": failed_additions_allowed,
        "max_hidden": config.ICL_MAX_HIDDEN,
        "epochs_trained": len(history),
        "train_seconds": train_seconds,
        "train_macro_f1": macro_f1(y_train, train_preds, all_labels),
        "val_macro_f1": macro_f1(y_val, val_preds, all_labels),
        "test_macro_f1": macro_f1(y_test, test_preds, all_labels),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, test_preds),
    }

    run_dir = results_dir / dataset.name / "icl" / split.run_name
    save_run_outputs(run_dir, history, y_test, test_probs, dataset.class_names, summary, events=events)
    return summary

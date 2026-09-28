import time

import numpy as np
from sklearn.metrics import f1_score
from config import BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, OPTIMISER, PATIENCE
from optimisers import make_optimiser

def macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

def train_one_epoch(net, optimiser, X_train, y_train, rng, batch_size=BATCH_SIZE):
    shuffled_order = rng.permutation(len(y_train))

    for batch_start in range(0, len(y_train), batch_size):
        batch_rows = shuffled_order[batch_start:batch_start + batch_size]
        _, grads = net.loss_and_gradients(X_train[batch_rows], y_train[batch_rows])
        optimiser.step(net.params, grads)

def evaluate(net, X, y, labels):
    probs = net.predict_proba(X)
    loss = net.cross_entropy(probs, y)
    score = macro_f1(y, probs.argmax(axis=1), labels)
    return loss, score

def train_with_early_stopping(net, X_train, y_train, X_val, y_val, rng,
                              max_epochs=MAX_EPOCHS, patience=PATIENCE, optimiser=None):

    if optimiser is None:
        optimiser = make_optimiser(OPTIMISER, LEARNING_RATE)
    all_labels = np.arange(net.n_classes)

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_weights = net.get_weights()
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_one_epoch(net, optimiser, X_train, y_train, rng)

        train_loss, train_f1 = evaluate(net, X_train, y_train, all_labels)
        val_loss, val_f1 = evaluate(net, X_val, y_val, all_labels)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_macro_f1": train_f1, "val_macro_f1": val_f1,
            "n_hidden": net.n_hidden, "n_classes": net.n_classes,
        })

        # has to actually beat the best
        # a tie doesnt reset patience
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = net.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    net.set_weights(best_weights)

    run_info = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "train_seconds": time.perf_counter() - start_time,
    }
    return history, run_info

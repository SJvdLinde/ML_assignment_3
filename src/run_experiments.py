import json
import sys
import time

from joblib import Parallel, delayed

import config
from experiment import run_baseline
from icl import run_icl
from preprocessing import load_dataset
from splits import make_splits


def main():
    method, dataset_name = sys.argv[1], sys.argv[2]
    dataset = load_dataset(dataset_name)
    all_splits = make_splits(dataset.y)

    if method == "baseline":
        with open(config.RESULTS_DIR / "tuning" / "chosen_hidden_sizes.json") as f:
            n_hidden = json.load(f)[dataset_name]["chosen_n_hidden"]
        jobs = [delayed(run_baseline)(dataset, split, n_hidden) for split in all_splits]
    elif method == "icl":
        with open(config.RESULTS_DIR / "tuning" / "chosen_icl_k.json") as f:
            k = json.load(f)[dataset_name]["chosen_k"]
        print(f"using k={k}", flush=True)
        jobs = [delayed(run_icl)(dataset, split, k) for split in all_splits]
    else:
        raise SystemExit(f"unknown method '{method}', use baseline or icl")

    print(f"{method} on {dataset_name}: {len(jobs)} runs, {config.N_WORKERS} at a time", flush=True)
    start = time.perf_counter()
    finished = 0

    for summary in Parallel(n_jobs=config.N_WORKERS, return_as="generator_unordered")(jobs):
        finished += 1
        elapsed_min = (time.perf_counter() - start) / 60
        print(f"[{finished:2d}/{len(jobs)}] {elapsed_min:5.1f} min | r{summary['repeat']}_f{summary['fold']} "
              f"took {summary['train_seconds'] / 60:4.1f} min, {summary['epochs_trained']} epochs, "
              f"h={summary['n_hidden']}, test macro-F1={summary['test_macro_f1']:.3f}", flush=True)

    print(f"done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

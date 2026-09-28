from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 19

DATASET_NAMES = ["wall_robot", "cardiotocography", "covertype"]
COVERTYPE_SUBSAMPLE_SIZE = 20_000

# cross-validation
N_FOLDS = 5
N_REPEATS = 3
VALIDATION_FRACTION = 0.2

# training
ACTIVATION = "tanh"
OPTIMISER = "adam"
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 30

# hidden layer sizes
HIDDEN_SIZES_TO_TRY = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
HIDDEN_SIZE_OVERRIDES = {"wall_robot": 32}

# incremental class learning
ICL_ACCEPTABLE_F1 = 0.95
ICL_MIN_IMPROVEMENT = 0.005
ICL_FAILED_ADDITIONS_TO_TRY = [3, 5, 10, 20, 40]
ICL_MAX_HIDDEN = 256
N_WORKERS = 8

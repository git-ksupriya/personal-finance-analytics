import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Database configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "spakov_finance")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Data path
DATA_DIR = Path(os.getenv("DATA_DIR", "c:/Users/megha/Downloads/Spakov_DL"))
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing & Windowing parameters
SEQ_LEN = 8          # Default window: 8 weeks
SEQ_LENS = [8, 10, 12]
MIN_NONZERO_WEEKS = 10

# Chronological split weeks (1-indexed calendar weeks)
TRAIN_WEEK_END = 73
VAL_WEEK_END = 89
TEST_WEEK_END = 105

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
PATIENCE = 7
SEED = 42

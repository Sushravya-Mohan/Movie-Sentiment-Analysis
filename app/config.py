import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Project structure
    ROOT_DIR = Path(__file__).parent.parent
    APP_DIR = ROOT_DIR / "app"
    MODEL_DIR = ROOT_DIR / "models"
    DATA_DIR = ROOT_DIR / "data"

    # Ensure directories exist
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # Model files
    MODEL_PATH = MODEL_DIR / "hybrid_model.h5"
    TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"

    # Data parameters
    MAX_WORDS = 10000
    MAX_LEN = 200
    EMBEDDING_DIM = 100

    # Model parameters
    LSTM_UNITS = 64
    GRU_UNITS = 64
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.3

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 10
    VALIDATION_SPLIT = 0.2

    # Runtime environment
    ENV = os.getenv("ENV", "development")

    @classmethod
    def is_model_available(cls) -> bool:
        """Check if pre-trained model files exist"""
        return cls.MODEL_PATH.exists() and cls.TOKENIZER_PATH.exists()

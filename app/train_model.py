import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import Config
from app.data_preprocessing import TextPreprocessor
from app.model import SentimentModel


def download_imdb_dataset(data_dir: Path):
    """Download and extract IMDB dataset"""
    import requests
    import tarfile
    import io

    # IMDB dataset URL
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # Download dataset
    response = requests.get(url)
    tar = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
    tar.extractall(path=data_dir)
    tar.close()


def load_imdb_data(data_dir: Path):
    """Load positive and negative movie reviews"""
    # Read positive and negative reviews
    pos_files = list((data_dir / "aclImdb/train/pos").glob("*.txt"))
    neg_files = list((data_dir / "aclImdb/train/neg").glob("*.txt"))

    # Read reviews and labels
    pos_reviews = [(open(f, "r", encoding="utf-8").read(), 1) for f in pos_files]
    neg_reviews = [(open(f, "r", encoding="utf-8").read(), 0) for f in neg_files]

    # Combine and shuffle
    all_reviews = pos_reviews + neg_reviews
    np.random.shuffle(all_reviews)

    # Create dataframe
    df = pd.DataFrame(all_reviews, columns=["text", "sentiment"])

    return df


def train_sentiment_model(config: Config):
    """Full training pipeline"""
    # Ensure data directory exists
    config.DATA_DIR.mkdir(exist_ok=True)

    # Download dataset if not exists
    dataset_path = config.DATA_DIR / "aclImdb"
    if not dataset_path.exists():
        download_imdb_dataset(config.DATA_DIR)

    # Load data
    df = load_imdb_data(config.DATA_DIR)

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    # Initialize preprocessor
    preprocessor = TextPreprocessor(config)

    # Prepare training data
    X_train = preprocessor.prepare_data(train_df["text"], fit=True)
    X_test = preprocessor.prepare_data(test_df["text"])

    # Initialize and build model
    model_builder = SentimentModel(config, vocab_size=preprocessor.vocabulary_size)
    model = model_builder.build()

    # Train model
    history = model.fit(
        X_train,
        train_df["sentiment"],
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=config.VALIDATION_SPLIT,
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, test_df["sentiment"])
    print(f"Test accuracy : {test_acc}")

    # Save model and tokenizer
    model_builder.save()

    return model, history


def main():
    config = Config()
    train_sentiment_model(config)


if __name__ == "__main__":
    main()

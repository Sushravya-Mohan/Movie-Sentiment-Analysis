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

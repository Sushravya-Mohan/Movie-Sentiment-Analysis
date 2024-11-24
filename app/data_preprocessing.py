# Importing necessary libraries
import re
import pickle
from pathlib import Path
from token import OP
from typing import List, Tuple, Optional
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from app.config import Config


class TextPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer: Optional[Tokenizer] = None

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data
        Args:
            text(str) : Raw text input
        Returns:
            str : Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def prepare_data(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Prepare text data for model training or inference.
        Args:
            texts (List[str]) : List of text samples
            fit (bool) : Whether to fit the tokenizer on the data
        Returns:
            np.ndarray : Padded sequences ready for model input
        """
        # Clean texts
        cleaned_texts = [
            self.clean_text(text) for text in tqdm(texts, desc="Cleaning texts")
        ]

        if fit:
            # Create and fit tokenizer
            self.tokenizer = Tokenizer(num_words=self.config.MAX_WORDS)
            self.tokenizer.fit_on_texts(cleaned_texts)

            # Save tokenizer
            self._save_tokenizer()

        elif self.tokenizer is None:
            # Load existing tokenizer if not fitting
            self.load_tokenizer()

        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Either fit new data or load existing tokenizer."
            )

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.config.MAX_LEN,
            padding="post",
            truncating="post",
        )
        return padded_sequences

    def _save_tokenizer(self) -> None:
        """Save tokenizer to disk"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save")

        with open(self.config.TOKENIZER_PATH, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load_tokenizer(self) -> None:
        """Load saved tokenizer from disk"""
        if not self.config.TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"No tokenizer found at {self.config.TOKENIZER_PATH}"
            )

        with open(self.config.TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)

    @property
    def vocabulary_size(self) -> int:
        """Get size of vocabulary"""
        if self.tokenizer is None;
            raise ValueError("No tokenizer available")
        return len(self.tokenizer.word_index) + 1
# Import required libraries
from os import name
from tkinter import NO
from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    GRU,
    Dense,
    Concatenate,
    Dropout,
    GlobalMaxPooling1D,
)
from app.config import Config


class SentimentModel:
    """Hybrid RNN model combining LSTM and GRU for sentiment analysis"""

    def __init__(self, config: Config, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size
        self.model: Model = None

    def build(self) -> Model:
        """
        Build hybrid RNN model architecture
        Returns :
            Model : Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.config.MAX_LEN,))

        # Embedding layer
        embedding = Embedding(
            self.vocab_size,
            self.config.EMBEDDING_DIM,
            input_length=self.config.MAX_LEN,
            name="embedding",
        )(inputs)

        # LSTM branch
        lstm_branch = LSTM(
            self.config.LSTM_UNITS, return_sequences=True, name="lstm_branch"
        )(embedding)
        lstm_max = GlobalMaxPooling1D(name="lstm_pooling")(lstm_branch)

        # GRU branch
        gru_branch = GRU(
            self.config.GRU_UNITS, return_sequences=True, name="gru_branch"
        )(embedding)
        gru_max = GlobalMaxPooling1D(name="gru_pooling")(gru_branch)

        # Combine branches
        merged = Concatenate(name="merge_branches")([lstm_max, gru_max])

        # Dense layers
        dense = Dense(self.config.DENSE_UNITS, activation="relu", name="dense_1")(
            merged
        )
        dropout = Dropout(self.config.DROPOUT_RATE, name="dropout")(dense)
        outputs = Dense(1, activation="sigmoid", name="output")(dropout)

        # Create model
        self.model = Model(
            inputs=inputs, outputs=outputs, name="hybrid_sentiment_model"
        )

        # Compile model
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return self.model

    def load(self) -> Model:
        """
        Load pre-trained model from disk
        Returns:
            Model : Loaded Keras model
        """
        if not self.config.MODEL_PATH.exists():
            raise FileNotFoundError(f"No model found at {self.config.MODEL_PATH}")

        self.model = load_model(self.config.MODEL_PATH)
        return self.model

    def save(self) -> None:
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(self.config.MODEL_PATH)

    def get_summary(self) -> str:
        """Get model summary as string"""
        if self.model is None:
            raise ValueError("No model available")

        # Create string buffer
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)

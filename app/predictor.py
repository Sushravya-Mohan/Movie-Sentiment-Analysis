from typing import Dict, Union, Any

from altair import sequence
import numpy as np
from tensorflow.keras.models import Model

from app.config import Config
from app.data_preprocessing import TextPreprocessor
from app.model import SentimentModel


class SentimentPredictor:
    """Manages sentiment prediction using pre-trained model"""

    def __init__(self, config: Config):
        """
        Initialize predictor with configuration
        Args:
            config (Config) : Application configuration
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config)
        self.model_builder = None
        self.model = None

    def load(self) -> None:
        """
        Load pre-trained model and tokenizer
        Raises:
            FileNotFoundError : If model or tokenizer files are missing
        """
        # Load tokenizer
        self.preprocessor.load_tokenizer()

        # Initalize and load model
        self.model_builder = SentimentModel(
            self.config, vocab_size=self.preprocessor.vocabulary_size
        )
        self.model = self.model_builder.load()

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for given text
        Args:
            text (str) : Input text for sentiment analysis
        Returns:
            Dict containing sentiment prediction details
        """
        # Ensure model is loaded
        if self.model is None or self.preprocessor.tokenizer is None:
            self.load()

        # Preprocess text
        sequence = self.preprocessor.prepare_data([text])

        # Get prediction
        prediction = self.model.predict(sequence)[0][0]

        # Interpret results
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "raw_score": float(prediction),
        }

    def batch_predict(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        Args:
            text (list) : List of texts to analyze
        Returns:
            list of prediction dictionaries
        """
        return [self.predict(text) for text in texts]

    def get_model_performance(self) -> Dict[str, str]:
        """
        Retrieve basic model performance information
        Returns:
            Dict with model performance metrics
        """
        if self.model is None:
            self.load()

        try:
            # Extract model performance details
            return {
                "model_name": self.model.name,
                "input_shape": str(self.model.input_shape),
                "output_shape": str(self.model.output_shape),
                "total_params": str(self.model.count_params()),
            }
        except Exception as e:
            return {"error": str(e)}

    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """
        Provide additional insights into prediction
        Args:
            text (str) : Input text to analyze
        Returns:
            Dict with additional prediction insights
        """
        prediction_result = self.predict(text)

        # Basic text analysis
        return {
            **prediction_result,
            "text_length": len(text),
            "word_count": len(text.split()),
            "is_short_text": len(text.split()) < 5,
        }

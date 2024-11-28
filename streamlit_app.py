import trace
import streamlit as st
import traceback

from app import config
from app.config import Config
from app.predictor import SentimentPredictor
from app.train_model import train_sentiment_model


def init_session_state():
    """Initialize Streamlit session state"""
    if "predictor" not in st.session_state:
        config = Config()
        st.session_state.predictor = SentimentPredictor(config)


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Sentiment Analysis App", page_icon="🔍", layout="centered"
    )

    # Initialize session state
    init_session_state()
    predictor = st.session_state.predictor

    # Main app
    st.title("🎭 Movie Review Sentiment Analyzer")
    st.write("Analyze the sentiment of movie reviews using hybrid RNN model")

    # Sidebar
    st.sidebar.title("App Controls")

    # Check model availability
    config = Config()
    if not config.is_model_available():
        st.warning("No pre-trained model found. Training a new model......")
        with st.spinner("Training model..... This may take a few minutes"):
            train_sentiment_model(config)

    # Main interaction
    with st.form("sentiment_form"):
        text = st.text_area(
            "Enter your movie review:",
            placeholder="Type your review here...",
            height=200,
        )
        submitted = st.form_submit_button("Analyze sentiment")

    if submitted:
        if not text:
            st.error("Please enter a review to analyze")
        else:
            try:
                # Load model if not already loaded
                if predictor.model is None:
                    predictor.load()

                # Predict sentiment
                result = predictor.predict(text)

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sentiment", result["sentiment"])
                with col2:
                    st.metric("Confidence", f"{result["confidence"]:.2%}")
                with col3:
                    st.metric("Raw Score", f"{result["raw_score"]:.3f}")

                # Additional insights
                st.subheader("Prediction Insights")
                extra_info = predictor.explain_prediction(text)
                st.json(extra_info)

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.error(traceback.format_exc())

    # Sidebar information
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses hybrid RNN model (LSTM + GRU)"
        "to analyze sentiment in movie reviews."
        "Trained on IMDB dataset"
    )


if __name__ == "__main__":
    main()

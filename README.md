# Sentiment Analysis Web App

## Overview
Hybrid RNN Sentiment Analysis web application using LSTM and GRU architectures to analyze movie review sentiments.

## Features
- Hybrid RNN model combining LSTM and GRU
- Sentiment prediction with confidence score
- Streamlit web interface
- Pre-trained model with IMDB dataset

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/Sushravya-Mohan/Movie-Sentiment-Analysis.git
cd sentiment-analysis
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Training Model
```bash
python scripts/train_model.py
```

### Running App
```bash
streamlit run streamlit_app.py
```

## Deployment
Deploy on Streamlit Cloud:
1. Push code to GitHub
2. Connect Streamlit Cloud to repository
3. Set root directory to project root

## Project Structure
- `app/`: Core application modules
- `models/`: Saved model and tokenizer
- `scripts/`: Training and utility scripts
- `streamlit_app.py`: Main application entry point

## Technologies
- TensorFlow
- Streamlit
- Pandas
- NumPy
- scikit-learn

## License
MIT License

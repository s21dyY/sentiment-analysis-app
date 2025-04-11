# 🧠 Sentiment Analysis with LSTM

This project is a deep learning–based sentiment analysis app that uses an LSTM (Long Short-Term Memory) neural network to classify user input text as **Positive** or **Negative** sentiment.

✅ Try it live: [https://sy-lstm.streamlit.app/](https://sy-lstm.streamlit.app/)

---

## 📌 Project Overview

This Streamlit app allows users to input any English sentence or product review and instantly receive a predicted sentiment.

It was trained on the **Amazon Review Polarity** dataset using a Keras-based LSTM model.

---

## 📦 Dataset

- **Source**: [Amazon Review Polarity (Fancyzhx on Hugging Face)](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- **Samples**: Over 4 million product reviews from Amazon
- **Labels**:
  - `1`: Negative
  - `2`: Positive

---

## 🧠 Model Details

The app uses a deep learning model with the following architecture:

- `Tokenizer`: top 5,000 words
- `Input length`: 200 padded tokens
- `Embedding`: 128-dimensional vectors
- `Model`:
  - Bidirectional LSTM (64 units)
  - Dropout (rate = 0.5)
  - Dense (1 unit with sigmoid activation)

- **Loss function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Training**: 3 epochs, batch size of 64

---

## 🖥️ App Features

- Clean, minimal UI via Streamlit
- Live sentiment prediction for any text
- Shows model confidence score
- Friendly explanations and visualizations
- Example reviews for easy testing
- Expandable sections explaining:
  - How the model works
  - Dataset details
  - Confidence bar charts

---

## 📁 Project Structure
```bash
sentiment-analysis-app
model
├── sentiment_lstm_model.h5 # Trained model file
notebooks
├── EDA_and_Modeling.ipynb  # Model training and EDA
src
├── app.py                  # Main Streamlit app
├── tokenizer.pkl           # Tokenizer used for text preprocessing
├── utils.py                # Helper functions for preprocessing & prediction
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Author
Sandy Yang
📧 sandy.yang992@gmail.com

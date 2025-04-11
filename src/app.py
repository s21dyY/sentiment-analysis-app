import streamlit as st
from utils import predict_sentiment

st.title("🧠 Smart Sentiment Analysis")
text = st.text_area("Enter text for analysis:")

if st.button("Analyze"):
    if text.strip():
        sentiment = predict_sentiment(text)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text.")

# sidebar & footer
st.sidebar.header("Smart Sentiment Analysis")
st.sidebar.info(
    "This app uses a pre-trained LSTM model to analyze the sentiment of your text. "
    "It can classify the sentiment as either Positive or Negative."
)

with st.sidebar.expander("📦 Dataset Info"):
    st.markdown(
        """
        This model was trained on the **Amazon Review Polarity** dataset, available on [Hugging Face](https://huggingface.co/datasets/fancyzhx/amazon_polarity).

        **Dataset Summary:**
        - **Source:** Amazon.com product reviews
        - **Labels:**  
            - 1 = Negative  
            - 2 = Positive  
        - **Size:** 4 million+ training samples
        - **Type:** Binary sentiment classification
        - **Text Content:**  
          Each entry includes a product title and review body.

        This large-scale dataset helps the model learn how people express opinions in real-world scenarios.
        """
    )
with st.sidebar.expander("📘 How the Model Works"):
    st.markdown(
        """
        This app uses a deep learning model called an **LSTM (Long Short-Term Memory)** to understand the emotion in your text — whether it's positive or negative.

        ### 🧠 Step-by-step: How your text is processed
        1. **Your sentence is cleaned**  
           It’s converted to lowercase and removes symbols like `!`, `#`, and links.
           
        2. **Words are turned into numbers**  
           Each word is matched to a number using a *Tokenizer*, based on the most common 5,000 words the model saw during training.
           
        3. **Padding is added**  
           All inputs are padded to the same length (200 words) so the model can process them evenly.

        4. **Words become vectors**  
           Each word number is transformed into a 128-dimensional vector — basically giving it meaning in context.

        5. **LSTM reads the sentence**  
           Like a human, it reads your sentence from left to right, remembering important parts using its memory cells.

        6. **The model predicts sentiment**  
           Finally, it outputs a number between 0 and 1. If it’s above 0.5 → **Positive**, below 0.5 → **Negative**.

        ---
        ### 🖼️ Example: Behind the scenes

        Let’s say you enter:
        > *"I absolutely loved this product. It works perfectly!"*

        Here's how it's processed:

        | Step | Output |
        |------|--------|
        | Cleaned Text | "i absolutely loved this product it works perfectly" |
        | Tokenized    | [12, 87, 456, 33, 942, 6, 1093] |
        | Padded       | [12, 87, 456, 33, 942, 6, 1093, 0, 0, ..., 0] |
        | Prediction   | 0.92 → **Positive** |

        ---
        ### 🤖 Model Training Info
        - Architecture: Embedding → BiLSTM → Dropout → Dense
        - Trained on: Amazon product reviews
        - Optimizer: Adam | Loss: Binary Crossentropy | Epochs: 3

        ---
        > *Even if you use slang or casual writing, the model tries to understand the meaning behind your words.* 💬
        """,
        unsafe_allow_html=True
    )
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>© 2025 Sandy Yang. All rights reserved.</p>",
    unsafe_allow_html=True
)
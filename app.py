import os
import pickle

import numpy as np
import streamlit as st
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.saving import load_model
from keras.src.utils import pad_sequences

# Define the project's root directory
project_root = "/Users/sunnythesage/PythonProjects/Data-Science-BootCamp/03-Deep-Learning-BootCamp/11 - End to End LSTM & GRU Deep Learning Project/advanced-next-word-prediction-using-LSTM-and-GRU"

# Change the current working directory to the project's root
os.chdir(project_root)

# --- Artifacts ---

# Define the relative path to the artifacts directory
artifacts_dir = os.path.join(os.getcwd(), "artifacts")

# --- Raw Data ---

# Define the relative path to the data/raw directory
raw_data_dir = os.path.join(os.getcwd(), "data", "raw")

# Create the directory if it doesn't exist
os.makedirs(artifacts_dir, exist_ok=True)

# Load the pre-trained LSTM Model
model = load_model(os.path.join(artifacts_dir, "next_word_gru_model.keras"))

# Load or create tokenizer
tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pickle")

if os.path.exists(tokenizer_path):
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded from pickle.")
    except (EOFError, pickle.UnpicklingError):
        print("Tokenizer pickle corrupted. Creating new tokenizer.")
        tokenizer = None
else:
    print("Tokenizer pickle not found. Creating new tokenizer.")
    tokenizer = None

if tokenizer is None:
    # Load the training text
    text_file_path = os.path.join(raw_data_dir, "shakespeare-hamlet.txt")
    with open(text_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    print(f"New tokenizer created with {total_words} words.")


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[
            -(max_sequence_len - 1) :
        ]  # Ensure the sequence length matches max_sequence_len-1

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None


# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")

input_text = st.text_input("Enter the sequence of Words", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = (
        model.input_shape[1] + 1
    )  # Retrieve the max sequence length from the model input shape

    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

    st.write(f"Next word: {next_word}")

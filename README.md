# Advanced Next Word Prediction Using LSTM and GRU

This project develops a sophisticated next word prediction model leveraging Long-Short-Term Memory (LSTM) and Gated
Recurrent Unit (GRU) networks.
Designed to understand and generate a human-like text, the model is trained on the text
of
Shakespeare's "Hamlet," providing a challenging dataset for sequence prediction.
The project encompasses comprehensive
data preprocessing, advanced model architecture, rigorous evaluation, and interactive deployment via Streamlit.

## Problem Statement:

Next word prediction is a fundamental task in natural language processing (NLP), with applications ranging from
predictive text input to sophisticated text generation systems.
This project aims to create a high-performance model
capable of accurately predicting the next word in a given sequence, using the rich and complex text of "Hamlet" as
a training corpus.

## Motivation and Business Value:

Accurate next word prediction is crucial for enhancing user experience in various applications, including chatbots,
search engines, and writing assistance tools.
This project demonstrates the application of advanced recurrent neural
networks (RNNs), specifically LSTM and GRU, to address this challenge.
By capturing long-range dependencies in text,
these models can generate contextually relevant predictions, leading to improved user interactions and more natural
language generation.

## Dataset:

The project uses the complete text of Shakespeare's "Hamlet."
This dataset offers a rich linguistic structure and
complex vocabulary, providing a challenging yet rewarding training environment for the model.

## Methodology and Implementation:

This project provides a detailed, reproducible workflow for next word prediction using Python within a Jupyter Notebook
environment.
The methodology encompasses:

1. **Data Collection and Ingestion:**
    * Loading the "Hamlet" text file.
2. **Data Preprocessing:**
    * Text cleaning, including removal of special characters and punctuation.
    * Tokenization: Converting the text into a sequence of words.
    * Vocabulary creation: Building a unique set of words from the text.
    * Sequence generation: Creating input-output pairs for training the model.
    * Padding sequences: Ensuring uniform input lengths for the RNN.
    * One-hot encoding of the output words.
3. **Model Building:**
    * Embedding layer for word vector representation.
    * LSTM and/or GRU layers to capture long-range dependencies.
    * Dense layers for output prediction.
    * Softmax activation functions to predict the probability of the next word.
    * Experimenting with different architectures, including stacked LSTMs/GRUs and bidirectional RNNs.
4. **Model Training:**
    * Splitting the data into training and validation sets.
    * Training the model with optimized hyperparameters.
    * Implementing early stopping to prevent overfitting.
    * Utilizing techniques like dropout and recurrent dropout for regularization.
5. **Model Evaluation:**
    * Evaluating the model's performance on a test set.
    * Testing the model with custom input sequences.
    * Assessing the model's ability to generate coherent and contextual relevant text.
6. **Model Deployment and Streamlit Integration:**
    * Saving the trained model.
    * Developing a Streamlit application (`app.py`) for interactive testing.
    * Creating a user-friendly interface for inputting word sequences and displaying predicted next words.

## Key Technical Aspects:

* Utilization of TensorFlow/Keras for deep learning model development.
* Implementation of word embeddings for effective text representation.
* Handling sequential data using LSTM and GRU networks.
* Advanced regularization techniques to prevent overfitting.
* Streamlit for interactive model deployment and testing.

## Streamlit Integration:

This project includes a Streamlit application (`app.py`) for easy deployment and interaction with the trained LSTM/GRU
model.
Users can input a sequence of words through a user-friendly interface, and the application will predict the next
word in real-time.

To run the Streamlit app:

1. Ensure you have Streamlit installed (`pip install streamlit`).
2. Navigate to the project's root directory in your terminal.
3. Run the command `streamlit run app.py`.

This will launch the application in your web browser, allowing you to test the model with various input sequences.

## Potential Applications:

* **Predictive Text Input:** Enhancing user experience in text messaging and email applications.
* **Chatbot Development:** Improving the conversational capabilities of chatbots.
* **Text Generation Systems:** Creating coherent and contextually relevant text.
* **Language Modeling for Search Engines:** Improving search query suggestions.
* **Code Completion Tools:** Providing intelligent code suggestions for developers.

## Note on Model Selection:

This project focuses on LSTM and GRU networks due to their proven effectiveness in capturing long-range dependencies in
sequential data.
While other RNN architectures and transformer-based models could be considered for further performance
enhancements, this project emphasizes a robust and practical implementation of LSTM and GRU for next word prediction.
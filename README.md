# Chatbot Intent Classification Project

An end-to-end Machine Learning project to build a simple chatbot that classifies user messages into predefined intents and returns appropriate responses.

## Project Overview

This project implements an NLP pipeline to classify textual inputs into distinct usage tags (intents), such as greetings, farewells, or factual questions. It leverages structural preprocessing techniques like tokenization, stop-word removal, and lemmatization using the `nltk` library, and then maps the cleaned text into a multidimensional numerical space mapping using Term-Frequency Inverse Document-Frequency (TF-IDF). 

It trains multiple models like:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine

The script evaluates, compares them based on accuracy & F1 score, and allows predictions via a web interface or CLI.

## Project Structure

```
chatbot_intent_classifier/
│
├── dataset/
│   └── intents.json         # Chatbot intent tags, patterns, and responses
│
├── models/                  # Will contain generated models
│   ├── intent_model.pkl     # Trained ML classifier
│   └── vectorizer.pkl       # TF-IDF Vectorizer
│
├── src/
│   ├── preprocess.py        # NLP Preprocessing pipeline containing lemmatization etc
│   ├── train_model.py       # ML Training pipeline comparing various classifiers
│   ├── predict_intent.py    # Runtime intent prediction logic
│   └── chatbot.py           # Command Line interface chatbot
│
├── app.py                   # Streamlit web interface
├── requirements.txt         # Project Dependencies
└── README.md                # This file
```

## Dataset Format

The `dataset/intents.json` file dictates how the classification patterns are established. Example:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hello", "Hi"],
      "responses": ["Hello!", "Hi there!"]
    }
  ]
}
```

## Installation

1. Create and source into a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to train the model

Run the training pipeline. It evaluates different models and saves the best one (based on evaluation parameters) to the `models` directory:
```bash
python src/train_model.py
```

## How to run the chatbot

### 1. Command Line Interface
Interact with the bot in your terminal:
```bash
python src/chatbot.py
```

### 2. Streamlit Web Interface
Start the application in your browser:
```bash
streamlit run app.py
```

## Future Improvements

1. **Deep Learning Models:** Use Recurrent Neural Networks (RNN) or LSTMs to memorize longer interaction states natively sequences of text.
2. **Transformer-based models:** Fine-tune BERT/RoBERTa representations instead of TF-IDF word frequencies for deeper contextual semantic understanding.
3. **Context-Aware chatbot:** Append conversational logs in state to maintain context dynamically across previous follow-up queries.
4. **API Deployment:** Expose the resulting model over FastAPI and deploy via Docker to cloud services.

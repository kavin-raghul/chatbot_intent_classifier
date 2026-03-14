import os
import joblib
import json
import random

try:
    from src.preprocess import TextPreprocessor
except ImportError:
    from preprocess import TextPreprocessor

class IntentPredictor:
    """Class to load the trained model and make intent predictions."""
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.model_path = os.path.join(base_dir, 'models', 'intent_model.pkl')
        self.vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')
        self.data_path = os.path.join(base_dir, 'dataset', 'intents.json')
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError("Model or vectorizer not found. Please run train_model.py first.")
            
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.preprocessor = TextPreprocessor()
        
        with open(self.data_path, 'r') as f:
            self.intents_data = json.load(f)

    def predict(self, text):
        """Predicts the intent tag and confidence for the given text."""
        processed_text = self.preprocessor.preprocess(text)
        features = self.vectorizer.transform([processed_text])

        # Get prediction
        prediction = self.model.predict(features)[0]

        # Get confidence score
        probabilities = self.model.predict_proba(features)
        confidence = max(probabilities[0])

        return prediction, confidence
        
    def get_response(self, text):
        """Returns a matching response for the predicted intent."""
        intent_tag, confidence = self.predict(text)
        
        for intent in self.intents_data['intents']:
            if intent['tag'] == intent_tag:
                responses = intent['responses']
                return random.choice(responses), confidence
                
        return "I'm not sure how to respond to that.", confidence


if __name__ == "__main__":
    # Test script standalone
    try:
        predictor = IntentPredictor()
        sample = "Hello there!"
        tag, confidence = predictor.predict(sample)
        response, conf = predictor.get_response(sample)

        print(f"Input: {sample}")
        print(f"Predicted Intent: {tag}")
        print(f"Confidence: {round(confidence*100,2)}%")
        print(f"Response: {response}")

    except FileNotFoundError as e:
        print(e)
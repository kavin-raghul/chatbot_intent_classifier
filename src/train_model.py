import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

try:
    from src.preprocess import TextPreprocessor
except ImportError:
    from preprocess import TextPreprocessor
def load_data(filepath):
    """Loads intent matching JSON data and converts it to a pandas DataFrame."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    records = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            records.append({
                'text': pattern,
                'intent': intent['tag']
            })
            
    return pd.DataFrame(records)

def train_and_eval():
    """
    Main function to load the dataset, preprocess text, engineer features using TF-IDF,
    and train multiple classification models, selecting the best one to act as intent classifier.
    """
    print("Loading data...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dataset', 'intents.json')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    df = load_data(data_path)
    print(f"Loaded {len(df)} samples.")
    
    print("Preprocessing text...")
    preprocessor = TextPreprocessor()
    df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    
    # Feature Engineering
    print("Extracting features with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['intent']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nEvaluating Models:")
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel='linear', probability=True)
    }
    
    best_model = models["Multinomial Naive Bayes"]
    best_f1 = -1
    best_name = "Multinomial Naive Bayes"
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            
    # Train the FINAL model on all data since the intents dataset is typically small
    print(f"\nTraining final '{best_name}' model on ALL data for maximum chatbot accuracy...")
    best_model.fit(X, y)
    
    print("\nEvaluation on full dataset:")
    y_full_pred = best_model.predict(X)
    print(f"Final Accuracy: {accuracy_score(y, y_full_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_full_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_full_pred))
    
    # Save the model and vectorizer
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'intent_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nSaved best model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")

if __name__ == "__main__":
    train_and_eval()

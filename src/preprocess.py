import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """
    Handles robust text preprocessing including tokenization,
    punctuation removal, stopword removal, and lemmatization.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)

    def preprocess(self, text):
        """
        Preprocesses a string of text.
        """
        # 1. Lowercase text
        text = text.lower()
        
        # 2. Tokenization
        tokens = nltk.word_tokenize(text)
        
        # 3. Remove punctuation, stop words, and apply Lemmatization
        clean_tokens = []
        for token in tokens:
            if token not in self.punctuation and token not in self.stop_words:
                lemmatized_token = self.lemmatizer.lemmatize(token)
                clean_tokens.append(lemmatized_token)
                
        return " ".join(clean_tokens)

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    sample_text = "Hello! How are you doing today? I need some help."
    print("Original:", sample_text)
    print("Preprocessed:", preprocessor.preprocess(sample_text))

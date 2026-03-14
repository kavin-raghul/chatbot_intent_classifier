import sys
import os

# Add the project root to sys.path so the module can be found
# regardless of whether it's run via `python src/chatbot.py` or from the root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predict_intent import IntentPredictor

def run_chatbot():
    print("=" * 50)
    print("🤖 Chatbot Intent Classifier Console")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50)
    
    try:
        predictor = IntentPredictor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python src/train_model.py' first to generate the models.")
        sys.exit(1)
        
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Bot: Goodbye! Have a great day.")
                break
                
            if user_input.strip() == "":
                continue
                
            response = predictor.get_response(user_input)
            print(f"Bot: {response}")
            
        except KeyboardInterrupt:
            print("\nBot: Goodbye! Have a great day.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()

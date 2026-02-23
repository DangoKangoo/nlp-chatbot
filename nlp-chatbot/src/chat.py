import json
import random
from pathlib import Path

import joblib

INTENTS_PATH = Path("data/intents.json")
MODEL_PATH = Path("models/intent_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")


def load_intents():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # map tag -> responses
    return {i["tag"]: i["responses"] for i in data["intents"]}


def predict_intent(text: str, model, vectorizer) -> str:
    X = vectorizer.transform([text])
    return model.predict(X)[0]


def main():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        print("Model files not found in /models.")
        print("Run: python src\\train_intent_model.py")
        return

    intents = load_intents()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("CP4310 Chatbot (type 'quit' to exit)")
    while True:
        user = input("\nYou: ").strip()
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            print("Bot: Bye!")
            break

        tag = predict_intent(user, model, vectorizer)
        responses = intents.get(tag, ["Sorry — I’m not sure about that."])
        print(f"Bot ({tag}): {random.choice(responses)}")


if __name__ == "__main__":
    main()
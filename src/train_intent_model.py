import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

DATA_PATH = Path("data/intents.json")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, MODEL_DIR / "intent_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")

print("\nModel training complete. Saved:")
print(f"- {MODEL_DIR / 'intent_model.pkl'}")
print(f"- {MODEL_DIR / 'vectorizer.pkl'}")
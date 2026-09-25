"""
Step 3: Inference script - classify a new email as Phishing or Safe,
with a confidence score and a list of suspicious words that influenced the decision.
"""

import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200

model = load_model("phishing_lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

SUSPICIOUS_TERMS = [
    "urgent", "verify", "suspend", "click here", "act now", "password",
    "confirm", "account", "limited time", "winner", "prize",
    "bank", "security alert", "update your", "login", "credentials",
    "wire transfer", "gift card", "social security", "unusual activity"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_suspicious_terms(original_text):
    text_lower = original_text.lower()
    found = [term for term in SUSPICIOUS_TERMS if term in text_lower]
    return found

def classify_email(email_text):
    cleaned = clean_text(email_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Phishing" if prob > 0.5 else "Safe"
    confidence = prob if label == "Phishing" else 1 - prob

    suspicious = find_suspicious_terms(email_text)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "phishing_probability": round(prob * 100, 2),
        "suspicious_terms_found": suspicious
    }

if __name__ == "__main__":
    test_emails = [
        "Dear customer, your account has been suspended due to unusual activity. "
        "Click here to verify your password and confirm your identity immediately or your account will be closed.",

        "Hi team, attached is the quarterly report for review before tomorrow's meeting. Let me know if you have questions.",

        "URGENT: You have won a prize! Claim your gift card now by confirming your bank details. Limited time offer, act now!"
    ]

    print("=" * 70)
    for i, email in enumerate(test_emails, 1):
        result = classify_email(email)
        print(f"\nEmail {i}:")
        print(f"  Text: {email[:80]}...")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Phishing probability: {result['phishing_probability']}%")
        print(f"  Suspicious terms found: {result['suspicious_terms_found']}")
        print("-" * 70)

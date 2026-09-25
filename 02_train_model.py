"""
Step 2: Train an LSTM model to classify phishing vs. legitimate emails.
Includes text preprocessing, tokenization, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
EPOCHS = 5
BATCH_SIZE = 64

print("Loading dataset...")
df = pd.read_csv("phishing_emails.csv")
df = df.dropna(subset=["Email Text", "Email Type"])

df["label"] = (df["Email Type"] == "Phishing Email").astype(int)

print(f"Total emails: {len(df)}")
print(f"Phishing: {df['label'].sum()} | Safe: {(df['label']==0).sum()}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
df["clean_text"] = df["Email Text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print("Tokenizing...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post")
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding="post", truncating="post")

print("Building model...")
model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

print("Training...")
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

print("\nEvaluating on test set...")
y_pred_prob = model.predict(X_test_seq)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("phishing_lstm_model.keras")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nModel saved as phishing_lstm_model.keras")
print("Tokenizer saved as tokenizer.pkl")

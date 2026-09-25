# AI-phishing-email-detector
ML-based phishing email classifier using NLP, with confidence scoring and explainability.

# AI-Powered Phishing Email Detector

An LSTM-based deep learning model that classifies emails as phishing or legitimate, using NLP preprocessing and word embeddings. Outputs a confidence score and a list of suspicious terms for analyst-friendly explainability.

## What This Project Does

- Trains an LSTM neural network on 18,650 real labeled emails (phishing + legitimate)
- Preprocesses raw email text: URL/email normalization, punctuation stripping, lowercasing
- Tokenizes and embeds text for sequence-based deep learning classification
- Outputs a phishing probability, a confidence score, and flagged suspicious terms for each prediction
- Achieves 91% accuracy on a held-out test set

## Model Performance

Safe emails: 94% precision, 92% recall
Phishing emails: 88% precision, 90% recall
Overall accuracy: 91% on 3,727 test emails

## Tech Stack

- Python 3.11
- TensorFlow / Keras (LSTM architecture)
- scikit-learn (train/test split, evaluation metrics)
- Hugging Face datasets (data loading)
- pandas / numpy

## Dataset

Phishing Email Dataset (Hugging Face) - a mirrored, cleaned copy of a Kaggle dataset combining Enron, Ling, CEAS, Nazario, Nigerian, and SpamAssassin email corpora. 18,650 emails total (11,322 legitimate, 7,328 phishing).
https://huggingface.co/datasets/zefang-liu/phishing-email-dataset

## Architecture

Raw Email Text -> Text Cleaning -> Tokenization (10,000 word vocabulary) -> Embedding Layer (128 dim) -> LSTM Layer (64 units) -> Dense Layers -> Phishing Probability + Confidence + Flagged Terms

## Setup

1. Install dependencies:

python3 -m venv venv
source venv/bin/activate
pip install tensorflow pandas numpy scikit-learn datasets

2. Download the dataset:

python3 01_download_data.py

3. Train the model (takes a few minutes):

python3 02_train_model.py

4. Classify new emails:

python3 03_classify_email.py

## Example Output

Email: "URGENT: You have won a prize! Claim your gift card now..."
Prediction: Phishing
Confidence: 82.85%
Suspicious terms found: urgent, act now, confirm, limited time, prize, bank, gift card

## What I Learned

- Building an end-to-end NLP pipeline: text cleaning, tokenization, sequence padding
- LSTM architecture for sequence classification tasks
- Debugging deep environment/dependency issues: diagnosed and fixed a low-level TensorFlow mutex crash caused by an incompatible Python build (Xcode Command Line Tools Python vs a proper Homebrew-installed Python), by rebuilding the environment from scratch on Python 3.11
- Adding explainability to a black-box model output via keyword flagging, to make predictions actionable for a security analyst rather than just a raw probability

## Next Steps

- Add attention visualization to show exactly which words the LSTM weighted most heavily
- Expand the suspicious terms list based on model misclassifications
- Build a simple web interface (Flask/Streamlit) for interactive testing
- Experiment with transformer-based models (BERT) for comparison


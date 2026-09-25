"""
Step 1: Download the phishing email dataset and save it locally as a CSV.
Dataset source: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
(a mirrored copy of the Kaggle "Phishing Email Detection" dataset)
"""

from datasets import load_dataset
import pandas as pd

print("Downloading phishing email dataset from Hugging Face...")
dataset = load_dataset("zefang-liu/phishing-email-dataset")

df = dataset["train"].to_pandas()

print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nLabel distribution:\n{df.iloc[:, -1].value_counts()}")

df.to_csv("phishing_emails.csv", index=False)
print("\nSaved to phishing_emails.csv")

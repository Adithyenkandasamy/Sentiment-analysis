import kagglehub
import pandas as pd
import pytesseract
from PIL import Image
import re
from textblob import TextBlob
import os

# Step 1: Download the Kaggle dataset
path = kagglehub.dataset_download("danushkumarv/sentiment-analysis-in-tamilenglish-text")
print("Dataset downloaded at:", path)

# Step 2: Load the dataset
dataset_path = os.path.join(path, "Tamil_sentiments.csv")  # Correct path handling
df = pd.read_csv(dataset_path)  # Load dataset into pandas DataFrame
print("Dataset loaded successfully!")

# Step 3: Extract text from an Instagram comment screenshot
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)  # OCR extraction
    return extracted_text.strip()

# Step 4: Clean text before analysis
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower().strip()

# Step 5: Analyze sentiment using TextBlob (can be replaced with Kaggle dataset-based model)
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Step 6: Process Instagram comment screenshot
screenshot_path = "insta_comments.png"  # Change this to your actual file

# Extract comments
extracted_comments = extract_text_from_image(screenshot_path)
comments_list = extracted_comments.split("\n")  # Split into separate comments

# Analyze each comment
for comment in comments_list:
    clean_comment = clean_text(comment)
    if clean_comment:  # Ignore empty comments
        sentiment = analyze_sentiment(clean_comment)
        print(f"Comment: {comment} | Sentiment: {sentiment}")

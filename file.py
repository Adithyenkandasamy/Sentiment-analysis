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

# Step 2: Load the dataset (Fixing delimiter issue)
dataset_path = os.path.join(path, "Tamil_sentiments.csv")  # Ensure correct file path
print("Dataset path:", dataset_path)  # Debugging

try:
    df = pd.read_csv(dataset_path, sep=None, engine="python", on_bad_lines="skip")
    print("Dataset loaded successfully!")
    print(df.head())  # Display first few rows
except Exception as e:
    print("Error loading dataset:", e)

# Step 3: Ensure Tesseract is configured correctly
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Adjust path as needed

# Step 4: Extract text from an Instagram comment screenshot
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)  # OCR extraction
        return extracted_text.strip()
    except Exception as e:
        print("Error extracting text from image:", e)
        return ""

# Step 5: Clean text before analysis
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower().strip()

# Step 6: Analyze sentiment using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Step 7: Process Instagram comment screenshot
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

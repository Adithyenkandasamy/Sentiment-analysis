import kagglehub
import pandas as pd
import pytesseract
from PIL import Image
import re
from textblob import TextBlob
import os
import cv2
import numpy as np

# Step 1: Download the Kaggle dataset
path = kagglehub.dataset_download("danushkumarv/sentiment-analysis-in-tamilenglish-text")
print("Dataset downloaded at:", path)

# Step 2: Load the dataset (Fixing delimiter issue)
dataset_path = os.path.join(path, "Tamil_sentiments.csv")  
print("Dataset path:", dataset_path)

try:
    df = pd.read_csv(dataset_path, sep=None, engine="python", on_bad_lines="skip")
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)

# Step 3: Ensure Tesseract is configured correctly
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Step 4: Improve OCR accuracy with image preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (0, 0), fx=2, fy=2)  # Scale up
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

def extract_text_from_image(image_path):
    try:
        processed_image = preprocess_image(image_path)
        extracted_text = pytesseract.image_to_string(processed_image, lang="eng+tam")
        return extracted_text.strip()
    except Exception as e:
        print("Error extracting text from image:", e)
        return ""

# Step 5: Clean text before analysis
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s\u0B80-\u0BFF]', '', text)  # Allow Tamil characters
    return text.lower().strip()

# Step 6: Analyze sentiment using TextBlob
def analyze_sentiment(text):
    if not text:
        return "Neutral"
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Step 7: Process Instagram comment screenshot
screenshot_path = "insta_comments.png"

# Extract comments
extracted_comments = extract_text_from_image(screenshot_path)
comments_list = extracted_comments.split("\n")

# Analyze each comment
for comment in comments_list:
    clean_comment = clean_text(comment)
    if clean_comment:  
        sentiment = analyze_sentiment(clean_comment)
        print(f"Comment: {comment} | Sentiment: {sentiment}")

from flask import Flask, render_template, request, redirect, url_for
import kagglehub
import pandas as pd
import pytesseract
from PIL import Image
import re
import os
import cv2
import numpy as np
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded file
    screenshot_path = os.path.join('uploads', file.filename)
    file.save(screenshot_path)

    # Process the image and analyze sentiment
    extracted_comments = extract_text_from_image(screenshot_path)
    comments_list = extracted_comments.split("\n")
    
    results = []
    for comment in comments_list:
        clean_comment = clean_text(comment)
        if clean_comment:  
            sentiment = analyze_sentiment(clean_comment)
            results.append(f"Comment: {comment} | Sentiment: {sentiment}")

    return render_template('results.html', results=results)
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

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s\u0B80-\u0BFF]', '', text)  # Allow Tamil characters
    return text.lower().strip()

def analyze_sentiment(text):
    if not text:
        return "Neutral"
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

if __name__ == '__main__':
    app.run(debug=True)
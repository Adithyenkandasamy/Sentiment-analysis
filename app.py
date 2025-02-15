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

# Add the existing functions here (dataset download, image preprocessing, text extraction, cleaning, and sentiment analysis)

if __name__ == '__main__':
    app.run(debug=True)
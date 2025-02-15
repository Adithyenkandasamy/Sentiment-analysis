import os
import cv2
import re
import pytesseract
import numpy as np
from PIL import Image
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage, SystemMessage
from azure.core.credentials import AzureKeyCredential

# Azure GPT-4o Configuration
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"
token = os.getenv("GITHUB_TOKEN")  # Ensure you have set GITHUB_TOKEN

if not token:
    raise ValueError("GITHUB_TOKEN environment variable is not set.")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def preprocess_image(image_path):
    """Preprocesses the image for better text extraction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (0, 0), fx=2, fy=2)  # Scale up
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

def extract_text_from_image(image_path):
    """Extracts text from the given image using OCR (English + Tamil support)."""
    try:
        processed_image = preprocess_image(image_path)
        extracted_text = pytesseract.image_to_string(processed_image, lang="eng+tam")
        return extracted_text.strip()
    except Exception as e:
        print("Error extracting text from image:", e)
        return ""

def clean_text(text):
    """Cleans the extracted text by removing unwanted characters."""
    return re.sub(r'[^a-zA-Z0-9\s\u0B80-\u0BFF]', '', text).strip()

def get_feelings_from_gpt(text):
    """Sends extracted text to GPT-4o and returns only the feeling (e.g., Happy, Angry, etc.)."""
    if not text:
        return "Neutral"
    
    response = client.chat_completions.create(
        model=model_name,
        messages=[
            SystemMessage(content="You are an AI that extracts only the main feeling from a given text."),
            UserMessage(content=f"Analyze this text and return only one word for the dominant feeling: {text}")
        ],
        temperature=0.5,
        max_tokens=10
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip()

    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        exit()

    extracted_text = extract_text_from_image(image_path)
    cleaned_text = clean_text(extracted_text)

    feeling = get_feelings_from_gpt(cleaned_text)
    print("Detected Feeling:", feeling)

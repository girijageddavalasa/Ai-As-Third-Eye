import torch
import nltk
import re
import pytesseract
import cv2
import numpy as np
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertForMaskedLM

# Set correct Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Girija Pro\TesseractOCR\tesseract.exe'

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()

# Load BERT model & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

# Load and preprocess image
image_path = "traffic.jpg"
img = cv2.imread(image_path)

# Convert to grayscale & apply OTSU thresholding (Ensuring same output as original)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Processed Image", thresh)
cv2.waitKey(0)  # Wait until you press a key
cv2.destroyAllWindows()

# Convert processed image to PIL format for OCR
processed_image = Image.fromarray(thresh)

# Extract text using Tesseract
raw_text = pytesseract.image_to_string(processed_image, lang='eng')

print("\n🔍 Raw Extracted Text:\n", raw_text)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    lemmatized_text = " ".join([lemmatizer.lemmatize(token) for token in tokens])  # Lemmatization
    return lemmatized_text

cleaned_text = preprocess_text(raw_text)
print("\n🔧 Preprocessed Text:\n", cleaned_text)

# BERT-based text correction
def correct_text_with_bert(text):
    words = text.split()
    
    if len(words) < 2:
        return text

    # Keep only actual words in mask logic (fixing previous issue)
    masked_text = " ".join(["[MASK]" if not word.isalnum() else word for word in words])

    print("\n🔹 Masked Text:", masked_text)  # Debugging line

    inputs = tokenizer(masked_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits
    mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    if mask_index.numel() == 0:
        return text  

    # Replace all masked words at once
    predicted_ids = torch.argmax(predictions[0, mask_index], dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    for i, idx in enumerate(mask_index):
        words[idx] = predicted_tokens[i]  

    return " ".join(words)

# Correct the text using BERT
corrected_text = correct_text_with_bert(cleaned_text)
print("\n✅ BERT Corrected Text:\n", corrected_text)

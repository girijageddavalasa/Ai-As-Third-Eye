import cv2
import pytesseract
import numpy as np
import os

# Set correct Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Girija Pro\TesseractOCR\tesseract.exe'

def capture_and_extract():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Camera not working")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        
        if text:
            print("\n✅ Text detected:", text)
            cap.release()
            cv2.destroyAllWindows()
            return text, gray  # Return both text and processed frame
        
        cv2.imshow("Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None, None

# Step 1: Capture and detect text
text_detected, processed_frame = capture_and_extract()

# Step 2: Only proceed if text is found
if text_detected:
    import torch
    import nltk
    import re
    import pygame  # For audio playback
    from PIL import Image
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from transformers import BertTokenizer, BertForMaskedLM
    from gtts import gTTS  # Text-to-Speech

    nltk.download('punkt')
    nltk.download('wordnet')
    
    # Initialize NLP tools
    lemmatizer = WordNetLemmatizer()

    # Load BERT model & tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

    # Convert frame to PIL format for OCR
    processed_image = Image.fromarray(processed_frame)

    # Re-extract text using Tesseract
    raw_text = pytesseract.image_to_string(processed_image, lang='eng')
    print("\n🔍 Raw Extracted Text:", raw_text)

    # Text preprocessing
    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
        tokens = word_tokenize(text.lower())
        return " ".join([lemmatizer.lemmatize(token) for token in tokens])

    cleaned_text = preprocess_text(raw_text)
    print("\n🔧 Preprocessed Text:", cleaned_text)

    # BERT-based text correction
    def correct_text_with_bert(text):
        words = text.split()
        if len(words) < 2:
            return text

        masked_text = " ".join(["[MASK]" if not word.isalnum() else word for word in words])
        inputs = tokenizer(masked_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = outputs.logits
        mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        
        if mask_index.numel() == 0:
            return text
        
        predicted_ids = torch.argmax(predictions[0, mask_index], dim=-1)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)
        
        corrected_words = words[:]
        for i, idx in enumerate(mask_index.tolist()):
            if idx < len(corrected_words):
                corrected_words[idx] = predicted_tokens[i]

        return " ".join(corrected_words)

    corrected_text = correct_text_with_bert(cleaned_text)
    print("\n✅ BERT Corrected Text:", corrected_text)

    # Convert corrected text to speech
    audio_file = os.path.abspath("corrected_text.mp3")
    tts = gTTS(text=corrected_text, lang='en')
    tts.save(audio_file)

    # Play the audio file using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue
    
    pygame.mixer.quit()
    os.remove(audio_file)
    print("🗑️ Audio file deleted after playback.")
else:
    print("⚠️ No text detected. Exiting.")

import streamlit as st
import numpy as np
import pytesseract
import cv2
import urllib
from PIL import Image
from clearml import Task

# Initialize ClearML Task
task = Task.init(project_name='OCR Streamlit', task_name='Text Extraction')

def ocr(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Extract text from each column
    texts = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        column_image = image[y:y+h, x:x+w]
        text = ocr(column_image)
        texts.append(text)

    return "\n".join(texts)

# Create a Streamlit app
st.title("Multi-Column OCR Text Extraction")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    text = extract_text(image)
    task.get_logger().report_text("Extracted Text", text)
    st.write(text)

import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
from glob import glob
from numpy import random
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = '7843011691:AAG99Q1KGx70DKBb6r8EF__9_vBsSlj1e6c'  # Replace with your bot token
CHAT_ID = '6723260132'  # Replace with your chat ID

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )
    
    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}
    
    # Count occurrences of each class
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'
        if v > 1:
            prediction_text += 's'
        prediction_text += ', '

    prediction_text = prediction_text[:-2] if class_counts else "No objects detected"

    # Calculate inference latency
    latency = round(sum(res[0].speed.values()) / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    # Convert result image to RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
    
    return res_image, prediction_text

# Function to send image via Telegram
def send_to_telegram(image, caption, bot_token, chat_id):
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    
    # Create the payload for the Telegram request
    files = {'photo': ('image.png', image_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption}
    
    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            st.success("Image sent to Telegram successfully!")
        else:
            st.error(f"Failed to send image to Telegram: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error sending image to Telegram: {e}")

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        layout="centered",
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #1e1e1e;
            color: white;
            font-family: Arial, sans-serif;
        }
        h1, h2, h3 {
            color: #ff4500;
        }
        .centered {
            text-align: center;
        }
        .description {
            padding: 10px;
            border-radius: 8px;
            background-color: #2f2f2f;
            margin-bottom: 20px;
        }
        .prediction {
            padding: 10px;
            border-radius: 5px;
            background-color: #3b3b3b;
        }
    </style>
    """, unsafe_allow_html=True)

    # App title and logo
    st.markdown("<h1 class='centered'>🔥 Wildfire Detection</h1>", unsafe_allow_html=True)
    
    # Add your logo
    st.image("path_to_logo/logo.png", use_column_width=True)

    # Description section
    st.markdown(
        """
        <div class='description'>
        <h2 class='centered'>Detect Wildfires Early!</h2>
        <p class='centered'>Use our advanced AI model to predict wildfires based on real-time imagery. Simply upload an image or provide a URL, adjust the detection thresholds, and let the model do the rest!</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Sidebar info
    st.sidebar.markdown("**Developed by Siddharth Vats and Khayati Sharma**")

    # Model selection
    model_type = st.selectbox("Model Type", ("Fire Detection", "General"), index=0)
    models_dir = "fire-models" if model_type == "Fire Detection" else "general-models"
    
    # Confidence threshold
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5)

    # Image input section
    image_source = st.radio("Select Image Source", ("Upload", "URL"))
    
    image = None
    if image_source == "Upload":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        image_url = st.text_input("Image URL")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
            except:
                st.error("Failed to load image from URL.")
    
    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Run prediction
        with st.spinner("Detecting..."):
            prediction_image, text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(prediction_image, caption="Prediction Results")
            st.success(text)
        
        # Telegram Alert
        if 'fire' in text.lower():
            send_to_telegram(image, text, TELEGRAM_BOT_TOKEN, CHAT_ID)

if __name__ == "__main__":
    main()

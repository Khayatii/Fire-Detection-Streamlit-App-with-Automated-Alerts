import streamlit as st
import cv2
from ultralytics import YOLO
import requests
from PIL import Image
import os
from glob import glob
from numpy import random
import io

# Import necessary modules for live webcam feed
import time

# Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = 'your_bot_token'
CHAT_ID = 'your_chat_id'

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

    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
    
    return res_image, prediction_text

# Function to send image via Telegram
def send_to_telegram(image, caption, bot_token, chat_id):
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    
    files = {'photo': ('image.png', image_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption}
    
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        st.success("Image sent to Telegram successfully!")
    else:
        st.error(f"Failed to send image to Telegram: {response.status_code} - {response.text}")

# Function to process live camera feed
def process_live_feed(model, conf_threshold, iou_threshold):
    stframe = st.empty()
    
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is usually the default camera)
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            st.error("Failed to capture image from camera.")
            break

        # Process the frame for fire/smoke detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction, text = predict_image(model, frame_rgb, conf_threshold, iou_threshold)
        
        # Display the prediction
        stframe.image(prediction, caption="Live Camera Feed Prediction", use_column_width=True)

        # Send to Telegram if fire or smoke is detected
        if 'fire' in text.lower() or 'smoke' in text.lower():
            prediction_pil = Image.fromarray(prediction)
            send_to_telegram(prediction_pil, text, TELEGRAM_BOT_TOKEN, CHAT_ID)

        # Break the loop if Streamlit app is stopped
        time.sleep(1)

    cap.release()

def main():
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )
    
    st.sidebar.markdown("Developed by Siddharth Vats and Khayati Sharma")

    st.markdown("<div class='title'>Wildfire Detection</div>", unsafe_allow_html=True)
    
    # Sidebar information
    model_type = st.sidebar.radio("Select Model Type", ("Fire Detection", "General"))

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]

    selected_model = st.sidebar.selectbox("Select Model Size", sorted(model_files), index=2)
    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    # Set thresholds
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    # Image source selection (including live feed)
    image_source = st.radio("Select image source:", ("Upload from Computer", "Enter URL", "Live Camera Feed"))

    if image_source == "Upload from Computer":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    elif image_source == "Enter URL":
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
                    image = None
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    elif image_source == "Live Camera Feed":
        process_live_feed(model, conf_threshold, iou_threshold)

    if image:
        with st.spinner("Detecting..."):
            prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(prediction, caption="Prediction", use_column_width=True)
            st.success(text)

            prediction = Image.fromarray(prediction)
            image_buffer = io.BytesIO()
            prediction.save(image_buffer, format='PNG')

            st.download_button(
                label='Download Prediction',
                data=image_buffer.getvalue(),
                file_name='prediction.png',
                mime='image/png'
            )

            if 'fire' in text.lower() or 'smoke' in text.lower():
                image_buffer.seek(0)
                send_to_telegram(prediction, text, TELEGRAM_BOT_TOKEN, CHAT_ID)


if __name__ == "__main__":
    main()

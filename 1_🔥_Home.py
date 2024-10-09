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
TELEGRAM_BOT_TOKEN = 'your_bot_token_here'
CHAT_ID = 'your_chat_id_here'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    res = model.predict(image, conf=conf_threshold, iou=iou_threshold, device='cpu')
    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}
    
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}' + ('s' if v > 1 else '') + ', '
    prediction_text = prediction_text[:-2] if class_counts else "No objects detected"

    latency = sum(res[0].speed.values())  # in ms, convert to seconds
    prediction_text += f' in {round(latency / 1000, 2)} seconds.'

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
    
    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            st.success("Image sent to Telegram successfully!")
        else:
            st.error(f"Failed to send image to Telegram: {response.status_code}")
    except Exception as e:
        st.error(f"Error sending image to Telegram: {e}")

def main():
    st.set_page_config(page_title="Wildfire Detection", page_icon="🔥", initial_sidebar_state="collapsed")
    st.sidebar.markdown("Developed by Alim Tleuliyev")

    model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)
    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    model_path = os.path.join(models_dir, selected_model + ".pt")  # type: ignore
    model = load_model(model_path)

    # Image selection
    image = None
    image_source = st.radio("Select image source:", ("Enter URL", "Upload from Computer"))
    
    if image_source == "Upload from Computer":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
            except UnidentifiedImageError:
                st.error("Cannot identify image file. Please upload a valid image.")
    
    elif image_source == "Enter URL":
        image_url = st.text_input("Enter Image URL")
        if image_url:
            try:
                image = Image.open(requests.get(image_url, stream=True).raw)
            except Exception as e:
                st.error(f"Error fetching image from URL: {e}")

    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # New button to analyze the image
        if st.button("Analyze Image"):
            res_image, prediction_text = predict_image(model, image, conf_threshold=0.2, iou_threshold=0.5)
            st.image(res_image, caption="Detection Results", use_column_width=True)
            st.success(prediction_text)

            if st.button("Send to Telegram"):
                send_to_telegram(res_image, prediction_text, TELEGRAM_BOT_TOKEN, CHAT_ID)

if __name__ == "__main__":
    main()

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
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'  # Replace with your bot token
CHAT_ID = 'YOUR_CHAT_ID'  # Replace with your chat ID

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)  # Corrected: No need for weights_only=True
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

# Function to capture from webcam
def capture_webcam(model, conf_threshold, iou_threshold):
    st.markdown("<h3>Live Webcam Detection</h3>", unsafe_allow_html=True)

    # Try different camera indices if needed
    cam_index = 0
    cap = cv2.VideoCapture(cam_index)  # Start with index 0

    if not cap.isOpened():
        st.error("Could not open webcam. Trying different index...")
        cam_index = 1
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        st.error(f"Failed to open the camera at index {cam_index}. Ensure it's connected.")
        return

    # Capture and process frames from webcam
    stframe = st.empty()  # Placeholder for webcam feed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        # Convert frame to RGB (for model processing)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction_image, prediction_text = predict_image(model, image_rgb, conf_threshold, iou_threshold)

        # Display the frame with detections
        stframe.image(prediction_image, channels="RGB", use_column_width=True)
        st.write(prediction_text)

        # Send image to Telegram if fire or smoke detected
        if 'fire' in prediction_text.lower() or 'smoke' in prediction_text.lower():
            prediction_pil = Image.fromarray(prediction_image)
            send_to_telegram(prediction_pil, prediction_text, TELEGRAM_BOT_TOKEN, CHAT_ID)

    cap.release()

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )
    
    # Sidebar information
    st.sidebar.markdown("Developed by Siddharth Vats and Khayati Sharma")

    # Set custom CSS styles
    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 35px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .description {
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App title
    st.markdown("<div class='title'>Wildfire Detection</div>", unsafe_allow_html=True)

    # Logo and description
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        logos = glob('dalle-logos/*.png')
        logo = random.choice(logos)
        st.image(logo, use_column_width=True, caption="Generated by DALL-E")
    with col3:
        st.write("")

    st.sidebar.image(logo, use_column_width=True, caption="Generated by DALL-E")

    # Description
    st.markdown(
    """
    <div style='text-align: center;'>
        <h2>🔥 <strong>Wildfire Detection App</strong></h2>
        <p>Welcome to our Wildfire Detection App! Powered by the <a href='https://github.com/ultralytics/ultralytics'>YOLOv8</a> detection model trained on the <a href='https://github.com/gaiasd/DFireDataset'>D-Fire: an image dataset for fire and smoke detection</a>.</p>
        <h3>🌍 <strong>Preventing Wildfires with Computer Vision</strong></h3>
        <p>Our goal is to prevent wildfires by detecting fire and smoke in images with high accuracy and speed.</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Add a separator
    st.markdown("---")

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    
    with col2:
        selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    # Load the selected model
    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    # Add a section divider
    st.markdown("---")

    # Set confidence and IOU thresholds
    col1, col2 = st.columns(2)
    with col2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    with col1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    # Add a section divider
    st.markdown("---")

    # Webcam button to start capturing
    if st.button("Use Webcam for Fire Detection"):
        capture_webcam(model, conf_threshold, iou_threshold)

if __name__ == "__main__":
    main()

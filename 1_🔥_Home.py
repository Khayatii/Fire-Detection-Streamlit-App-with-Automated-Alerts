import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
from numpy import random
import io
import time

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

# Function to process live camera feed
def process_live_feed(model, conf_threshold, iou_threshold):
    stframe = st.empty()

    # Open the default webcam (ID 0)
    cap = cv2.VideoCapture(0)  # Use 1 if 0 doesn't work
    if not cap.isOpened():
        st.error("Unable to access the webcam. Please check your camera settings.")
        return

    # Check camera properties, you can add messages for resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.info(f"Webcam resolution: {width}x{height}")

    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            st.error("Failed to capture image from the webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction, text = predict_image(model, frame_rgb, conf_threshold, iou_threshold)
        
        # Display the prediction on live feed
        stframe.image(prediction, caption="Live Camera Feed Prediction", use_column_width=True)

        # If fire/smoke detected, send to Telegram
        if 'fire' in text.lower() or 'smoke' in text.lower():
            prediction_pil = Image.fromarray(prediction)
            send_to_telegram(prediction_pil, text, TELEGRAM_BOT_TOKEN, CHAT_ID)

        time.sleep(1)

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
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
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
        st.image("logo.png", use_column_width=True)  # Add your logo file here
    with col3:
        st.write("")

    # Remove URL option and keep only upload from computer and live feed
    st.markdown("---")
    image_source = st.radio("Select image source:", ("Upload from Computer", "Use Webcam Live Feed"))
    
    if image_source == "Upload from Computer":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image:
                # Display the uploaded image
                with st.spinner("Detecting"):
                    prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
                    st.image(prediction, caption="Prediction", use_column_width=True)
                    st.success(text)
                
                prediction = Image.fromarray(prediction)
            
                # Create a BytesIO object to temporarily store the image data
                image_buffer = io.BytesIO()
            
                # Save the image to the BytesIO object in PNG format
                prediction.save(image_buffer, format='PNG')
            
                # Create a download button for the image
                st.download_button(
                    label='Download Prediction',
                    data=image_buffer.getvalue(),
                    file_name='prediction.png',
                    mime='image/png'
                )
            
                # Automatically send the image to Telegram if fire or smoke is detected
                if 'fire' in text.lower() or 'smoke' in text.lower():  # Check for fire or smoke in prediction text
                    # Reset the buffer position to the beginning
                    image_buffer.seek(0)
                    send_to_telegram(prediction, text, TELEGRAM_BOT_TOKEN, CHAT_ID)

    elif image_source == "Use Webcam Live Feed":
        # Confidence and IOU threshold settings
        col1, col2 = st.columns(2)
        with col2:
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
        with col1:
            iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
        
        # Load the selected model
        model_type = "fire-models"  # Assume we are only using fire detection model
        selected_model = "fire_model.pt"  # Replace with your model file name
        model_path = os.path.join(model_type, selected_model)
        model = load_model(model_path)
        
        # Process the live camera feed
        process_live_feed(model, conf_threshold, iou_threshold)


if __name__ == "__main__":
    main()

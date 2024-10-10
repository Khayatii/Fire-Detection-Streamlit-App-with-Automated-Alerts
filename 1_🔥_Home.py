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

def process_webcam_feed(model, conf_threshold, iou_threshold):
    cap = cv2.VideoCapture(0)  # Open webcam (index 0)

    if not cap.isOpened():
        st.error("Could not open webcam. Please check your connection and permissions.")
        return

    stframe = st.empty()  # Create a placeholder for live feed

    # Loop to process frames from the webcam
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(frame_rgb)  # Convert to PIL image

        # Make predictions on the current frame
        prediction, text = predict_image(model, image, conf_threshold, iou_threshold)

        # Display the current frame with predictions
        stframe.image(prediction, channels="RGB", use_column_width=True)

        # If fire or smoke is detected, send an alert
        if 'fire' in text.lower() or 'smoke' in text.lower():
            st.warning("Wildfire detected!")
            prediction_image = Image.fromarray(prediction)
            send_to_telegram(prediction_image, text, TELEGRAM_BOT_TOKEN, CHAT_ID)

        # Stop the webcam feed if the user clicks the "Stop" button
        if st.button("Stop Webcam Feed"):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

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

    # Description
    st.markdown(
    """
    <div style='text-align: center;'>
        <h2>🔥 <strong>Wildfire Detection App</strong></h2>
        <p>Detect wildfires using live webcam feed or uploaded images with YOLOv8.</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    
    with col2:
        selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    # Load the selected model
    model_path = os.path.join(models_dir, selected_model + ".pt")  # type: ignore
    model = load_model(model_path)

    # Set confidence and IOU thresholds
    col1, col2 = st.columns(2)
    with col2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    with col1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    # Option to use webcam or upload an image
    image_source = st.radio("Select Image Source:", ("Webcam", "Upload an Image"))

    if image_source == "Webcam":
        st.write("Click the button to start the live webcam feed.")
        if st.button("Start Webcam Feed"):
            process_webcam_feed(model, conf_threshold, iou_threshold)

    else:
        # Image selection for upload
        image = None
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

        if image:
            # Display the uploaded image
            with st.spinner("Detecting..."):
                prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
                st.image(prediction, caption="Prediction", use_column_width=True)
                st.success(text)

            prediction = Image.fromarray(prediction)
            # Create a BytesIO object to temporarily store the image data
            image_buffer = io.BytesIO()
            prediction.save(image_buffer, format='PNG')

            # Create a download button for the image
            st.download_button(
                label='Download Prediction',
                data=image_buffer.getvalue(),
                file_name='prediction.png',
                mime='image/png'
            )

            # Automatically send the image to Telegram if fire or smoke is detected
            if 'fire' in text.lower() or 'smoke' in text.lower():
                image_buffer.seek(0)
                send_to_telegram(prediction, text, TELEGRAM_BOT_TOKEN, CHAT_ID)

if __name__ == "__main__":
    main()

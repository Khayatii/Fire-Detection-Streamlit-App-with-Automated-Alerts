import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
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
        initial_sidebar_state="collapsed",
    )

    # Sidebar information
    st.sidebar.markdown("Developed by Siddharth Vats and Khayati Sharma")

    # App title
    st.markdown("<h1 style='text-align: center;'>Wildfire Detection</h1>", unsafe_allow_html=True)

    # Model selection
    model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)
    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    # Load the selected model
    model_path = os.path.join(models_dir, selected_model + ".pt")  # type: ignore
    model = load_model(model_path)

    # Set confidence and IOU thresholds
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    # Create a video capture object
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    # Initialize a session state for capturing frames
    if 'frame' not in st.session_state:
        st.session_state.frame = None

    st.markdown("## Live Feed")

    # Display webcam feed
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        # Store the current frame in the session state
        st.session_state.frame = frame

        # Display the current frame
        st.image(frame, channels="BGR", use_column_width=True)

        # Predict the image
        prediction, text = predict_image(model, frame, conf_threshold, iou_threshold)

        # Convert the prediction to an image
        prediction_image = Image.fromarray(prediction)

        # Check if fire or smoke is detected
        if 'fire' in text.lower() or 'smoke' in text.lower():
            send_to_telegram(prediction_image, text, TELEGRAM_BOT_TOKEN, CHAT_ID)

        # Allow Streamlit to rerun the loop
        st.experimental_rerun()

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

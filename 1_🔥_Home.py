import streamlit as st
import cv2
from PIL import Image
import requests
import io

# Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'
CHAT_ID = 'YOUR_CHAT_ID'

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

# Function to capture webcam feed
def capture_webcam_feed():
    cap = cv2.VideoCapture(0)  # Use default webcam
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your connection and permissions.")
        st.write("Debug Info: Try changing the VideoCapture index (0, 1, 2, ...)")
        return None
    
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from webcam.")
        return None
    
    # Convert the captured frame (BGR to RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def main():
    st.title("Wildfire Detection with Webcam")
    
    # Option to choose between uploading an image or using the webcam
    option = st.radio("Choose input source:", ("Upload Image", "Use Webcam"))
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Image uploaded successfully!")
    else:
        st.write("Starting webcam...")
        # Capture and display webcam feed
        frame = capture_webcam_feed()
        if frame is not None:
            st.image(frame, caption="Webcam Feed", use_column_width=True)
            # Convert frame (numpy array) to PIL Image for Telegram
            frame_pil = Image.fromarray(frame)
            
            # Example: Automatically send an image to Telegram if webcam feed is working
            if st.button("Send Webcam Image to Telegram"):
                send_to_telegram(frame_pil, "Webcam snapshot", TELEGRAM_BOT_TOKEN, CHAT_ID)

if __name__ == "__main__":
    main()

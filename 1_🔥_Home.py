import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import os
from glob import glob
import numpy as np
from twilio.rest import Client

# Twilio configuration
TWILIO_ACCOUNT_SID = 'ACd8988f7f99fcd0726af48da6e181f789'  # Your Account SID
TWILIO_AUTH_TOKEN = '3878aa7da7b0e634e010116cc48307b5'  # Your Auth Token
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'  # Your Twilio Sandbox WhatsApp number

# Function to send WhatsApp messages
def send_whatsapp_message(body, to):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        body=body,
        to=to
    )
    return f"Message sent: {message.sid}"

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Main function
def main():
    st.set_page_config(page_title="Wildfire Detection", page_icon="🔥")
    
    st.title("Wildfire Detection App")
    
    # Model selection
    model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)
    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]

    selected_model = st.selectbox("Select Model Size", sorted(model_files))
    model_path = os.path.join(models_dir, selected_model)
    model = load_model(model_path)

    # Other app logic here...

if __name__ == "__main__":
    main()

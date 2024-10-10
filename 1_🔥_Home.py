import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Function to use webcam
def capture_webcam_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your connection and permissions.")
        st.write("Debug Info: Try changing the VideoCapture index (0, 1, 2, ...)")
        return None
    
    # Capture the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from webcam.")
        return None
    
    # Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cap.release()
    return frame_rgb

def main():
    st.title("Wildfire Detection using Webcam")

    # Sidebar information
    st.sidebar.markdown("Developed by Siddharth Vats and Khayati Sharma")

    # Select webcam feed
    if st.button("Capture Webcam Feed"):
        frame = capture_webcam_feed()

        if frame is not None:
            # Show the captured image
            st.image(frame, caption="Captured Image from Webcam", use_column_width=True)
            st.success("Webcam capture successful!")
        else:
            st.warning("No image to display from the webcam.")

if __name__ == "__main__":
    main()

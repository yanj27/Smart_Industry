import streamlit as st
from PIL import Image
import io
import requests
import numpy as np

st.set_page_config(page_title="Motion Detection", layout="centered")

st.title("Motion Detection")


# File uploader
uploaded_video = st.file_uploader("Kies een video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    try:
        
        st.video(uploaded_video)

        if st.button("Maak een motion mask"):
            with st.spinner("Even geduld..."):
                response = requests.post(
                    "http://localhost:8000/create_motion_mask/",
                    files = {"file": uploaded_video}
                )

            if response.status_code == 200:
                st.success("Received motion mask video!")

                st.video(response.content)
            else:
                st.error(f"Upload failed with status code {response.status_code}")


    except Exception as e:
        st.error(f"Er is een foutje opgetreden tijdens het inladen van: {e}")
else:
    st.info("Upload eerst een video!")

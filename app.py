import streamlit as st
from utils import process_video_and_detect_accent

st.set_page_config(page_title="English Accent Detector", layout="centered")

st.title("🎙️ English Accent Detector")
st.markdown("Provide a public video URL (e.g., Youtube URL or direct .mp4) to analyse the speaker’s accent.")

video_url = st.text_input("Video URL")

if st.button("Analyse") and video_url:
    with st.spinner("Processing video and detecting accent..."):
        try:
            result = process_video_and_detect_accent(video_url)
            st.success("Accent analysis completed!")

            st.markdown(f"**Predicted Accent:** `{result['accent']}`")
            st.markdown(f"**Confidence Score:** `{result['confidence']:.2f}%`")
            if result.get("explanation"):
                st.markdown("**Explanation:**")
                st.info(result["explanation"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
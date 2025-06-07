import os
import tempfile
import requests
import yt_dlp
from pydub import AudioSegment
import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Load model and processor
MODEL_NAME = "ylacombe/accent-classifier"
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
id2label = model.config.id2label  # Maps class indices to labels

def download_video(url):
    print(f"Attempting to download video from URL: {url}")
    tmp_path = os.path.join(tempfile.gettempdir(), f"{next(tempfile._get_candidate_names())}.m4a")

    if "youtube.com" in url or "youtu.be" in url:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': tmp_path,
            'quiet': True,
            'noplaylist': True,
            'default_search': 'auto'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Using yt_dlp to download the video with format 18...")
            ydl.download([url])
    else:
        print("Direct URL detected. Attempting HTTP GET request...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed HTTP GET request with status code: {response.status_code}")
            raise Exception("Failed to download video.")
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Video successfully downloaded from direct URL.")

    return tmp_path

def extract_audio(video_path):
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    tried_formats = ["m4a", "webm", "mp4", None]  # None = auto-detect
    errors = []

    for fmt in tried_formats:
        try:
            audio = AudioSegment.from_file(video_path, format=fmt)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(audio_path, format="wav")
            return audio_path
        except Exception as e:
            errors.append(f"Format {fmt}: {str(e)}")
            continue

    raise Exception("Audio decoding failed after trying multiple formats:\n" + "\n".join(errors))

def classify_accent(audio_path):
    # Load and preprocess audio
    waveform, sr = librosa.load(audio_path, sr=16000)
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform)

    inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    print("Predicted index:", pred_idx)
    print("id2label keys:", id2label.keys())

    accent_label = id2label.get(str(pred_idx)) or id2label.get(pred_idx) or f"Unknown accent ({pred_idx})"

    return {
        "accent": accent_label,
        "confidence": round(confidence * 100, 2),
        "explanation": f"The model is {round(confidence * 100, 2)}% confident the speaker's accent is {accent_label}."
    }

def process_video_and_detect_accent(url):
    try:
        video_path = download_video(url)
        print("Video downloaded to:", video_path)

        audio_path = extract_audio(video_path)
        print("Audio extracted to:", audio_path)

        result = classify_accent(audio_path)
        print("Classification result:", result)

    except Exception as e:
        import traceback
        print("Error occurred during processing:")
        traceback.print_exc()
        raise e
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

    return result
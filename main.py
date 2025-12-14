import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
from gtts import gTTS
from transformers import BlipProcessor, BlipForConditionalGeneration

def text_to_speech_bytes(text: str, lang: str = "en") -> bytes:
    fp = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(fp)
    fp.seek(0)
    return fp.read()

st.header('A tool for classifying images for Visual Aids')

st.write('Introduction to Python - Final Project')
@st.cache_resource
def get_processor(): return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
@st.cache_resource
def load():return BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = get_processor()
model = load()
image = st.camera_input("Take a picture")
if image is not None:   
    image = Image.open(image)

    # unconditional image captioning
    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)
    text_out = (processor.decode(out[0], skip_special_tokens=True))
    st.write (text_out)

    audio = text_to_speech_bytes(text_out)
    st.audio(audio, format="audio/mp3", autoplay=True)

# pip install streamlit
import streamlit as st
#from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
#from torchvision.io import decode_image
from PIL import Image
#from torchvision.transforms import ToTensor
#import torch
import io
from gtts import gTTS

def text_to_speech_bytes(text: str, lang: str = "en") -> bytes:
    fp = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# bring in model and all
#weights = ResNet50_Weights.DEFAULT
#transforms = weights.transforms()
#model = resnet50(weights=weights)

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
    text = ""
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    text_out = (processor.decode(out[0], skip_special_tokens=True))
    st.write (text_out)

    audio = text_to_speech_bytes(text_out)
    st.audio(audio, format="audio/mp3", autoplay=True)

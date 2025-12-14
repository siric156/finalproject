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
model = YOLO("yolov8n.pt")
image = st.camera_input("Take a picture")
if image is not None:   
    image = Image.open(image)
    results = model.predict(source=image, save=False)  # save plotted images
    #t = ToTensor()
    annotated = results[0].plot()
    #image = t(image)
    st.image(annotated)
    #transformed_image = transforms(image[:, :, :])

#pass it through the model
    #prediction = model(transformed_image.unsqueeze(0))
    #pred_val = torch.argmax(prediction)
#st.write(pred_val)
    #st.write(weights.meta['categories'][pred_val])
    st.write(results)
    #audio = text_to_speech_bytes(weights.meta['categories'][pred_val])
    #st.audio(audio, format="audio/mp3", autoplay=True)

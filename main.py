# pip install streamlit
import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import decode_image
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import io
import streamlit as st
from gtts import gTTS

# bring in model and all
weights = ResNet50_Weights.DEFAULT
transforms = weights.transforms()
model = resnet50(weights=weights)

st.header('Generalization of Hot Dog not Hot Dog')

st.write('This project uses ResNet50')

image = st.camera_input("Take a picture")
if image is not None:   
    st.image(image)
    image = Image.open(image)
    t = ToTensor()
    image = t(image)

    transformed_image = transforms(image[:, :, :])

#pass it through the model
    prediction = model(transformed_image.unsqueeze(0))
    pred_val = torch.argmax(prediction)
#st.write(pred_val)
    st.write(weights.meta['categories'][pred_val])

def text_to_speech_bytes(text: str, lang: str = "en") -> bytes:
    fp = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(fp)
    fp.seek(0)
    return fp.read()

text = st.text_input("Type something")
if st.button("Speak") and text:
    audio = text_to_speech_bytes(text)
    st.audio(audio, format="audio/mp3", autoplay=True)


# pip install streamlit
import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import decode_image
from PIL import Image
from torchvision.transforms import ToTensor
import torch
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

import streamlit as st
from streamlit_TTS import text_to_speech

# Simple text-to-speech
text_to_speech("Hello! Welcome to my Streamlit app!", language='en')

# With user input
text = st.text_input("Enter text to speak:")
if st.button("Speak") and text:
    text_to_speech(text, language='en')

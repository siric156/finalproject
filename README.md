My final project interactive streamlit app aims to help visually impaired people navigate and accomodate their day to day life activities, acting if so they are a pair of eyes for those individuals
through image classification. 
The flow of this app is that you take a photo of an object or somewhere you are walking to, the app will then collect the data and it will response with a voice, stating what is in the photo taken.
For example: "A girl holding a red lipstick".
This is to help ease the pain of running into something and to know what is in front of them through sound/ audio based medium.

In the future, I think to develop this app further, it could be paired with a meta glass to help those individual navigate through their daily lifestyles. The way it would work is that the camera 
on the glass will be a pair of "eyes" for the app and the glasses leg (the one that goes onto the side of your head) will use some sort of bone vibration headphone transmitter 
for the person wearing them to hear the audio.

This app uses a pretrained model from Huggingface called Salesforce/blip-image-captioning-base and instead of using a dataset from an online source, the data source of this app comes from the image 
taken by the users in real time, supporting flexibilty for the app's users.
My process began with browsing a pretrained model which I have come across - YOLO, an image classification model and I tried integrating it into a simple user 
interface that allows users to upload images and receive real-time predictions. With after a few trys and errors, I got them to work in the app's interface, however, the results generated are 
not at all accurate. 
I then changed route and went back to browse Huggingface again and search in the image-to-audio section and came across this pretrained model that I am currently using in my project which turns 
out to be very accurate than the previous one.
Throughout the development, I encountered a lot of technical challenges and diffculties related to environment configuration, including missing dependencies such as FFmpeg 
and file path errors when loading model weights on Streamlit Cloud. 
Resolving these issues were challenging in which I have done research to understand how external system packages, Python libraries, and cloud deployment environments interact,
as well as debugging stack traces to identify the root causes of these errors.
After correcting model naming conventions, managing the file paths, and configuring the deployment environment properly, I was able to successfully load the model and run the application.

To refelct, I found this app to be quite challenging to do as it is my first time taking a coding class let alone building an app from that. I was really greatful when the app work 
and are actually able to help the visually impaired people and its accuracy is something that is really cool for me. 

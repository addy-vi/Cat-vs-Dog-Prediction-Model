import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from PIL import Image

model=tf.keras.models.load_model(r"C:\Users\haider\Desktop\New folder (3)\Cat_Dog_Prediction.keras")

st.title("Cat and dog identifier")
inputted_image=st.file_uploader("Enter the image")

if st.button("Peridect"):
    inputted_image=Image.open(inputted_image)
    inputted_image=inputted_image.convert("RGB")
    inputted_image=inputted_image.resize((224,224))
    inputted_image=np.array(inputted_image)
    inputted_image=inputted_image/255.0
    inputted_image=inputted_image.reshape(1,224,224,3)
    prediction=model.predict(inputted_image)
    pre=prediction.argmax()
    if pre==0:
        st.write("this is a dog")
    else:
        st.write("this is a cat")
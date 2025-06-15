

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import random

#  Absolute path to your dataset (update if needed)
data_dir = r'C:\Users\Aditya Kumar\Downloads\Bone Break Classification'

#  Load your trained model
model = tf.keras.models.load_model(r'C:\Users\Aditya Kumar\Downloads\bone_fracture_model.h5')

#  Class names (in the same order as used during training)
class_names = ['Avulsion Fracture', 'Comminuted Fracture', 'Fracture-Dislocation',
               'Greenstick Fracture', 'Hairline Fracture', 'Impacted Fracture',
               'Longitudinal Fracture', 'Oblique Fracture',
               'Pathological Fracture', 'Spiral Fracture']

#  Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)
    
    # Collect all image paths
image_paths = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            image_paths.append(os.path.join(root, file))
    
    # Streamlit App
st.title("Bone Fracture Classification")
    
    # Random file selection
selected_file = st.selectbox("Choose a random X-ray image", random.sample(image_paths, 10))
    
    # Streamlit interface
    # st.title("Bone Fracture Classification")
    # uploaded_file = st.file_uploader("Upload an X-ray image", type=['jpg', 'png', 'jpeg'])
    
if selected_file:
    img = Image.open(selected_file).convert('RGB')
    st.image(img, caption=f"Selected Image: {os.path.basename(selected_file)}", use_container_width=True)
    
        # Predict
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100
    
    st.markdown(f"### Predicted Fracture Type: **{class_names[predicted_index]}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")
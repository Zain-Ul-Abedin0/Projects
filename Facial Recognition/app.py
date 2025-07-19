import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the saved model
#model = load_model('model.h5')
model = load_model('model_2.h5')

# Class names (update exactly as per your dataset)
class_names = ['Cristiano_Ronaldo', 'Kylian Mbappe', 'Lamine Yamal', 'Lionel Messi', 'Neymar Junior']

# Set page config
st.set_page_config(page_title="FootBall Celebrity Face Recognition", page_icon="🤖", layout="wide")

# App title and description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚽ FootBall Celebrity Face Recognition 🤖</h1>", unsafe_allow_html=True)
st.write("Upload an image and let the AI guess the celebrity!")

# File uploader
uploaded_file = st.file_uploader("📷 Upload an image (jpg, png, jpeg):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Prepare the layout: 2 columns (one for image and description, another for prediction)
    col1, col2 = st.columns([3, 2])

    with col1:
        # Show the image on the left side
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        
        # Resize for display (smaller)
        max_width = 300
        img_resized_for_display = image.resize((max_width, int(max_width * image.height / image.width)))
        st.image(img_resized_for_display, caption='Uploaded Image', use_container_width=False)
        
        st.write("🔍 **Classifying... Please wait.**")

    with col2:
        # Prepare image for prediction
        img_array = np.array(image)
        resized_img = cv2.resize(img_array, (210, 210))
        img_batch = np.expand_dims(resized_img, axis=0)  # Add batch dimension
        img_batch = img_batch / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_batch)
        confidence = tf.nn.softmax(predictions[0]).numpy()
        predicted_class = np.argmax(confidence)
        confidence_score = confidence[predicted_class]

        # Display top result
        st.markdown(
            f"""
            <div style='padding: 15px; border-radius: 10px; background-color: #f0f2f6;'>
                <h3 style='color: #3366cc;'>🎉 Prediction: <strong>{class_names[predicted_class]}</strong></h3>
                <p style='font-size: 16px;'>Confidence: <strong>{confidence_score:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True
        )

        # Show all probabilities in a table with progress bars
        st.markdown("## 🔢 Prediction Probabilities:")

        for i in range(len(class_names)):
            st.write(f"**{class_names[i]}:** {confidence[i]:.2%}")
            st.progress(float(confidence[i]))


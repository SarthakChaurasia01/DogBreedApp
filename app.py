import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

# Load labels
labels_df = pd.read_csv("labels.csv")
breed_labels = labels_df["breed"].tolist()

# Load the model
model = tf.keras.models.load_model("dog_breed_model.h5")

# Image preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize to model input
    image_array = np.array(image) / 255.0  # Normalize
    if image_array.shape[-1] == 4:  # Remove alpha channel if present
        image_array = image_array[..., :3]
    return np.expand_dims(image_array, axis=0)

# Streamlit app
st.title("🐶 Dog Breed Classification App")
st.write("Upload a dog image and get the predicted breed!")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]

    top_index = np.argmax(prediction)
    top_breed = breed_labels[top_index]
    confidence = prediction[top_index] * 100

    st.markdown(f"### 🐾 Predicted Breed: `{top_breed}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

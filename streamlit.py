import streamlit as st
import numpy as np
import os
import requests

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Load model
# model = load_model("model.h5")
# class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_URL = "https://huggingface.co/Ayesha0303/machine-learning/resolve/main/model.h5"
MODEL_PATH = "model.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)

    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model
model = load_model(MODEL_PATH)

class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.title("Scene Classifier")
st.write("Upload an image to predict the scene type")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image

    # Open original image
    img = Image.open(uploaded_file)

    # Show ORIGINAL image clearly
    st.image(uploaded_file, caption="Uploaded Image")

    # Resize ONLY for model prediction
    img_resized = img.resize((200, 200), Image.LANCZOS)

    # Convert image to array
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    preds = model.predict(img_array, verbose=0)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.success(f"Prediction: *{predicted_class}*")
    st.write(f"Confidence: *{confidence*100:.2f}%*")

    st.write("Class probabilities:")
    for cls, prob in zip(class_names, preds):
        st.write(f"- {cls}: {prob*100:.2f}%")


























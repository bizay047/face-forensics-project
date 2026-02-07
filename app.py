import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Face Detector (OpenCV)
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Load Classification Model
# -------------------------------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deepfake_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Image Transform (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Deepfake Face Detection")
st.write("Upload an image. The face will be detected and classified as REAL or FAKE.")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        st.error("No face detected. Please upload a clear face image.")
    else:
        # Take the largest face
        x, y, w, h = sorted(
            faces, key=lambda b: b[2] * b[3], reverse=True
        )[0]

        face_crop = image.crop((x, y, x + w, y + h))
        st.image(face_crop, caption="Detected Face", width=200)

        # Preprocess face
        face_tensor = transform(face_crop).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label_map = {0: "REAL", 1: "FAKE"}

        st.subheader(f"Prediction: {label_map[pred]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

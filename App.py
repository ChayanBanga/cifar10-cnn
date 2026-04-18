import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="centered"
)

# ── Model Architecture (must match training) ─────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            ResBlock(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Constants ─────────────────────────────────────────────
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_EMOJIS = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦', 'cat': '🐱',
    'deer': '🦌', 'dog': '🐶', 'frog': '🐸', 'horse': '🐴',
    'ship': '🚢', 'truck': '🚛'
}

# ── Replace with your HuggingFace model path ─────────────
# Format: "username/repo-name"  →  file inside repo: "model.pt"
HF_MODEL_REPO = "YOUR_HF_USERNAME/cifar10-resnet"   # ← change this
HF_MODEL_FILE = "model.pt"


# ── Load Model (cached) ───────────────────────────────────
@st.cache_resource
def load_model():
    # Download from HuggingFace Hub
    url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/{HF_MODEL_FILE}"
    
    with st.spinner("Downloading model from HuggingFace..."):
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to download model. Status: {response.status_code}\n"
                     f"Make sure your HF repo '{HF_MODEL_REPO}' is public and contains '{HF_MODEL_FILE}'.")
            st.stop()
        
        model_bytes = BytesIO(response.content)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    state_dict = torch.load(model_bytes, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# ── Inference ─────────────────────────────────────────────
def predict(image: Image.Image, model, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
    
    results = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
    return results


# ── UI ────────────────────────────────────────────────────
st.title("🧠 CIFAR-10 Image Classifier")
st.markdown(
    "Upload any image and the ResNet model will classify it into one of **10 CIFAR-10 categories**."
)
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** Custom ResNet (4 blocks)  
    **Dataset:** CIFAR-10  
    **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
    **Framework:** PyTorch  
    **Hosted on:** HuggingFace Hub
    """)
    st.markdown("---")
    st.markdown("Built with Streamlit 🎈")

# Load model
model, device = load_model()
st.success(f"✅ Model loaded! Running on `{device}`")

# Upload
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Classifying..."):
            results = predict(image, model, device)
        
        top_class, top_prob = results[0]
        emoji = CLASS_EMOJIS[top_class]
        
        st.markdown(f"### {emoji} {top_class.capitalize()}")
        st.metric("Confidence", f"{top_prob * 100:.1f}%")
    
    st.markdown("---")
    st.subheader("All class probabilities")
    
    for cls, prob in results:
        e = CLASS_EMOJIS[cls]
        bar_col, label_col = st.columns([4, 1])
        with bar_col:
            st.progress(prob, text=f"{e} {cls}")
        with label_col:
            st.write(f"**{prob*100:.1f}%**")

else:
    st.info("👆 Upload an image to get started")
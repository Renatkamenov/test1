
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from unet import UNet
import gdown
import os

# === Скачивание моделей из Google Drive ===
def download_weights_if_needed():
    if not os.path.exists("model.pth"):
        gdown.download(id="1YOiR43UNS6uddwOm2dfT5XYaZKAZp1KA", output="model.pth", quiet=False)
    if not os.path.exists("model_classifier.pth"):
        gdown.download(id="12IYj0cjpMsWPeTOZMlgqZ2pucziMrutC", output="model_classifier.pth", quiet=False)

download_weights_if_needed()

# === Константы ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CLS_SIZE = 224

CLASSES = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven']
DEFECT_LABELS = {
    "MT_Blowhole": "Пористость",
    "MT_Break": "Разрыв",
    "MT_Crack": "Трещина",
    "MT_Fray": "Изношенность края",
    "MT_Free": "Без дефекта",
    "MT_Uneven": "Неровность"
}

# === Загрузка моделей ===
@st.cache_resource
def load_unet():
    model = UNet()
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("model_classifier.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# === Трансформации ===
tf_unet = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

tf_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(CLS_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# === Интерфейс ===
st.title("🧠 Поверхностные дефекты — UNet + ResNet")
st.markdown("Загрузите изображение. Модель выделит дефект и определит его тип.")

uploaded_file = st.file_uploader("📎 Загрузите .jpg/.png", type=["jpg", "jpeg", "png"])

if uploaded_file:
    orig_img = Image.open(uploaded_file).convert("RGB")

    # === UNet сегментация ===
    unet_input = tf_unet(orig_img).unsqueeze(0).to(DEVICE)
    unet = load_unet()
    with torch.no_grad():
        output = unet(unet_input)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

    # Маска + наложение
    img_resized = orig_img.resize((IMG_SIZE, IMG_SIZE))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    overlay = Image.new("RGBA", img_resized.size, (255, 0, 0, 0))
    overlay_np = np.array(overlay)
    mask_np = np.array(mask_img)
    overlay_np[mask_np > 0] = [255, 0, 0, 100]
    overlay = Image.fromarray(overlay_np, mode="RGBA")
    result_img = Image.alpha_composite(img_resized.convert("RGBA"), overlay)

    # === ResNet классификация ===
    resnet_input = tf_resnet(orig_img).unsqueeze(0).to(DEVICE)
    resnet = load_resnet()
    with torch.no_grad():
        pred = resnet(resnet_input)
        pred_class = CLASSES[pred.argmax(1).item()]
        pred_desc = DEFECT_LABELS.get(pred_class, "Неизвестный дефект")

    # === Вывод
    st.subheader("🎯 Результат сегментации:")
    st.image(result_img, caption="Обнаруженные дефекты", use_container_width=True)
    st.markdown(f"### 📌 Тип дефекта: **{pred_desc} ({pred_class})**")

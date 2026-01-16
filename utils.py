# utils.py
import os
import torch
from torchvision import transforms
from PIL import Image

from model import SiameseEfficientNet

# เลือก device
device = "cuda" if torch.cuda.is_available() else "cpu"

# transform ตอน infer
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def load_model(model_path="face_encoder.pt"):
    """โหลด weight โมเดล siamese จากไฟล์ .pt"""
    model = SiameseEfficientNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_face_db(db_path="face_db.pt"):
    """โหลด dict face_db จากไฟล์ .pt ถ้าไม่มีให้คืน {}"""
    if os.path.exists(db_path):
        db = torch.load(db_path, map_location="cpu")
    else:
        db = {}
    return db

def save_face_db(face_db, db_path="face_db.pt"):
    """บันทึก dict face_db ลงไฟล์ .pt"""
    torch.save(face_db, db_path)

def get_embedding_from_pil(img_pil, model):
    """รับรูป PIL -> คืน embedding (tensor) จากโมเดล"""
    img = inference_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat, _ = model(img, img)  # siamese forward
    return feat.cpu().squeeze(0)

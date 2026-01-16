# attendance.py
import cv2
import csv
import os
import datetime
import torch
import torch.nn.functional as F
from PIL import Image

from utils import load_model, load_face_db, get_embedding_from_pil

# 1) โหลดโมเดล + face database
model = load_model("face_encoder.pt")
face_db = load_face_db("face_db.pt")      # dict: {name: embedding tensor}

print("Loaded model and face_db:", list(face_db.keys()))

# 2) เตรียมไฟล์ CSV สำหรับเก็บเวลาเข้าเรียน/เข้างาน
csv_file = "attendance.csv"
file_exists = os.path.isfile(csv_file)

f = open(csv_file, "a", newline="", encoding="utf-8-sig")
writer = csv.writer(f)

if not file_exists:
    writer.writerow(["name", "datetime", "status"])

# ใช้ set กันลงชื่อซ้ำในรอบการรันเดียว
marked_names = set()

# 3) โหลด Haarcascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def identify_face(pil_img, threshold=0.9):
    """รับรูปหน้า (PIL) -> ทำนายว่าเป็นใคร โดยเทียบกับ face_db"""
    q_emb = get_embedding_from_pil(pil_img, model)
    best_name = "unknown"
    best_dist = 999.0

    for name, proto in face_db.items():
        dist = F.pairwise_distance(
            q_emb.unsqueeze(0),
            proto.unsqueeze(0)
        ).item()
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist > threshold:
        return "unknown", best_dist
    return best_name, best_dist

# 4) เปิดกล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    f.close()
    exit()

print("เริ่มระบบเช็คชื่อด้วยใบหน้า กด Q เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_bgr = frame[y:y+h, x:x+w]

        # แปลง BGR -> RGB -> PIL
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        name, dist = identify_face(pil_img, threshold=0.9)

        # วาดกรอบ + ชื่อ
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{name}" if name != "unknown" else "unknown"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ถ้าระบุคนได้ และยังไม่ลงเวลาในรอบนี้ -> บันทึกเวลา
        if name != "unknown" and name not in marked_names:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([name, now, "check-in"])
            f.flush()
            marked_names.add(name)
            print(f"[LOG] {name} check-in เวลา {now}")

    cv2.imshow("Face Attendance", frame)

    # กด q เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
f.close()
cv2.destroyAllWindows()
print(">>> START attendance.py")

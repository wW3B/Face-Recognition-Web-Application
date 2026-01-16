# enroll.py
import cv2
import torch
from PIL import Image
import torch.nn.functional as F

from utils import load_model, load_face_db, get_embedding_from_pil

# ===== 1) โหลดโมเดล =====
model = load_model("face_encoder.pt")

# พยายามโหลด face_db; ถ้าไม่มีไฟล์ให้ใช้ dict ว่าง
try:
    face_db = load_face_db("face_db.pt")
except FileNotFoundError:
    face_db = {}
    print("face_db.pt ยังไม่มี จะสร้างใหม่ให้")

print("มีคนในฐานข้อมูลแล้ว:", list(face_db.keys()))

# ===== 2) รับชื่อคนที่จะลงทะเบียน =====
person_name = input("กรุณากรอกชื่อ/รหัสนักศึกษา/รหัสพนักงาน: ").strip()
if person_name == "":
    print("ชื่อว่างครับ ยกเลิกการลงทะเบียน")
    exit()

print(f"เริ่มลงทะเบียนให้: {person_name}")
print("จะใช้ Webcam ในการเก็บภาพหน้า กด 'c' เพื่อเก็บภาพ, กด 'q' เพื่อจบ")

# ===== 3) เตรียมกล้อง + face detector =====
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

emb_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # วาดกรอบใบหน้าให้เห็น
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Enroll Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if len(faces) == 0:
            print("ไม่พบใบหน้าในเฟรมนี้ ลองใหม่อีกครั้ง")
            continue

        # เลือกหน้าใหญ่สุดในเฟรม
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_bgr = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        emb = get_embedding_from_pil(pil_img, model)
        emb_list.append(emb)
        print(f"เก็บภาพแล้ว {len(emb_list)} รูป")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(emb_list) == 0:
    print("ไม่ได้เก็บภาพเลย ยกเลิกการลงทะเบียน")
    exit()

# ===== 4) เฉลี่ย embedding แล้วบันทึกลง face_db =====
emb_avg = torch.stack(emb_list).mean(dim=0)
face_db[person_name] = emb_avg

torch.save(face_db, "face_db.pt")
print(f"บันทึกโปรไฟล์ของ {person_name} เรียบร้อยแล้ว!")
print("ตอนเปิด attendance.py รอบหน้า จะรู้จักคนนี้แล้ว")
print(">>> START ENROLL")

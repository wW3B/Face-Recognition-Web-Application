# app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
import datetime
import csv
import os
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io
import time

from utils import load_model, load_face_db, get_embedding_from_pil, save_face_db

print(">>> START app.py")  # debug ให้รู้ว่าถูกรัน

app = Flask(__name__)

# ===== โหลดโมเดล + face database =====
model = load_model("face_encoder.pt")
face_db = load_face_db("face_db.pt")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ===== Global state for attendance tracking =====
FRAME_SKIP = 5  # Process every 5th frame
COOLDOWN_SECONDS = 300  # 5 minutes per person
last_checkin = {}  # {name: timestamp}
today_checkins = {}  # {name: count}


def reset_daily_attendance():
    """Reset attendance if it's a new day"""
    global today_checkins, last_checkin
    csv_file = "attendance.csv"

    if not os.path.exists(csv_file):
        return

    # Check file creation date
    file_mtime = os.path.getmtime(csv_file)
    file_date = datetime.datetime.fromtimestamp(file_mtime).date()
    today = datetime.datetime.now().date()

    if file_date < today:
        today_checkins = {}
        last_checkin = {}
        print("[INFO] Daily attendance reset")


# ===== ฟังก์ชันช่วยระบุชื่อจากใบหน้า =====
def identify_face(pil_img, threshold=0.9):
    """รับรูปหน้า (PIL) -> ทำนายชื่อจาก face_db"""
    q_emb = get_embedding_from_pil(pil_img, model)
    best_name = "unknown"
    best_dist = 999.0

    for name, proto in face_db.items():
        dist = F.pairwise_distance(q_emb.unsqueeze(0), proto.unsqueeze(0)).item()
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist > threshold:
        return "unknown", best_dist
    return best_name, best_dist


def can_checkin(name):
    """Check if person can check-in (cooldown + daily limit)"""
    now = time.time()

    # Cooldown check: prevent re-check-in within COOLDOWN_SECONDS
    if name in last_checkin:
        if now - last_checkin[name] < COOLDOWN_SECONDS:
            return False

    return True


# ===== อ่านกล้อง + เช็คชื่อ แล้ว stream ออกไปหน้าเว็บ =====
def gen_frames():
    """อ่านกล้อง + เช็คชื่อ + stream วิดีโอเป็น MJPEG ให้ browser"""
    global last_checkin, today_checkins

    reset_daily_attendance()

    cap = cv2.VideoCapture(0)
    frame_count = 0

    csv_file = "attendance.csv"
    file_exists = os.path.isfile(csv_file)
    f = open(csv_file, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["name", "datetime", "status"])

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # ===== FRAME SKIPPING: Process every Nth frame =====
        if frame_count % FRAME_SKIP != 0:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face_bgr = frame[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            name, dist = identify_face(pil_img, threshold=0.9)

            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # ===== Display name + confidence distance =====
            display_text = name if name != "unknown" else f"unknown ({dist:.2f})"
            cv2.putText(
                frame,
                display_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # ===== Check-in with cooldown =====
            if name != "unknown" and can_checkin(name):
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([name, now, "check-in"])
                f.flush()
                last_checkin[name] = time.time()
                today_checkins[name] = today_checkins.get(name, 0) + 1
                print(f"[LOG] {name} check-in at {now}")

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()
    f.close()


# ===== ROUTES สำหรับหน้าเว็บ =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs")
def logs():
    rows = []
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
    return render_template("logs.html", rows=rows)


@app.route("/enroll")
def enroll():
    """หน้าเว็บสำหรับลงทะเบียนใบหน้าใหม่"""
    return render_template("enroll.html")


# ===== API ลงทะเบียนใบหน้าใหม่จากหน้าเว็บ =====
@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """รับรูปจาก browser + ชื่อ แล้วอัพเดต face_db"""
    global face_db

    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "message": "no json"}), 400

    name = (data.get("name") or "").strip()
    img_data = data.get("image")

    if not name:
        return jsonify({"ok": False, "message": "กรุณากรอกชื่อ"}), 400
    if not img_data:
        return jsonify({"ok": False, "message": "ไม่พบข้อมูลรูปภาพ"}), 400

    try:
        # img_data เป็น data URL เช่น "data:image/jpeg;base64,xxxx"
        header, encoded = img_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ใช้โมเดลเดิมสร้าง embedding
        emb = get_embedding_from_pil(pil_img, model).detach().cpu()

        # อัพเดต dict ในหน่วยความจำ + เซฟกลับไฟล์
        face_db[name] = emb
        save_face_db(face_db, "face_db.pt")

        print(f"[ENROLL] เพิ่มใบหน้าใหม่: {name}")
        return jsonify({"ok": True, "message": "ลงทะเบียนสำเร็จ"})

    except Exception as e:
        print("ENROLL ERROR:", e)
        return jsonify({"ok": False, "message": f"เกิดข้อผิดพลาด: {e}"}), 500


# ===== API ดึงรายชื่อทั้งหมดใน face_db =====
@app.route("/api/list_faces", methods=["GET"])
def api_list_faces():
    """ส่งรายชื่อคนทั้งหมดใน face_db กลับไปให้ frontend"""
    names = sorted(face_db.keys())
    return jsonify({"ok": True, "names": names})


# ===== API ลบใบหน้าจากฐานข้อมูล =====
@app.route("/api/delete_face", methods=["POST"])
def api_delete_face():
    """ลบชื่อออกจาก face_db แล้วเซฟไฟล์กลับลง face_db.pt"""
    global face_db

    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "message": "no json"}), 400

    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "message": "กรุณาระบุชื่อที่จะลบ"}), 400

    if name not in face_db:
        return jsonify({"ok": False, "message": f'ไม่พบชื่อ "{name}" ในระบบ'}), 404

    # ลบจาก dict
    del face_db[name]
    # เซฟกลับไฟล์
    save_face_db(face_db, "face_db.pt")

    print(f"[DELETE] ลบใบหน้า: {name}")
    return jsonify({"ok": True, "message": f'ลบ "{name}" เรียบร้อยแล้ว'})


# ===== API สำหรับดึงสถิติเช็คอินวันนี้ =====
@app.route("/api/stats", methods=["GET"])
def api_stats():
    """ส่งจำนวนคนที่เช็คอินวันนี้กลับไป"""
    reset_daily_attendance()
    total_checkins = len(today_checkins)
    return jsonify(
        {
            "ok": True,
            "total_checkins": total_checkins,
            "checkins": today_checkins,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )


# ===== RUN APP =====
if __name__ == "__main__":
    print(">>> FLASK APP RUNNING ON http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

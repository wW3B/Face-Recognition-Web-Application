import cv2

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if cascade.empty():
    print("❌ โหลดไม่ได้")
else:
    print("✅ โหลดสำเร็จ ใช้งานได้")

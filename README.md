#Aiตรวจจับผู้บุกรุกด้วย Opencv
from ultralytics import YOLO
import cv2
import requests
import os
from datetime import datetime
import time


#LINE Messaging API settings

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "YOU LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID", "YOU LINE_USER_ID")

def push_line_text(token: str, to_user_id: str, message: str) -> bool:
    """ส่งข้อความตัวอักษรผ่าน LINE Messaging API (push message)"""
    try:
        url = "https://api.line.me/v2/bot/message/push"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "to": to_user_id,
            "messages": [
                {"type": "text", "text": message}
            ]
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            print("ส่ง LINE ข้อความสำเร็จ")
            return True
        print(f"❌ ส่ง LINE ข้อความล้มเหลว: {resp.status_code} - {resp.text}")
        return False
    except Exception as e:
        print(f"❌ ข้อผิดพลาดส่ง LINE ข้อความ: {e}")
        return False

#ใช้โมเดลYOLOv8 โมเดลn หากรันแล้วERROR ให้ทำการติดตั้ง YOLO ก่อนนะครับ
model = YOLO("yolov8n.pt")

#เปิดกล้องใช้คำสั่งนี้ครับ
cap = cv2.VideoCapture(0)

#หากคูณจะใช้เป็นVideoให้ใช้คำสั่งนี้ครับ
cap = cv2.VideoCapture("video") 

#ตัวแปรสำหรับป้องกันการแจ้งเตือนซ้ำ
last_notify_time = 0
notify_cooldown = 5  # เว้นช่วงแจ้งเตือน 5 วินาที

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break

    # กำหนดโซนตรวจจับ
    zone_x1, zone_y1, zone_x2, zone_y2 = 0, 0, 0, 0  # <--บรรทัดนี้คำสัญ ซึ่งคำสั่งนี้จะเป็นตัวกำหนด เขตที่เมื่อมีผู้บุกรุกเข้ามาในเขตนี้จะทำการแจ้งเตือนโปรดตั้งค่าบรรทัดนี้ก่อนใช้งาน
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)

    # ตรวจจับวัตถุด้วย YOLO
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                inside_zone = (zone_x1 <= center_x <= zone_x2) and (zone_y1 <= center_y <= zone_y2)

                if inside_zone and conf > 0.5:
                    current_time = time.time()
                    if current_time - last_notify_time > notify_cooldown:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        text_msg = (
                            f"⚠️ ตรวจพบผู้บุกรุกในเขตบ้านของคุณ\n"
                            f"โปรดระวัง\n"
                            f"ตำแหน่ง: ({center_x:.1f}, {center_y:.1f})\n"
                            f"เวลา: {timestamp}\n"
                            f"ความมั่นใจ: {conf:.2f}"
                        )
                        push_line_text(LINE_CHANNEL_ACCESS_TOKEN, LINE_USER_ID, text_msg)
                        last_notify_time = current_time
                    else:
                        remaining_time = notify_cooldown - (current_time - last_notify_time)
                        print(f"รอแจ้งเตือนอีก {remaining_time:.1f} วินาที")

                # วาดกรอบ
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
                tag = f"{label} {conf:.2f}"
                if inside_zone:
                    tag += " IN-ZONE"
                cv2.putText(frame, tag, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # แสดงภาพ
    cv2.imshow("YOLO Detection", frame)

    # ออกจากโปรแกรมถ้ากด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()

# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# โหลดโมเดล (ใช้ cache เพื่อไม่โหลดซ้ำ)
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("❌ ไม่พบไฟล์ best.pt! ตรวจสอบว่าอัปโหลดถูกต้อง")
        st.stop()
    return YOLO("best.pt")

model = load_model()
VALUE_MAP = {"1baht": 1, "5baht": 5, "10baht": 10}

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="🪙 นับเหรียญไทยอัตโนมัติ",
    page_icon="🪙",
    layout="centered"
)

st.title("🪙 นับเหรียญไทยอัตโนมัติ")
st.markdown("""
อัปโหลดรูปเหรียญ **1, 5 หรือ 10 บาท**  
ระบบจะตรวจจับ → นับจำนวน → คำนวณยอดเงินรวมให้ทันที!
""")

# อัปโหลดภาพ
uploaded_file = st.file_uploader(
    "เลือกภาพเหรียญ (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # โหลดภาพ
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    with st.spinner("กำลังวิเคราะห์เหรียญ..."):
        try:
            # ทำนาย
            results = model(image, conf=0.5)
            result = results[0]

            # วาด bounding box
            plotted_img = result.plot()  # BGR
            plotted_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
            st.image(plotted_rgb, caption="ผลลัพธ์", use_column_width=True)

            # นับเหรียญ
            coin_count = {"1baht": 0, "5baht": 0, "10baht": 0}
            total_value = 0

            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                if class_name in VALUE_MAP:
                    coin_count[class_name] += 1
                    total_value += VALUE_MAP[class_name]

            # แสดงผล
            st.subheader("📊 ผลการนับเหรียญ")
            has_coin = False
            for coin, count in coin_count.items():
                if count > 0:
                    has_coin = True
                    value = VALUE_MAP[coin]
                    st.write(f"🪙 **{coin}**: {count} เหรียญ → {count * value} บาท")

            if has_coin:
                st.success(f"💰 **ยอดเงินรวม: {total_value} บาท**")
            else:
                st.warning("⚠️ ไม่พบเหรียญในภาพ กรุณาลองใหม่")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")
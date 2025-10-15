# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
VALUE_MAP = {"1baht": 1, "5baht": 5, "10baht": 10}

st.title("🪙 นับเหรียญไทยอัตโนมัติ")
uploaded = st.file_uploader("อัปโหลดภาพ", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    with st.spinner("กำลังวิเคราะห์..."):
        results = model(image, conf=0.5)  # Ultralytics รับ PIL Image ได้เลย!
        res = results[0]
        
        # แสดงผลลัพธ์ (Ultralytics จัดการการวาดให้เอง)
        plotted_img = Image.fromarray(res.plot()[:, :, ::-1])  # BGR → RGB
        st.image(plotted_img, caption="ผลลัพธ์", use_column_width=True)

        # นับเหรียญ
        count = {"1baht": 0, "5baht": 0, "10baht": 0}
        total = 0
        for box in res.boxes:
            name = model.names[int(box.cls[0])]
            if name in VALUE_MAP:
                count[name] += 1
                total += VALUE_MAP[name]

        # แสดงผล
        for coin, qty in count.items():
            if qty > 0:
                st.write(f"{coin}: {qty} เหรียญ")
        st.success(f"ยอดเงินรวม: {total} บาท")
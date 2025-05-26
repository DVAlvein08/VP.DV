
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="VP-AI", layout="wide")
st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

@st.cache_data
def load_data_and_train():
    df = pd.read_csv("Mô hình AI.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={" SpO2": "SpO2"})
    df["Tuoi"] = pd.to_numeric(df["Tuoi"], errors="coerce")
    df["Benh ngay thu truoc khi nhap vien"] = pd.to_numeric(df["Benh ngay thu truoc khi nhap vien"], errors="coerce")
    df["SpO2"] = pd.to_numeric(df["SpO2"], errors="coerce")
    df = df.drop(columns=[
        "ID", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện", "So ngay dieu tri",
        "Erythomycin", "Tetracyline", "Chloranphenicol", "Arithromycin", "Ampicillin",
        "Ampicillin-Sulbalactam", "Cefuroxime", "Cefuroxime Axetil", "Ceftazidine", "Viprofloxacin"
    ], errors="ignore")
    df = df[df["Tac nhan"].notna()]
    text_cols = df.select_dtypes(include="object").columns.difference(["Tac nhan"])
    df[text_cols] = df[text_cols].applymap(lambda x: 1 if str(x).strip().lower() in ["x", "có", "yes"] else 0)
    df = df.fillna(0)
    X = df.drop(columns=["Tac nhan"])
    y = df["Tac nhan"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_data_and_train()

st.markdown("## Nhập dữ liệu lâm sàng")

user_input = {}
user_input["Tuoi"] = st.number_input("Tuoi", min_value=0.0)
user_input["Benh ngay thu truoc khi nhap vien"] = st.number_input("Benh ngay thu truoc khi nhap vien", min_value=0.0)
user_input["Sot"] = st.radio("Sot", ["Không", "Có"], horizontal=True) == "Có"
user_input["Ho"] = st.radio("Ho", ["Không", "Có"], horizontal=True) == "Có"
user_input["Non"] = st.radio("Non", ["Không", "Có"], horizontal=True) == "Có"
user_input["Tieu chay"] = st.radio("Tieu chay", ["Không", "Có"], horizontal=True) == "Có"
user_input["Kich thich"] = st.radio("Kich thich", ["Không", "Có"], horizontal=True) == "Có"
user_input["Tho ren, nhanh"] = st.radio("Tho ren, nhanh", ["Không", "Có"], horizontal=True) == "Có"
user_input["Bo an"] = st.radio("Bo an", ["Không", "Có"], horizontal=True) == "Có"
user_input["Chay mui"] = st.radio("Chay mui", ["Không", "Có"], horizontal=True) == "Có"
user_input["Dam"] = st.radio("Dam", ["Không", "Có"], horizontal=True) == "Có"
user_input["Kho tho"] = st.radio("Kho tho", ["Không", "Có"], horizontal=True) == "Có"
user_input["Kho khe"] = st.radio("Kho khe", ["Không", "Có"], horizontal=True) == "Có"
user_input["Ran phoi"] = st.radio("Ran phoi", ["Không", "Có"], horizontal=True) == "Có"
user_input["Dong dac"] = st.radio("Dong dac", ["Không", "Có"], horizontal=True) == "Có"
user_input["Co lom long nguc"] = st.radio("Co lom long nguc", ["Không", "Có"], horizontal=True) == "Có"
user_input["Nhip tho"] = st.number_input("Nhip tho", min_value=0.0)
user_input["Mach"] = st.number_input("Mach", min_value=0.0)
user_input["SpO2"] = st.number_input("SpO2", min_value=0.0)
user_input["Nhiet do"] = st.number_input("Nhiet do", min_value=0.0)
user_input["CRP"] = st.number_input("CRP", min_value=0.0)
user_input["Bach cau"] = st.number_input("Bach cau", min_value=0.0)

if st.button("🔍 Dự đoán"):
    df_input = pd.DataFrame([user_input])
    for col in df_input.columns:
        if df_input[col].dtype == bool:
            df_input[col] = df_input[col].astype(int)
    prediction = model.predict(df_input)[0]
    st.success(f"Tác nhân gây bệnh được dự đoán: **{prediction}**")

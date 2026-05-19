from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Dự đoán GPA", layout="centered")


CSV_FILE = Path("Khảo sát Thói quen học tập và Kết quả học tập (GPA) của sinh viên.csv")


def find_column(columns, keyword):
    keyword = keyword.lower()
    for column in columns:
        if keyword in column.lower():
            return column
    return None


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")

    cols = {
        "gpa": find_column(df.columns, "gpa"),
        "study_hours": find_column(df.columns, "tự học"),
        "courses": find_column(df.columns, "bao nhiêu môn"),
        "part_time": find_column(df.columns, "part-time"),
        "sleep_hours": find_column(df.columns, "ngủ"),
        "club": find_column(df.columns, "câu lạc bộ"),
        "attendance": find_column(df.columns, "attendance"),
        "method": find_column(df.columns, "phương pháp học tập"),
        "social_hours": find_column(df.columns, "mạng xã hội"),
    }

    data = pd.DataFrame()
    data["gpa"] = pd.to_numeric(df[cols["gpa"]], errors="coerce")
    data["study_hours"] = pd.to_numeric(df[cols["study_hours"]].astype(str).str.strip(), errors="coerce")
    data["courses"] = pd.to_numeric(df[cols["courses"]].astype(str).str.strip(), errors="coerce")
    data["part_time"] = df[cols["part_time"]].astype(str).str.strip()
    data["sleep_hours"] = pd.to_numeric(df[cols["sleep_hours"]].astype(str).str.strip(), errors="coerce")
    data["club"] = df[cols["club"]].astype(str).str.strip()
    data["attendance"] = df[cols["attendance"]].astype(str).str.strip()
    data["method"] = df[cols["method"]].astype(str).str.strip()
    data["social_hours"] = pd.to_numeric(df[cols["social_hours"]].astype(str).str.strip(), errors="coerce")

    data = data.dropna()
    data = data[(data["gpa"] >= 0) & (data["gpa"] <= 4)]
    return data


def train_model(data):
    x = data.drop(columns=["gpa"])
    y = data["gpa"]

    numeric_cols = ["study_hours", "courses", "sleep_hours", "social_hours"]
    category_cols = ["part_time", "club", "attendance", "method"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), category_cols),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x, y)
    return model


data = load_data()
model = train_model(data)

st.title("Dự đoán GPA sinh viên")
st.write("Nhập thói quen học tập để dự đoán GPA kỳ tới.")

study_hours = st.number_input("Số giờ tự học / tuần", min_value=0.0, max_value=80.0, value=10.0, step=0.5)
courses = st.number_input("Số môn đang học", min_value=1, max_value=40, value=8, step=1)
part_time = st.selectbox("Có làm thêm không?", sorted(data["part_time"].unique()))
sleep_hours = st.number_input("Số giờ ngủ / ngày", min_value=0.0, max_value=16.0, value=7.0, step=0.5)
club = st.selectbox("Có tham gia CLB không?", sorted(data["club"].unique()))
attendance = st.selectbox("Tỷ lệ tham dự lớp", sorted(data["attendance"].unique()))
method = st.selectbox("Phương pháp học tập", sorted(data["method"].unique()))
social_hours = st.number_input("Số giờ dùng mạng xã hội / ngày", min_value=0.0, max_value=24.0, value=4.0, step=0.5)

if st.button("Dự đoán GPA"):
    new_student = pd.DataFrame(
        [
            {
                "study_hours": study_hours,
                "courses": courses,
                "part_time": part_time,
                "sleep_hours": sleep_hours,
                "club": club,
                "attendance": attendance,
                "method": method,
                "social_hours": social_hours,
            }
        ]
    )

    prediction = model.predict(new_student)[0]
    prediction = max(0, min(4, prediction))

    st.success(f"GPA dự đoán: {prediction:.2f} / 4.00")

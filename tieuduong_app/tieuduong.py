from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).with_name("diabetes.csv")
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"


st.set_page_config(
    page_title="Dự đoán tiểu đường",
    page_icon="🩺",
    layout="centered",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"File dữ liệu thiếu các cột: {missing_text}")

    return df[required_columns]


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)

    return model, accuracy


def build_input_form() -> pd.DataFrame:
    st.sidebar.header("Thông số sức khỏe")

    values = {
        "Pregnancies": st.sidebar.number_input("Số lần mang thai", 0, 20, 1),
        "Glucose": st.sidebar.slider("Nồng độ glucose", 0, 200, 100),
        "BloodPressure": st.sidebar.slider("Huyết áp tâm trương", 0, 140, 70),
        "SkinThickness": st.sidebar.slider("Độ dày da", 0, 99, 20),
        "Insulin": st.sidebar.slider("Chỉ số insulin", 0, 846, 79),
        "BMI": st.sidebar.number_input("Chỉ số BMI", 0.0, 70.0, 25.0, step=0.1),
        "DiabetesPedigreeFunction": st.sidebar.number_input(
            "Chức năng di truyền DPF",
            0.0,
            2.5,
            0.5,
            step=0.01,
        ),
        "Age": st.sidebar.slider("Tuổi", 1, 100, 30),
    }

    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def show_prediction(model: RandomForestClassifier, input_df: pd.DataFrame) -> None:
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    safe_probability = prediction_proba[0] * 100
    risk_probability = prediction_proba[1] * 100

    st.subheader("Kết quả dự đoán")

    if prediction == 1:
        st.error("Nguy cơ mắc bệnh tiểu đường đang ở mức cao.")
    else:
        st.success("Nguy cơ mắc bệnh tiểu đường đang ở mức thấp.")

    col1, col2 = st.columns(2)
    col1.metric("Khả năng không mắc bệnh", f"{safe_probability:.2f}%")
    col2.metric("Khả năng mắc bệnh", f"{risk_probability:.2f}%")

    probability_df = pd.DataFrame(
        {
            "Nhóm": ["Không mắc bệnh", "Có nguy cơ"],
            "Xác suất (%)": [safe_probability, risk_probability],
        }
    ).set_index("Nhóm")
    st.bar_chart(probability_df)


def main() -> None:
    st.title("Dự đoán nguy cơ tiểu đường")
    st.write(
        "Ứng dụng sử dụng mô hình Random Forest để ước lượng nguy cơ tiểu đường "
        "dựa trên các chỉ số sức khỏe trong bộ dữ liệu."
    )

    try:
        df = load_data()
        model, accuracy = train_model(df)
    except Exception as exc:
        st.error(f"Không thể khởi tạo ứng dụng: {exc}")
        st.stop()

    st.caption(f"Dữ liệu: {len(df)} dòng | Độ chính xác kiểm thử: {accuracy:.2%}")

    input_df = build_input_form()

    st.subheader("Thông số đã nhập")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Kiểm tra kết quả", type="primary"):
        show_prediction(model, input_df)

    st.info(
        "Lưu ý: Kết quả chỉ có giá trị tham khảo và không thay thế cho chẩn đoán "
        "hoặc tư vấn từ bác sĩ."
    )


if __name__ == "__main__":
    main()

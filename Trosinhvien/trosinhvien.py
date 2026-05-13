import streamlit as st
import pandas as pd
import numpy as np
import html
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="UEH - Dự đoán giá thuê trọ",
    page_icon="🏠",
    layout="wide",
)


ADDRESS_COL = "Địa chỉ phòng"
DISTANCE_COL = "Khoảng cách tới UEH cơ sở B (km)"
AREA_COL = "Diện tích phòng (m²)"
AMENITY_COL = "Có máy lạnh / WC riêng / bếp"
TIME_COL = "Giờ giấc tự do?"
PEOPLE_COL = "Số người ở tối đa"
PARKING_COL = "Có chỗ để xe?"
NEARBY_COL = "Gần tiện ích (siêu thị, bus)"
LINK_COL = "Link truy cập tham khảo"
WARD_COL = "Phường"
PRICE_COL = "Giá thuê (Triệu)"


st.markdown(
    """
    <style>
        :root {
            --primary: #0f766e;
            --primary-dark: #115e59;
            --ink: #17202a;
            --muted: #64748b;
            --line: #dbe3ea;
            --soft: #f6f8fb;
            --accent: #f59e0b;
        }

        .stApp {
            background: #f4f7fa;
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--ink);
        }

        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #eef8f6 100%);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
            letter-spacing: 0;
            color: var(--ink);
        }

        .hero p {
            margin: .55rem 0 0 0;
            color: var(--muted);
            font-size: 1rem;
        }

        .section {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem 1.1rem;
            height: 100%;
        }

        .section h3 {
            margin-top: 0;
            margin-bottom: .75rem;
            font-size: 1rem;
            color: var(--ink);
        }

        .result-box {
            border: 1px solid #99f6e4;
            border-left: 6px solid var(--primary);
            border-radius: 8px;
            background: #f0fdfa;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .result-label {
            color: var(--primary-dark);
            font-size: .86rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .result-price {
            color: var(--ink);
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 800;
            margin-top: .35rem;
        }

        .result-note {
            color: var(--muted);
            margin-top: .4rem;
            font-size: .95rem;
        }

        .insight {
            border-top: 1px solid var(--line);
            padding-top: .75rem;
            margin-top: .75rem;
            color: var(--ink);
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: .45rem;
            margin-top: .75rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: .28rem .55rem;
            border-radius: 999px;
            background: #eef2f7;
            color: #334155;
            font-size: .82rem;
            border: 1px solid #d9e2ec;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .85rem 1rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--primary);
            background: var(--primary);
            color: white;
            font-weight: 700;
            padding: .65rem 1rem;
        }

        .stButton > button:hover {
            border-color: var(--primary-dark);
            background: var(--primary-dark);
            color: white;
        }

        .match-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: .9rem 1rem;
            margin-bottom: .75rem;
        }

        .match-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: .35rem;
        }

        .match-meta {
            color: var(--muted);
            font-size: .9rem;
            margin-bottom: .45rem;
        }

        .match-score {
            display: inline-flex;
            align-items: center;
            padding: .22rem .5rem;
            border-radius: 999px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #047857;
            font-size: .82rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("trosinhvien.csv", sep=";")

    cols = [
        ADDRESS_COL,
        DISTANCE_COL,
        AREA_COL,
        AMENITY_COL,
        TIME_COL,
        PEOPLE_COL,
        PARKING_COL,
        NEARBY_COL,
    ]
    if LINK_COL in df.columns:
        cols.append(LINK_COL)
    df = df[cols].copy()

    if LINK_COL not in df.columns:
        df[LINK_COL] = ""

    df[DISTANCE_COL] = df[DISTANCE_COL].astype(str).str.replace(",", ".", regex=False).astype(float)
    df[AREA_COL] = pd.to_numeric(df[AREA_COL], errors="coerce").fillna(20)
    df[PEOPLE_COL] = pd.to_numeric(df[PEOPLE_COL], errors="coerce").fillna(2).astype(int)

    amenity_text = df[AMENITY_COL].fillna("")
    df["is_may_lanh"] = amenity_text.str.contains("Máy lạnh|nội thất|Full", case=False, regex=True).astype(int)
    df["is_wc_rieng"] = amenity_text.str.contains("WC riêng", case=False, regex=False).astype(int)
    df["is_tu_do"] = df[TIME_COL].fillna("").str.contains("Có", case=False, regex=False).astype(int)
    df["is_xe"] = df[PARKING_COL].fillna("").str.contains("Có|Gửi xe", case=False, regex=True).astype(int)

    df[WARD_COL] = df[ADDRESS_COL].fillna("").str.extract(r"(P\.\d+|Phường \d+)")
    df[WARD_COL] = df[WARD_COL].fillna("Khác")

    # File hiện tại chưa có giá thuê thật, nên giá được mô phỏng từ diện tích,
    # khoảng cách và tiện ích. Khi có dữ liệu giá thật, thay công thức này bằng cột giá.
    df[PRICE_COL] = (
        1.5
        + df[AREA_COL] * 0.12
        - df[DISTANCE_COL] * 0.3
        + df["is_may_lanh"] * 0.8
        + df["is_wc_rieng"] * 0.5
        + df["is_tu_do"] * 0.3
    ).round(1)

    return df


@st.cache_resource
def train_model(data):
    encoder = LabelEncoder()
    work_df = data.copy()
    work_df["phuong_encoded"] = encoder.fit_transform(work_df[WARD_COL])

    feature_cols = [
        "phuong_encoded",
        DISTANCE_COL,
        AREA_COL,
        "is_may_lanh",
        "is_wc_rieng",
        "is_tu_do",
        PEOPLE_COL,
        "is_xe",
    ]
    model_data = work_df[feature_cols]
    target = work_df[PRICE_COL]

    regressor = RandomForestRegressor(n_estimators=120, random_state=42)
    regressor.fit(model_data, target)
    return regressor, encoder


def format_vnd(value):
    return f"{value:.1f} triệu VNĐ/tháng"


def get_distance_message(distance):
    if distance <= 1:
        return "Rất gần UEH cơ sở B, phù hợp đi bộ hoặc di chuyển ngắn."
    if distance <= 2.5:
        return "Khoảng cách vừa phải, phù hợp đi xe máy hoặc xe buýt."
    return "Xa hơn trung tâm học tập, nên cân nhắc thời gian di chuyển hằng ngày."


def render_feature_pills(has_ac, has_wc, has_time, has_bike, limit_people):
    features = [
        ("Máy lạnh / nội thất", has_ac),
        ("WC riêng", has_wc),
        ("Giờ giấc tự do", has_time),
        ("Có chỗ để xe", has_bike),
        (f"Tối đa {limit_people} người", True),
    ]
    pills = "".join(f'<span class="pill">{name}</span>' for name, active in features if active)
    st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)


def get_best_matches(data, phuong, distance, area, has_ac, has_wc, has_time, has_bike, people, prediction):
    matches = data.copy()

    matches["Điểm phù hợp"] = 100
    matches["Điểm phù hợp"] -= (matches[WARD_COL] != phuong).astype(int) * 18
    matches["Điểm phù hợp"] -= (matches[DISTANCE_COL] - distance).abs().clip(0, 5) * 8
    matches["Điểm phù hợp"] -= (matches[AREA_COL] - area).abs().clip(0, 30) * 1.2
    matches["Điểm phù hợp"] -= (matches[PEOPLE_COL] < people).astype(int) * 22
    matches["Điểm phù hợp"] -= (matches[PRICE_COL] - prediction).abs().clip(0, 10) * 4

    requested_features = {
        "is_may_lanh": has_ac,
        "is_wc_rieng": has_wc,
        "is_tu_do": has_time,
        "is_xe": has_bike,
    }
    for feature, requested in requested_features.items():
        if requested:
            matches["Điểm phù hợp"] -= (matches[feature] == 0).astype(int) * 10

    matches["Điểm phù hợp"] = matches["Điểm phù hợp"].clip(lower=0).round().astype(int)
    return matches.sort_values(
        by=["Điểm phù hợp", WARD_COL, DISTANCE_COL],
        ascending=[False, True, True],
    ).head(5)


df = load_and_clean_data()
model, le = train_model(df)

st.markdown(
    f"""
    <div class="hero">
        <h1>Dự đoán giá thuê trọ quanh UEH</h1>
        <p>Ước tính giá thuê theo khu vực, khoảng cách, diện tích và tiện ích từ {len(df)} mẫu dữ liệu khảo sát.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Thông tin phòng trọ")
    st.caption("Điều chỉnh các thông số để xem giá thuê ước tính.")

    with st.form("input_form"):
        phuong_sel = st.selectbox("Khu vực / phường", options=le.classes_)
        km = st.slider("Khoảng cách tới UEH cơ sở B", 0.0, 5.0, 1.2, 0.1, format="%.1f km")
        m2 = st.number_input("Diện tích phòng", 10, 50, 15, step=1)

        st.markdown("**Tiện ích**")
        has_ac = st.checkbox("Máy lạnh / full nội thất")
        has_wc = st.checkbox("WC riêng")
        has_time = st.checkbox("Giờ giấc tự do")
        has_bike = st.checkbox("Có chỗ để xe")

        limit_people = st.number_input("Số người ở tối đa", 1, 5, 2, step=1)
        submitted = st.form_submit_button("Dự đoán giá thuê")

if submitted:
    p_enc = le.transform([phuong_sel])[0]
    input_data = pd.DataFrame(
        [[p_enc, km, m2, int(has_ac), int(has_wc), int(has_time), limit_people, int(has_bike)]],
        columns=[
            "phuong_encoded",
            DISTANCE_COL,
            AREA_COL,
            "is_may_lanh",
            "is_wc_rieng",
            "is_tu_do",
            PEOPLE_COL,
            "is_xe",
        ],
    )
    prediction = model.predict(input_data)[0]
else:
    prediction = df[PRICE_COL].median()

avg_price = df[PRICE_COL].mean()
avg_area = df[AREA_COL].mean()
avg_distance = df[DISTANCE_COL].mean()

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Giá trung bình", format_vnd(avg_price))
metric_col2.metric("Diện tích TB", f"{avg_area:.1f} m²")
metric_col3.metric("Khoảng cách TB", f"{avg_distance:.1f} km")
metric_col4.metric("Số mẫu dữ liệu", f"{len(df)} phòng")

left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Kết quả ước tính")
    if submitted:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Giá thuê phù hợp</div>
                <div class="result-price">{format_vnd(prediction)}</div>
                <div class="result-note">Dựa trên thông tin bạn vừa nhập và dữ liệu khảo sát hiện có.</div>
                <div class="insight"><strong>Nhận xét:</strong> {get_distance_message(km)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_feature_pills(has_ac, has_wc, has_time, has_bike, limit_people)
    else:
        st.info("Nhập thông tin ở thanh bên trái rồi bấm Dự đoán giá thuê để xem kết quả.")
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Giá tham khảo theo dữ liệu hiện có</div>
                <div class="result-price">{format_vnd(prediction)}</div>
                <div class="result-note">Đây là mức trung vị của bộ dữ liệu, dùng để tham khảo ban đầu.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Phân tích nhanh")
    price_min = df[PRICE_COL].min()
    price_max = df[PRICE_COL].max()
    same_ward = df[df[WARD_COL] == phuong_sel]
    ward_price = same_ward[PRICE_COL].mean() if not same_ward.empty else avg_price

    st.write(f"**Khu vực đang chọn:** {phuong_sel}")
    st.write(f"**Giá trung bình khu vực:** {format_vnd(ward_price)}")
    st.write(f"**Khoảng giá trong dữ liệu:** {price_min:.1f} - {price_max:.1f} triệu VNĐ/tháng")
    st.progress(min(max((prediction - price_min) / (price_max - price_min), 0), 1))
    st.caption("Thanh tiến độ thể hiện vị trí giá ước tính trong khoảng giá của bộ dữ liệu.")
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    st.markdown("### Phòng phù hợp với lựa chọn của bạn")
    best_matches = get_best_matches(
        df,
        phuong_sel,
        km,
        m2,
        has_ac,
        has_wc,
        has_time,
        has_bike,
        limit_people,
        prediction,
    )

    for index, row in best_matches.iterrows():
        link = str(row.get(LINK_COL, "")).strip()
        link_html = (
            f'<a href="{html.escape(link)}" target="_blank">Xem link tham khảo</a>'
            if link and link.lower() != "nan"
            else ""
        )
        st.markdown(
            f"""
            <div class="match-card">
                <div class="match-title">{html.escape(str(row[ADDRESS_COL]))}</div>
                <div class="match-meta">
                    {html.escape(str(row[WARD_COL]))} · {row[DISTANCE_COL]:.1f} km tới UEH B ·
                    {row[AREA_COL]:.0f} m² · tối đa {row[PEOPLE_COL]} người ·
                    {row[PRICE_COL]:.1f} triệu VNĐ/tháng
                </div>
                <span class="match-score">{row["Điểm phù hợp"]}% phù hợp</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if link_html:
            st.markdown(link_html, unsafe_allow_html=True)

    match_table = best_matches[
        [
            ADDRESS_COL,
            WARD_COL,
            DISTANCE_COL,
            AREA_COL,
            PEOPLE_COL,
            AMENITY_COL,
            NEARBY_COL,
            PRICE_COL,
            "Điểm phù hợp",
        ]
    ].rename(
        columns={
            ADDRESS_COL: "Địa chỉ phù hợp",
            WARD_COL: "Khu vực",
            DISTANCE_COL: "Cách UEH B (km)",
            AREA_COL: "Diện tích (m²)",
            PEOPLE_COL: "Số người",
            AMENITY_COL: "Tiện ích phòng",
            NEARBY_COL: "Tiện ích xung quanh",
            PRICE_COL: "Giá ước tính (triệu)",
        }
    )

    with st.expander("Xem bảng so sánh chi tiết"):
        st.dataframe(match_table, use_container_width=True, hide_index=True)

tab_data, tab_map = st.tabs(["Dữ liệu phòng trọ", "Bản đồ khu vực"])

with tab_data:
    display_df = df[
        [
            ADDRESS_COL,
            WARD_COL,
            DISTANCE_COL,
            AREA_COL,
            PEOPLE_COL,
            AMENITY_COL,
            NEARBY_COL,
            LINK_COL,
            PRICE_COL,
        ]
    ].rename(
        columns={
            ADDRESS_COL: "Địa chỉ",
            WARD_COL: "Khu vực",
            DISTANCE_COL: "Cách UEH B (km)",
            AREA_COL: "Diện tích (m²)",
            PEOPLE_COL: "Số người",
            AMENITY_COL: "Tiện ích phòng",
            NEARBY_COL: "Tiện ích xung quanh",
            LINK_COL: "Link tham khảo",
            PRICE_COL: "Giá ước tính (triệu)",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Giá ước tính (triệu)": st.column_config.NumberColumn(format="%.1f"),
            "Cách UEH B (km)": st.column_config.NumberColumn(format="%.1f"),
            "Diện tích (m²)": st.column_config.NumberColumn(format="%d"),
        },
    )

with tab_map:
    st.caption("Tọa độ đang được mô phỏng quanh khu Nguyễn Tri Phương vì file CSV chưa có kinh độ/vĩ độ thật.")
    rng = np.random.default_rng(42)
    map_data = pd.DataFrame(
        {
            "lat": rng.uniform(10.755, 10.770, len(df)),
            "lon": rng.uniform(106.660, 106.670, len(df)),
            "weight": df[PRICE_COL],
        }
    )
    st.map(map_data, size=28, color="#0f766e")

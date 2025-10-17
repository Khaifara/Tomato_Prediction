# app_streamlit_modern.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time
from io import BytesIO

# ---------- Page config ----------
st.set_page_config(
    page_title="Klasifikasi Tomat — Modern",
    page_icon="🍅",
    layout="wide"
)

# ---------- Styling (simple material-like card) ----------
st.markdown(
    """
    <style>
    /* background */
    .stApp {
        background: linear-gradient(180deg, #fff 0%, #fffaf6 100%);
    }
    /* card */
    .card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .muted {
        color: #6b7280;
        font-size:13px;
    }
    .brand {
        color: #e74c3c;
        font-weight:700;
    }
    /* remove default streamlit padding on wide mode for nicer edge */
    .css-12oz5g7 { padding: 0rem 1rem 1rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load dataset and model (with graceful errors) ----------
@st.cache_data(show_spinner=False)
def load_data(path="dataset_tomat.csv"):
    return pd.read_csv(path)

def load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

# load
try:
    df = load_data("dataset_tomat.csv")
except FileNotFoundError:
    st.error("File dataset_tomat.csv tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

model = load_joblib("model_klasifikasi_tomat.joblib")
scaler = load_joblib("scaler_klasifikasi_tomat.joblib")

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.markdown("<h3 class='brand'>🍅 Klasifikasi Tomat</h3>", unsafe_allow_html=True)
    st.write("Masukkan fitur tomat (geser atau ketik).")
    berat = st.slider("Berat Tomat (gr)", min_value=int(df["berat"].min()), max_value=int(df["berat"].max()), value=int(df["berat"].median()))
    kekenyalan = st.slider("Kekenyalan Tomat (N)", min_value=float(df["kekenyalan"].min()), max_value=float(df["kekenyalan"].max()), value=float(df["kekenyalan"].median()))
    kadar_gula = st.slider("Kadar Gula (Bx)", min_value=float(df["kadar_gula"].min()), max_value=float(df["kadar_gula"].max()), value=float(df["kadar_gula"].median()))
    tebal_kulit = st.slider("Tebal Kulit Tomat (cm)", min_value=float(df["tebal_kulit"].min()), max_value=float(df["tebal_kulit"].max()), value=float(df["tebal_kulit"].median()))
    st.divider()
    st.caption("Theme: Material-inspired • Interactive charts • Lightweight")
    st.button_label = "Prediksi"

# ---------- Main layout ----------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input")
    st.write("Data input yang akan diprediksi:")
    data_baru = pd.DataFrame(
        [[berat, kekenyalan, kadar_gula, tebal_kulit]],
        columns=["berat", "kekenyalan", "kadar_gula", "tebal_kulit"]
    )
    st.dataframe(data_baru, use_container_width=True)
    with st.expander("Lihat dataset contoh (sample)"):
        st.dataframe(df.sample(min(10, len(df))).reset_index(drop=True))
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")
    pred_box = st.empty()
    conf_box = st.empty()
    badge_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Action: Prediksi ----------
predict_btn = st.sidebar.button("Prediksi", use_container_width=True)

if predict_btn:
    # spinner & simulate small delay for UX
    with st.spinner("Sedang memproses prediksi... ✨"):
        time.sleep(0.4)

        # scale if scaler available
        if scaler is not None:
            try:
                data_baru_scaled = scaler.transform(data_baru)
            except Exception as e:
                st.error(f"Error saat scaling: {e}")
                data_baru_scaled = None
        else:
            data_baru_scaled = None

        # perform prediction if model present
        if model is not None and data_baru_scaled is not None:
            try:
                prediksi = model.predict(data_baru_scaled)[0]
                proba = model.predict_proba(data_baru_scaled)[0]
                presentase = float(np.max(proba))
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                prediksi = None
                presentase = None
        else:
            prediksi = None
            presentase = None

    # ---------- Show results ----------
    if prediksi is not None:
        # nice metric
        pred_box.metric(label="Grade Prediksi", value=str(prediksi))
        conf_box.metric(label="Keyakinan", value=f"{presentase*100:.2f}%")
        # badge with color
        grade_color = {"Ekspor":"#e74c3c", "Lokal Premium":"#27ae60", "Industri":"#3498db"}
        color = grade_color.get(prediksi, "#9CA3AF")
        badge_box.markdown(
            f"<div style='padding:12px;border-radius:10px;background:{color};color:white;font-weight:700;text-align:center'>{prediksi}</div>",
            unsafe_allow_html=True
        )
        st.success(f"Model memprediksi **{prediksi}** dengan keyakinan **{presentase*100:.2f}%**.")
        st.balloons()
    else:
        st.warning("Model atau scaler tidak tersedia — prediksi dibatalkan. Pastikan model_klasifikasi_tomat.joblib dan scaler_klasifikasi_tomat.joblib ada di folder.")

    # show scaled in expander
    if data_baru_scaled is not None:
        scaled_df = pd.DataFrame(data_baru_scaled, columns=data_baru.columns)
        with st.expander("Lihat Data Baru (Scaled)"):
            st.dataframe(scaled_df)

    # allow download of result
    result_export = data_baru.copy()
    result_export["prediksi"] = prediksi if prediksi is not None else ""
    result_export["confidence"] = f"{presentase*100:.2f}%" if presentase is not None else ""
    csv = result_export.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download hasil prediksi (CSV)", data=csv, file_name="hasil_prediksi_tomat.csv", mime="text/csv")

# ---------- Charts (interactive) ----------
st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
st.subheader("Visualisasi Interaktif")
# prepare colors map for dataset
color_map = {"Ekspor":"Tomato", "Lokal Premium":"MediumSeaGreen", "Industri":"DodgerBlue"}
df["color_map"] = df["grade"].map(color_map)

# Scatter 1: Berat vs Kekenyalan
fig1 = px.scatter(
    df,
    x="berat",
    y="kekenyalan",
    color="grade",
    color_discrete_map=color_map,
    hover_data=["grade", "kadar_gula", "tebal_kulit"],
    title="Berat vs Kekenyalan"
)

# add the new point if exists
if 'data_baru' in locals():
    fig1.add_scatter(x=[data_baru["berat"].iloc[0]], y=[data_baru["kekenyalan"].iloc[0]],
                     mode="markers", marker=dict(size=12, symbol="x", color="black"),
                     name="Data Baru")

fig1.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig1, use_container_width=True)

# Scatter 2: Kadar Gula vs Tebal Kulit
fig2 = px.scatter(
    df,
    x="kadar_gula",
    y="tebal_kulit",
    color="grade",
    color_discrete_map=color_map,
    hover_data=["grade", "berat", "kekenyalan"],
    title="Kadar Gula vs Tebal Kulit"
)
if 'data_baru' in locals():
    fig2.add_scatter(x=[data_baru["kadar_gula"].iloc[0]], y=[data_baru["tebal_kulit"].iloc[0]],
                     mode="markers", marker=dict(size=12, symbol="x", color="black"),
                     name="Data Baru")
fig2.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("Dibuat dengan penuh 🍅 oleh Khairul Faiz — tampilan modern")

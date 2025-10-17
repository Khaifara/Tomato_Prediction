import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time

# ---------- Page Config ----------
st.set_page_config(
    page_title="Klasifikasi Tomat — Modern",
    page_icon="🍅",
    layout="wide"
)

# ---------- Custom Style (Light + Dark Mode) ----------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"

toggle = st.sidebar.toggle("🌗 Mode Gelap", value=False)
st.session_state.theme_mode = "dark" if toggle else "light"

if st.session_state.theme_mode == "light":
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(180deg, #f7f7f7 0%, #fffdfa 100%); color: #1e1e1e;}
        .card {background: #ffffff; border-radius: 12px; padding: 20px; margin-bottom: 18px;
               box-shadow: 0 6px 18px rgba(0,0,0,0.08); border: 1px solid #efefef;}
        h1,h2,h3,h4,h5 {color:#222!important;font-weight:600;}
        section[data-testid="stSidebar"] {background-color:#ffffff;border-right:1px solid #e0e0e0;}
        section[data-testid="stSidebar"] h3 {color:#e74c3c!important;font-weight:700;}
        table {border-collapse:collapse!important;border-radius:10px!important;overflow:hidden!important;}
        thead th {background-color:#f2f2f2!important;color:#1a1a1a!important;font-weight:600!important;text-align:center!important;}
        tbody tr {background-color:#ffffff!important;color:#1a1a1a!important;text-align:center!important;}
        tbody tr:nth-child(even){background-color:#fafafa!important;}
        .stMetric {background:#fff;border-radius:10px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,0.05);}
        .stDownloadButton button {background:#e74c3c;color:white;border:none;border-radius:8px;
                                  padding:10px 18px;font-weight:600;transition:0.2s;}
        .stDownloadButton button:hover {background:#c0392b;transform:scale(1.02);}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        .stApp {background: #1c1c1c; color: #f5f5f5;}
        .card {background: #2a2a2a; border-radius: 12px; padding: 20px; margin-bottom: 18px;
               box-shadow: 0 6px 18px rgba(0,0,0,0.5); border: 1px solid #333;}
        h1,h2,h3,h4,h5 {color:#f5f5f5!important;font-weight:600;}
        section[data-testid="stSidebar"] {background-color:#252525;border-right:1px solid #333;color:#f5f5f5;}
        section[data-testid="stSidebar"] h3 {color:#ff6347!important;font-weight:700;}
        thead th {background-color:#333!important;color:#f5f5f5!important;}
        tbody tr {background-color:#2a2a2a!important;color:#f5f5f5!important;}
        tbody tr:nth-child(even){background-color:#202020!important;}
        .stMetric {background:#333;border-radius:10px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,0.3);}
        .stDownloadButton button {background:#ff6347;color:white;border:none;border-radius:8px;
                                  padding:10px 18px;font-weight:600;transition:0.2s;}
        .stDownloadButton button:hover {background:#e74c3c;transform:scale(1.02);}
        </style>
        """, unsafe_allow_html=True)

# ---------- Load Dataset & Model ----------
@st.cache_data(show_spinner=False)
def load_data(path="dataset_tomat.csv"):
    return pd.read_csv(path)

def load_joblib(path):
    try:
        return joblib.load(path)
    except:
        return None

try:
    df = load_data("dataset_tomat.csv")
except FileNotFoundError:
    st.error("❌ File dataset_tomat.csv tidak ditemukan.")
    st.stop()

model = load_joblib("model_klasifikasi_tomat.joblib")
scaler = load_joblib("scaler_klasifikasi_tomat.joblib")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<h3 class='brand'>🍅 Klasifikasi Tomat</h3>", unsafe_allow_html=True)
    berat = st.slider("Berat Tomat (gr)", int(df["berat"].min()), int(df["berat"].max()), int(df["berat"].median()))
    kekenyalan = st.slider("Kekenyalan Tomat (N)", float(df["kekenyalan"].min()), float(df["kekenyalan"].max()), float(df["kekenyalan"].median()))
    kadar_gula = st.slider("Kadar Gula (Bx)", float(df["kadar_gula"].min()), float(df["kadar_gula"].max()), float(df["kadar_gula"].median()))
    tebal_kulit = st.slider("Tebal Kulit Tomat (cm)", float(df["tebal_kulit"].min()), float(df["tebal_kulit"].max()), float(df["tebal_kulit"].median()))
    st.divider()
    predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True)

# ---------- Main Layout ----------
col1, col2 = st.columns([1.1, 1])
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📥 Input Data")
    data_baru = pd.DataFrame([[berat, kekenyalan, kadar_gula, tebal_kulit]],
                              columns=["berat","kekenyalan","kadar_gula","tebal_kulit"])
    st.dataframe(data_baru, use_container_width=True)
    with st.expander("Lihat sampel dataset"):
        st.dataframe(df.sample(min(10, len(df))).reset_index(drop=True))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Hasil Prediksi")
    pred_box = st.empty()
    conf_box = st.empty()
    badge_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Predict ----------
if predict_btn:
    with st.spinner("Sedang memproses prediksi... 🍅"):
        time.sleep(0.5)
        if scaler is not None:
            data_baru_scaled = scaler.transform(data_baru)
        else:
            data_baru_scaled = data_baru

        if model is not None:
            prediksi = model.predict(data_baru_scaled)[0]
            proba = model.predict_proba(data_baru_scaled)[0]
            presentase = float(np.max(proba))
        else:
            prediksi = "Model Tidak Ditemukan"
            presentase = 0

    # Tampilkan hasil
    grade_color = {"Ekspor":"#e74c3c","Lokal Premium":"#27ae60","Industri":"#3498db"}
    color = grade_color.get(prediksi, "#777777")
    pred_box.metric("Grade Prediksi", prediksi)
    conf_box.metric("Keyakinan", f"{presentase*100:.2f}%")
    badge_box.markdown(
        f"<div style='padding:12px;border-radius:10px;background:{color};color:white;font-weight:700;text-align:center'>{prediksi}</div>",
        unsafe_allow_html=True)
    st.success(f"Model memprediksi **{prediksi}** dengan keyakinan **{presentase*100:.2f}%**")
    st.balloons()

    # Download hasil
    result = data_baru.copy()
    result["Prediksi"] = prediksi
    result["Confidence"] = f"{presentase*100:.2f}%"
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download hasil prediksi", csv, "hasil_prediksi_tomat.csv", "text/csv")

# ---------- Visualisasi ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Visualisasi Interaktif")
color_map = {"Ekspor":"Tomato","Lokal Premium":"MediumSeaGreen","Industri":"DodgerBlue"}
df["color_map"] = df["grade"].map(color_map)

fig1 = px.scatter(df, x="berat", y="kekenyalan", color="grade",
                  color_discrete_map=color_map, title="Berat vs Kekenyalan")
fig1.add_scatter(x=[berat], y=[kekenyalan], mode="markers",
                 marker=dict(size=12, symbol="x", color="black"), name="Data Baru")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(df, x="kadar_gula", y="tebal_kulit", color="grade",
                  color_discrete_map=color_map, title="Kadar Gula vs Tebal Kulit")
fig2.add_scatter(x=[kadar_gula], y=[tebal_kulit], mode="markers",
                 marker=dict(size=12, symbol="x", color="black"), name="Data Baru")
st.plotly_chart(fig2, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("Dibuat dengan penuh 🍅 oleh Khairul Faiz — tampilan modern & mode gelap")

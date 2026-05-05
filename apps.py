import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Bunga Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS Kustom
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f9f5ff; }
    .stButton>button {
        background-color: #7c3aed;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-size: 1rem;
        border: none;
        transition: background 0.3s;
    }
    .stButton>button:hover { background-color: #5b21b6; }
    .result-box {
        background: linear-gradient(135deg, #ede9fe, #ddd6fe);
        border-left: 6px solid #7c3aed;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    return df

# ─────────────────────────────────────────────
# Train Model
# ─────────────────────────────────────────────
@st.cache_data
def train_model(df):
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return model, accuracy, report, cm, model.classes_

# ─────────────────────────────────────────────
# Load semua data & model
# ─────────────────────────────────────────────
df = load_data()
model, accuracy, report, cm, classes = train_model(df)

# ─────────────────────────────────────────────
# SIDEBAR — Input Pengguna
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", use_column_width=True)
    st.markdown("## 🌿 Input Fitur Bunga")
    st.caption("Geser slider untuk memasukkan nilai fitur:")

    sepal_length = st.slider(
        "📏 Sepal Length (cm)",
        float(df.sepal_length.min()),
        float(df.sepal_length.max()),
        float(df.sepal_length.mean()),
        step=0.1,
    )
    sepal_width = st.slider(
        "📐 Sepal Width (cm)",
        float(df.sepal_width.min()),
        float(df.sepal_width.max()),
        float(df.sepal_width.mean()),
        step=0.1,
    )
    petal_length = st.slider(
        "🌿 Petal Length (cm)",
        float(df.petal_length.min()),
        float(df.petal_length.max()),
        float(df.petal_length.mean()),
        step=0.1,
    )
    petal_width = st.slider(
        "🌱 Petal Width (cm)",
        float(df.petal_width.min()),
        float(df.petal_width.max()),
        float(df.petal_width.mean()),
        step=0.1,
    )

    predict_btn = st.button("🔮 Prediksi Sekarang")

# ─────────────────────────────────────────────
# MAIN — Judul & Deskripsi
# ─────────────────────────────────────────────
st.title("🌸 Aplikasi Klasifikasi Bunga Iris")
st.markdown(
    "Aplikasi ini menggunakan model **Random Forest** untuk mengklasifikasikan "
    "jenis bunga iris berdasarkan ukuran sepal dan petal."
)
st.divider()

# ─────────────────────────────────────────────
# Tab Utama
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Performa Model", "📋 Dataset"])

# ── Tab 1: Prediksi ──
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📌 Nilai Fitur yang Dimasukkan")
        input_df = pd.DataFrame({
            "Fitur": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Nilai (cm)": [sepal_length, sepal_width, petal_length, petal_width],
        })
        st.dataframe(input_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 🧠 Hasil Prediksi")
        if predict_btn:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            emoji_map = {
                "setosa": "🌺",
                "versicolor": "🌼",
                "virginica": "🌷",
            }
            emoji = emoji_map.get(prediction, "🌸")

            st.markdown(
                f'<div class="result-box">'
                f'{emoji} Jenis Iris yang diprediksi: <b>{prediction.upper()}</b><br>'
                f'✅ Akurasi model: <b>{accuracy * 100:.2f}%</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Probabilitas per Kelas")
            proba_df = pd.DataFrame({
                "Spesies": classes,
                "Probabilitas (%)": (proba * 100).round(2),
            })
            st.bar_chart(proba_df.set_index("Spesies"))
        else:
            st.info("👈 Atur slider di sidebar, lalu klik **Prediksi Sekarang**.")

# ── Tab 2: Performa Model ──
with tab2:
    st.markdown("### 📈 Akurasi Model")
    st.metric("Akurasi pada Data Uji", f"{accuracy * 100:.2f}%")

    st.markdown("### 🗂️ Classification Report")
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("### 🔲 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ── Tab 3: Dataset ──
with tab3:
    st.markdown("### 📋 Dataset Iris (150 baris)")
    st.dataframe(df, use_container_width=True)

    st.markdown("### 📊 Statistik Deskriptif")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### 📉 Distribusi Fitur per Spesies")
    feature = st.selectbox(
        "Pilih fitur:",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for species in df["species"].unique():
        subset = df[df["species"] == species]
        ax2.hist(subset[feature], alpha=0.6, label=species, bins=15)
    ax2.set_xlabel(feature.replace("_", " ").title() + " (cm)")
    ax2.set_ylabel("Frekuensi")
    ax2.set_title(f"Distribusi {feature.replace('_', ' ').title()} per Spesies")
    ax2.legend()
    st.pyplot(fig2)

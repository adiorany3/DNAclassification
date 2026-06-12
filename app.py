from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

from feature_extraction import FEATURE_COLUMNS, build_feature_frame, clean_sequence, validate_sequence

st.set_page_config(page_title="DNA Sequence Classifier", page_icon="🧬", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILES = [
    MODEL_DIR / "organism_classifier.joblib",
    MODEL_DIR / "severity_classifier.joblib",
    MODEL_DIR / "high_risk_classifier.joblib",
    MODEL_DIR / "metadata.joblib",
]


def ensure_models_available():
    """Use uploaded models if available; otherwise train once inside Streamlit Cloud."""
    missing = [path.name for path in MODEL_FILES if not path.exists()]
    if not missing:
        return

    train_script = BASE_DIR / "train_model.py"
    data_file = BASE_DIR / "processed_dna_dataset.csv"
    if not train_script.exists() or not data_file.exists():
        raise FileNotFoundError(
            "Model belum tersedia dan file train_model.py / processed_dna_dataset.csv tidak ditemukan. "
            "Upload seluruh folder project ke GitHub."
        )

    with st.spinner("Model belum ditemukan. Sistem sedang training otomatis satu kali..."):
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=180,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "Training otomatis gagal. Log error:\n"
            + result.stderr[-3000:]
        )


def _load_model_files():
    class_model = joblib.load(MODEL_DIR / "organism_classifier.joblib")
    severity_model = joblib.load(MODEL_DIR / "severity_classifier.joblib")
    high_risk_model = joblib.load(MODEL_DIR / "high_risk_classifier.joblib")
    metadata = joblib.load(MODEL_DIR / "metadata.joblib")
    return class_model, severity_model, high_risk_model, metadata


@st.cache_resource
def load_models():
    ensure_models_available()
    try:
        return _load_model_files()
    except Exception:
        # If serialized models are incompatible with the cloud environment,
        # retrain once using the uploaded dataset, then reload.
        for path in MODEL_FILES:
            if path.exists():
                path.unlink()
        ensure_models_available()
        return _load_model_files()


def probability_table(model, X):
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame()
    probs = model.predict_proba(X)[0]
    labels = [str(label) for label in model.classes_]
    return pd.DataFrame({"Label": labels, "Probabilitas": probs}).sort_values("Probabilitas", ascending=False)


def format_percent(value):
    return f"{value * 100:.2f}%"


st.title("🧬 Sistem Identifikasi Sequence DNA")
st.write(
    "Sistem ini menerima input sequence DNA, menghitung fitur komposisi basa, "
    "memprediksi kelas DNA, dan memberi estimasi tingkat keparahan/risiko."
)

try:
    class_model, severity_model, high_risk_model, metadata = load_models()
except Exception as exc:
    st.error("Model belum bisa dimuat. Pastikan semua file project sudah di-upload ke GitHub.")
    st.exception(exc)
    st.stop()

with st.expander("Informasi model dan batasan"):
    st.write(f"Model klasifikasi DNA: **{metadata.get('best_class_model')}**")
    st.write(f"Model tingkat keparahan: **{metadata.get('best_risk_model')}**")
    st.warning(metadata.get("important_note", "Hasil prediksi perlu divalidasi ahli."))

example_sequence = (
    "CTTTCGGGATACTTTTGGGATGGTCTTGGTCAAGGGTTTTAGCCCGCAGACAGACTTTAAAACGAACCTTGCGGCAATTGCGGGCGAGAAGTTGGCTTAG"
)
sequence = st.text_area(
    "Masukkan sequence DNA",
    value=example_sequence,
    height=160,
    help="Gunakan karakter A, T, C, dan G saja.",
)
mutation_flag = st.radio(
    "Apakah ada indikasi mutasi dari pemeriksaan/lab?",
    options=[0, 1],
    format_func=lambda x: "Tidak / belum diketahui" if x == 0 else "Ya, mutasi terdeteksi",
    horizontal=True,
)

col1, col2 = st.columns([1, 2])
with col1:
    predict = st.button("Identifikasi DNA", type="primary")
with col2:
    st.caption(f"Panjang sequence saat ini: {len(clean_sequence(sequence))} karakter")

if predict:
    valid, message = validate_sequence(sequence)
    if not valid:
        st.error(message)
        st.stop()

    X = build_feature_frame([sequence], mutation_flags=[mutation_flag]).reindex(columns=FEATURE_COLUMNS)

    class_prediction = class_model.predict(X)[0]
    severity_prediction = severity_model.predict(X)[0]
    high_risk_prediction = high_risk_model.predict(X)[0]

    class_prob = probability_table(class_model, X)
    severity_prob = probability_table(severity_model, X)
    high_prob = probability_table(high_risk_model, X)

    high_risk_score = None
    if not high_prob.empty:
        row = high_prob[high_prob["Label"] == "1"]
        if not row.empty:
            high_risk_score = float(row["Probabilitas"].iloc[0])

    st.subheader("Hasil Identifikasi")
    result_1, result_2, result_3 = st.columns(3)
    with result_1:
        st.metric("Kelas DNA", str(class_prediction))
    with result_2:
        st.metric("Estimasi Keparahan", str(severity_prediction))
    with result_3:
        st.metric("Status High Risk", "High" if int(high_risk_prediction) == 1 else "Not High")
        if high_risk_score is not None:
            st.caption(f"Probabilitas high risk: {format_percent(high_risk_score)}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Probabilitas Kelas",
        "Probabilitas Keparahan",
        "Probabilitas High Risk",
        "Fitur DNA",
    ])
    with tab1:
        st.dataframe(class_prob, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(severity_prob, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(high_prob, use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(X.T.rename(columns={0: "Nilai"}), use_container_width=True)

    st.info(
        "Interpretasi: hasil kelas DNA dan tingkat keparahan merupakan prediksi model dari pola dataset. "
        "Untuk penelitian serius, gunakan dataset biologis/klinis yang tervalidasi."
    )

st.divider()
st.caption("Prototype machine learning untuk klasifikasi sequence DNA berbasis Streamlit.")

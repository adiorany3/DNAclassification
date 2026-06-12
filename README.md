# DNA Sequence Classifier — Streamlit Online

Project ini adalah versi baru dan diperbaiki dari notebook `dna-classification.ipynb` agar dapat digunakan untuk training model dan integrasi aplikasi Streamlit online.

## Perbaikan Utama

1. Path Kaggle diganti menjadi path lokal project.
2. Kolom leakage seperti `Class_Bacteria`, `Class_Human`, `Class_Plant`, `Class_Virus`, dan `Disease_Risk_Encoded` tidak dipakai sebagai fitur model.
3. Training dibuat ulang dalam `train_model.py`.
4. Ekstraksi fitur DNA dipisah ke `feature_extraction.py` agar konsisten antara training dan Streamlit.
5. Model disimpan ke folder `models/` dalam format `.joblib`.
6. Aplikasi Streamlit dibuat dalam `app.py`.

## Struktur Folder

```text
dna_streamlit_project/
├── app.py
├── train_model.py
├── feature_extraction.py
├── processed_dna_dataset.csv
├── requirements.txt
├── README.md
├── models/
│   ├── organism_classifier.joblib
│   ├── severity_classifier.joblib
│   ├── high_risk_classifier.joblib
│   ├── metadata.joblib
│   └── metrics.json
└── notebooks/
    ├── dna_classification_training_fixed.ipynb
    └── dna-classification-original.ipynb
```

## Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Cara Deploy ke Streamlit Community Cloud

1. Upload semua file project ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository GitHub.
4. Isi **Main file path** dengan `app.py`.
5. Klik Deploy.

## Catatan Penting

Prediksi `Disease_Risk` atau tingkat keparahan adalah estimasi berbasis dataset, bukan diagnosis medis final. Untuk penelitian medis/biologis, dataset harus divalidasi oleh ahli dan memiliki dasar klinis atau biologis yang jelas.

## Catatan perbaikan v2

Jika menjalankan notebook di Kaggle/Jupyter dan muncul error:

```python
ModuleNotFoundError: No module named 'feature_extraction'
```

Gunakan notebook `dna_classification_training_fixed_v2_self_contained.ipynb` atau `dna_classification_training_fixed.ipynb` versi terbaru di zip ini. Fungsi ekstraksi fitur sudah dimasukkan langsung ke dalam notebook, sehingga tidak perlu import `feature_extraction.py` saat training di notebook.

Untuk aplikasi Streamlit, file `feature_extraction.py` tetap disediakan karena dibutuhkan oleh `app.py` dan `train_model.py`.

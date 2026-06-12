"""Training script for DNA Streamlit application.

Run from the project folder:
    python train_model.py

This v4 version uses compact compressed models so every model file is safe for
GitHub browser upload limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from feature_extraction import FEATURE_COLUMNS, validate_sequence

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_dna_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

LEAKAGE_COLUMNS = [
    "Disease_Risk_Encoded",
    "Class_Bacteria",
    "Class_Human",
    "Class_Plant",
    "Class_Virus",
]


def load_and_validate_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = ["Sequence", "Class_Label", "Disease_Risk", *FEATURE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    valid_mask = []
    for seq in df["Sequence"].astype(str):
        valid, _ = validate_sequence(seq)
        valid_mask.append(valid)
    return df.loc[valid_mask].reset_index(drop=True)


def evaluate_model(X_train, X_test, y_train, y_test, model):
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)
    pred = fitted_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    return result, fitted_model, pred


def save_model(model, filename: str) -> None:
    # compress=3 keeps files small but still fast to load in Streamlit Cloud.
    joblib.dump(model, MODEL_DIR / filename, compress=3)


def main():
    df = load_and_validate_dataset()
    X = df[FEATURE_COLUMNS].copy()
    y_class = df["Class_Label"].astype(str)
    y_risk = df["Disease_Risk"].astype(str)
    y_high_risk = (df["Disease_Risk"].astype(str).str.lower() == "high").astype(int)

    compact_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
    )
    class_result, class_model, class_pred = evaluate_model(
        X_train, X_test, y_train, y_test, compact_model
    )

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_risk, test_size=0.2, random_state=RANDOM_STATE, stratify=y_risk
    )
    risk_result, risk_model, risk_pred = evaluate_model(
        Xr_train, Xr_test, yr_train, yr_test, compact_model
    )

    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
        X, y_high_risk, test_size=0.2, random_state=RANDOM_STATE, stratify=y_high_risk
    )
    high_result, high_model, high_pred = evaluate_model(
        Xh_train, Xh_test, yh_train, yh_test, compact_model
    )

    high_auc = None
    if hasattr(high_model, "predict_proba"):
        try:
            high_auc = float(roc_auc_score(yh_test, high_model.predict_proba(Xh_test)[:, 1]))
        except Exception:
            high_auc = None

    metadata = {
        "dataset_rows_used": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "class_labels": sorted(y_class.unique().tolist()),
        "risk_labels": sorted(y_risk.unique().tolist()),
        "best_class_model": "Compact RandomForestClassifier",
        "best_risk_model": "Compact RandomForestClassifier",
        "best_high_risk_model": "Compact RandomForestClassifier",
        "high_risk_auc": high_auc,
        "important_note": (
            "Disease_Risk adalah label prediktif dari dataset, bukan diagnosis medis. "
            "Gunakan hasil sebagai prototipe/pendukung analisis, bukan keputusan klinis final."
        ),
    }

    metrics = {
        "class_model_results": [class_result],
        "risk_model_results": [risk_result],
        "high_risk_model_results": [high_result],
        "best_classification_report": classification_report(y_test, class_pred, output_dict=True, zero_division=0),
        "best_risk_report": classification_report(yr_test, risk_pred, output_dict=True, zero_division=0),
        "best_high_risk_report": classification_report(yh_test, high_pred, output_dict=True, zero_division=0),
        "class_confusion_matrix": confusion_matrix(y_test, class_pred, labels=metadata["class_labels"]).tolist(),
        "risk_confusion_matrix": confusion_matrix(yr_test, risk_pred, labels=metadata["risk_labels"]).tolist(),
        "high_risk_confusion_matrix": confusion_matrix(yh_test, high_pred, labels=[0, 1]).tolist(),
        "metadata": metadata,
    }

    save_model(class_model, "organism_classifier.joblib")
    save_model(risk_model, "severity_classifier.joblib")
    save_model(high_model, "high_risk_classifier.joblib")
    save_model(metadata, "metadata.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training selesai.")
    print(f"Akurasi klasifikasi DNA: {class_result['accuracy']:.4f}")
    print(f"Akurasi tingkat keparahan: {risk_result['accuracy']:.4f}")
    print(f"Akurasi high-risk detector: {high_result['accuracy']:.4f}")
    print("File model tersimpan di folder models/ dengan ukuran kecil.")


if __name__ == "__main__":
    main()

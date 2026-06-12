"""Training script for DNA Streamlit application.

Run from the project folder:
    python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from feature_extraction import FEATURE_COLUMNS, validate_sequence

RANDOM_STATE = 42
DATA_PATH = Path("processed_dna_dataset.csv")
MODEL_DIR = Path("models")
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
    df = df.loc[valid_mask].reset_index(drop=True)

    # Keep only the feature columns that are safe for modeling.
    # Label/leakage columns are not included in X.
    return df


def evaluate_models(X_train, X_test, y_train, y_test, candidates, labels=None):
    rows = []
    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, candidate_model in candidates.items():
        model = clone(candidate_model)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0
        )
        rows.append({
            "model": name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        })
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    best_pred = best_model.predict(X_test)
    return rows, best_name, best_model, best_pred


def main():
    df = load_and_validate_dataset()
    X = df[FEATURE_COLUMNS].copy()
    y_class = df["Class_Label"].astype(str)
    y_risk = df["Disease_Risk"].astype(str)
    y_high_risk = (df["Disease_Risk"].astype(str).str.lower() == "high").astype(int)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=220,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=220,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
    )
    class_results, best_class_name, best_class_model, class_pred = evaluate_models(
        X_train, X_test, y_train, y_test, candidates
    )

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_risk, test_size=0.2, random_state=RANDOM_STATE, stratify=y_risk
    )
    risk_results, best_risk_name, best_risk_model, risk_pred = evaluate_models(
        Xr_train, Xr_test, yr_train, yr_test, candidates
    )

    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
        X, y_high_risk, test_size=0.2, random_state=RANDOM_STATE, stratify=y_high_risk
    )
    high_results, best_high_name, best_high_model, high_pred = evaluate_models(
        Xh_train, Xh_test, yh_train, yh_test, candidates
    )

    high_auc = None
    if hasattr(best_high_model, "predict_proba"):
        try:
            high_auc = float(roc_auc_score(yh_test, best_high_model.predict_proba(Xh_test)[:, 1]))
        except Exception:
            high_auc = None

    metadata = {
        "dataset_rows_used": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "class_labels": sorted(y_class.unique().tolist()),
        "risk_labels": sorted(y_risk.unique().tolist()),
        "best_class_model": best_class_name,
        "best_risk_model": best_risk_name,
        "best_high_risk_model": best_high_name,
        "high_risk_auc": high_auc,
        "important_note": (
            "Disease_Risk adalah label prediktif dari dataset, bukan diagnosis medis. "
            "Gunakan hasil sebagai prototipe/pendukung analisis, bukan keputusan klinis final."
        ),
    }

    metrics = {
        "class_model_results": class_results,
        "risk_model_results": risk_results,
        "high_risk_model_results": high_results,
        "best_classification_report": classification_report(y_test, class_pred, output_dict=True, zero_division=0),
        "best_risk_report": classification_report(yr_test, risk_pred, output_dict=True, zero_division=0),
        "best_high_risk_report": classification_report(yh_test, high_pred, output_dict=True, zero_division=0),
        "class_confusion_matrix": confusion_matrix(y_test, class_pred, labels=metadata["class_labels"]).tolist(),
        "risk_confusion_matrix": confusion_matrix(yr_test, risk_pred, labels=metadata["risk_labels"]).tolist(),
        "high_risk_confusion_matrix": confusion_matrix(yh_test, high_pred, labels=[0, 1]).tolist(),
        "metadata": metadata,
    }

    joblib.dump(best_class_model, MODEL_DIR / "organism_classifier.joblib")
    joblib.dump(best_risk_model, MODEL_DIR / "severity_classifier.joblib")
    joblib.dump(best_high_model, MODEL_DIR / "high_risk_classifier.joblib")
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training selesai.")
    print(f"Best model klasifikasi DNA: {best_class_name}")
    print(f"Best model tingkat keparahan: {best_risk_name}")
    print(f"Best model high-risk detector: {best_high_name}")
    print("File model tersimpan di folder models/.")


if __name__ == "__main__":
    main()

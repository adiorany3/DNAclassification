"""DNA feature extraction for Streamlit and model training.

This file keeps training and online prediction consistent.
Only non-label features are used. Label leakage columns such as
Class_Bacteria, Class_Human, Class_Plant, Class_Virus, and Disease_Risk_Encoded
are intentionally excluded from the model.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List

import pandas as pd

VALID_BASES = {"A", "T", "C", "G"}
FEATURE_COLUMNS: List[str] = [
    "GC_Content",
    "AT_Content",
    "Sequence_Length",
    "Num_A",
    "Num_T",
    "Num_C",
    "Num_G",
    "kmer_3_freq",
    "Mutation_Flag",
    "GC_to_AT_Ratio",
    "Sequence_Entropy",
]


def clean_sequence(sequence: str) -> str:
    """Return an uppercase DNA sequence without whitespace."""
    if sequence is None:
        return ""
    return "".join(str(sequence).upper().split())


def validate_sequence(sequence: str) -> tuple[bool, str]:
    """Validate DNA input sequence."""
    seq = clean_sequence(sequence)
    if not seq:
        return False, "Sequence masih kosong. Masukkan sequence DNA terlebih dahulu."
    invalid = sorted(set(seq) - VALID_BASES)
    if invalid:
        return False, f"Sequence mengandung karakter tidak valid: {', '.join(invalid)}. Gunakan hanya A, T, C, dan G."
    if len(seq) < 3:
        return False, "Sequence minimal 3 karakter agar fitur 3-mer dapat dihitung."
    return True, "OK"


def sequence_entropy(sequence: str) -> float:
    """Calculate Shannon entropy of A/T/C/G composition."""
    seq = clean_sequence(sequence)
    total = len(seq)
    if total == 0:
        return 0.0
    counts = Counter(seq)
    entropy = 0.0
    for base in VALID_BASES:
        count = counts.get(base, 0)
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def kmer_3_unique_frequency(sequence: str) -> float:
    """Ratio of unique 3-mers to all possible observed 3-mer windows."""
    seq = clean_sequence(sequence)
    total = max(len(seq) - 3 + 1, 0)
    if total == 0:
        return 0.0
    kmers = {seq[i:i + 3] for i in range(total)}
    return len(kmers) / total


def extract_features_from_sequence(sequence: str, mutation_flag: int = 0) -> Dict[str, float]:
    """Extract numeric features from a DNA sequence.

    mutation_flag is optional because mutation status usually comes from
    laboratory annotation/reference comparison, not from the raw sequence alone.
    """
    seq = clean_sequence(sequence)
    length = len(seq)
    counts = Counter(seq)
    num_a = counts.get("A", 0)
    num_t = counts.get("T", 0)
    num_c = counts.get("C", 0)
    num_g = counts.get("G", 0)
    gc = num_g + num_c
    at = num_a + num_t

    return {
        "GC_Content": round((gc / length * 100), 6) if length else 0.0,
        "AT_Content": round((at / length * 100), 6) if length else 0.0,
        "Sequence_Length": float(length),
        "Num_A": float(num_a),
        "Num_T": float(num_t),
        "Num_C": float(num_c),
        "Num_G": float(num_g),
        "kmer_3_freq": round(kmer_3_unique_frequency(seq), 6),
        "Mutation_Flag": int(mutation_flag),
        "GC_to_AT_Ratio": round((gc / at), 6) if at else 0.0,
        "Sequence_Entropy": round(sequence_entropy(seq), 6),
    }


def build_feature_frame(sequences: Iterable[str], mutation_flags: Iterable[int] | None = None) -> pd.DataFrame:
    """Build a model-ready feature DataFrame with stable column order."""
    seqs = list(sequences)
    if mutation_flags is None:
        flags = [0] * len(seqs)
    else:
        flags = list(mutation_flags)
        if len(flags) != len(seqs):
            raise ValueError("Jumlah mutation_flags harus sama dengan jumlah sequences.")
    rows = [extract_features_from_sequence(seq, flag) for seq, flag in zip(seqs, flags)]
    return pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

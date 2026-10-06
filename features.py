
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

STOPWORDS = {
    "a","an","the","is","are","was","were","am","to","of","in","on","for",
    "and","or","as","at","by","with","from","this","that","it","be","do",
    "does","did","how","what","why","when","where","who","which","can",
    "could","would","should","will","i","you","he","she","they","we"
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return [x for x in clean_text(text).split() if x]

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def common_token_features(q1, q2):
    a = set(tokenize(q1))
    b = set(tokenize(q2))
    common = a & b
    union = a | b
    return [
        len(common),
        safe_div(len(common), len(a)),
        safe_div(len(common), len(b)),
        safe_div(len(common), len(union)),
        safe_div(len(common), min(len(a), len(b))) if a and b else 0.0,
    ]

def pair_features(q1, q2):
    q1 = clean_text(q1)
    q2 = clean_text(q2)

    t1 = tokenize(q1)
    t2 = tokenize(q2)
    s1, s2 = set(t1), set(t2)

    common = s1 & s2
    union = s1 | s2
    non_stop_1 = {x for x in t1 if x not in STOPWORDS}
    non_stop_2 = {x for x in t2 if x not in STOPWORDS}

    l1, l2 = len(t1), len(t2)
    c1, c2 = len(q1), len(q2)

    # 21 engineered features
    f = [
        l1, l2,
        abs(l1-l2),
        safe_div(min(l1,l2), max(l1,l2)),
        c1, c2,
        abs(c1-c2),
        safe_div(min(c1,c2), max(c1,c2)),
        len(common),
        safe_div(len(common), len(s1)),
        safe_div(len(common), len(s2)),
        safe_div(len(common), len(union)),
        safe_div(len(common), min(l1,l2)) if l1 and l2 else 0.0,
        len(non_stop_1 & non_stop_2),
        safe_div(len(non_stop_1 & non_stop_2), len(non_stop_1)),
        safe_div(len(non_stop_1 & non_stop_2), len(non_stop_2)),
        fuzz.ratio(q1, q2) / 100.0,
        fuzz.token_sort_ratio(q1, q2) / 100.0,
        fuzz.token_set_ratio(q1, q2) / 100.0,
        fuzz.partial_ratio(q1, q2) / 100.0,
        float(q1 == q2),
    ]
    return np.asarray(f, dtype=np.float32)

def build_handcrafted_features(df):
    return np.vstack([
        pair_features(a, b)
        for a, b in zip(df["question1"], df["question2"])
    ]).astype(np.float32)

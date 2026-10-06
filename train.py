
import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

try:
    from .features import clean_text, build_handcrafted_features
except ImportError:
    from features import clean_text, build_handcrafted_features

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

HANDCRAFTED_NAMES = [
    "q1_word_len","q2_word_len","word_len_diff","word_len_ratio",
    "q1_char_len","q2_char_len","char_len_diff","char_len_ratio",
    "common_tokens","common_q1_ratio","common_q2_ratio","jaccard",
    "common_min_len_ratio","common_nonstop","nonstop_q1_ratio",
    "nonstop_q2_ratio","fuzz_ratio","fuzz_token_sort","fuzz_token_set",
    "fuzz_partial","exact_match"
]

def find_csv():
    candidates = list(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {DATA_DIR}. Put your Kaggle CSV there."
        )
    # Prefer the standard Quora file if present.
    for p in candidates:
        if "quora" in p.name.lower():
            return p
    return candidates[0]

def load_data(path):
    df = pd.read_csv(path)
    required = {"question1", "question2", "is_duplicate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["question1", "question2", "is_duplicate"]].copy()
    df["question1"] = df["question1"].fillna("").map(clean_text)
    df["question2"] = df["question2"].fillna("").map(clean_text)
    df["is_duplicate"] = pd.to_numeric(df["is_duplicate"], errors="coerce")
    df = df.dropna(subset=["is_duplicate"])
    df = df[df["is_duplicate"].isin([0, 1])]
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def make_split(df):
    # For the standard Quora CSV this is a reproducible stratified split.
    # If qid columns are available, a graph/group split is preferable for
    # research-grade leakage prevention; this project intentionally uses
    # the common benchmark split so results are comparable.
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["is_duplicate"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["is_duplicate"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def fit_tfidf(train_df):
    all_text = pd.concat([train_df["question1"], train_df["question2"]], ignore_index=True)

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        strip_accents="unicode"
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )
    word.fit(all_text)
    char.fit(all_text)
    return word, char

def tfidf_pair_features(df, word, char):
    w1 = word.transform(df["question1"])
    w2 = word.transform(df["question2"])
    c1 = char.transform(df["question1"])
    c2 = char.transform(df["question2"])

    # Cosine similarity for normalized TF-IDF vectors.
    word_cos = np.asarray(w1.multiply(w2).sum(axis=1)).ravel()
    char_cos = np.asarray(c1.multiply(c2).sum(axis=1)).ravel()

    return np.column_stack([word_cos, char_cos]).astype(np.float32)

def encode_semantic(df, model, batch_size=64):
    q1 = df["question1"].tolist()
    q2 = df["question2"].tolist()

    e1 = model.encode(
        q1, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )
    e2 = model.encode(
        q2, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )

    # Four compact semantic features.
    cosine = np.sum(e1 * e2, axis=1)
    abs_diff = np.mean(np.abs(e1 - e2), axis=1)
    l2 = np.linalg.norm(e1 - e2, axis=1)
    product = np.mean(e1 * e2, axis=1)

    return np.column_stack([cosine, abs_diff, l2, product]).astype(np.float32)

def build_features(df, word, char, semantic_model):
    handcrafted = build_handcrafted_features(df)
    tfidf = tfidf_pair_features(df, word, char)
    semantic = encode_semantic(df, semantic_model)
    return np.hstack([handcrafted, tfidf, semantic]).astype(np.float32)

def choose_threshold(y_val, p_val):
    best_t, best_acc = 0.50, 0.0
    for t in np.arange(0.30, 0.701, 0.005):
        pred = (p_val >= t).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc

def evaluate(name, y, p, threshold):
    pred = (p >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y, p)
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=4))
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="Optional path to CSV")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else find_csv()
    print(f"Loading dataset: {data_path}")
    df = load_data(data_path)

    print(f"Rows after cleaning: {len(df):,}")
    print(df["is_duplicate"].value_counts(normalize=True).sort_index())

    train_df, val_df, test_df = make_split(df)
    print(
        f"Train={len(train_df):,} | "
        f"Validation={len(val_df):,} | Test={len(test_df):,}"
    )

    # ---------------- BASIC MODEL ----------------
    print("\nTraining BASIC TF-IDF model...")
    word_basic = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2), min_df=2,
        max_features=150000, sublinear_tf=True
    )
    char_basic = TfidfVectorizer(
        analyzer="char", ngram_range=(2,5), min_df=2,
        max_features=120000, sublinear_tf=True
    )

    basic_text_train = (
        train_df["question1"] + " [SEP] " + train_df["question2"]
    )
    basic_text_val = (
        val_df["question1"] + " [SEP] " + val_df["question2"]
    )
    basic_text_test = (
        test_df["question1"] + " [SEP] " + test_df["question2"]
    )

    # Logistic regression on the full concatenated text is a baseline.
    # Use a separate compact cosine-feature baseline to keep RAM reasonable.
    word_pair = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2), min_df=2,
        max_features=100000, sublinear_tf=True
    )
    char_pair = TfidfVectorizer(
        analyzer="char", ngram_range=(2,5), min_df=2,
        max_features=80000, sublinear_tf=True
    )
    word_pair.fit(pd.concat([train_df.question1, train_df.question2]))
    char_pair.fit(pd.concat([train_df.question1, train_df.question2]))

    basic_train = np.hstack([
        build_handcrafted_features(train_df),
        tfidf_pair_features(train_df, word_pair, char_pair)
    ])
    basic_val = np.hstack([
        build_handcrafted_features(val_df),
        tfidf_pair_features(val_df, word_pair, char_pair)
    ])
    basic_test = np.hstack([
        build_handcrafted_features(test_df),
        tfidf_pair_features(test_df, word_pair, char_pair)
    ])

    basic_model = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=2.0, solver="lbfgs"
    )
    basic_model.fit(basic_train, train_df.is_duplicate.astype(int))

    p_val_basic = basic_model.predict_proba(basic_val)[:,1]
    basic_threshold, _ = choose_threshold(val_df.is_duplicate.values, p_val_basic)
    p_test_basic = basic_model.predict_proba(basic_test)[:,1]
    evaluate("BASIC MODEL", test_df.is_duplicate.values, p_test_basic, basic_threshold)

    joblib.dump(basic_model, ART_DIR / "basic_model.pkl")
    joblib.dump(word_pair, ART_DIR / "basic_word_tfidf.pkl")
    joblib.dump(char_pair, ART_DIR / "basic_char_tfidf.pkl")

    # ---------------- ADVANCED MODEL ----------------
    print("\nLoading semantic model...")
    semantic_name = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_model = SentenceTransformer(semantic_name)

    print("\nBuilding ADVANCED features...")
    X_train = build_features(train_df, word_pair, char_pair, semantic_model)
    X_val = build_features(val_df, word_pair, char_pair, semantic_model)
    X_test = build_features(test_df, word_pair, char_pair, semantic_model)

    feature_names = HANDCRAFTED_NAMES + [
        "word_tfidf_cosine", "char_tfidf_cosine",
        "semantic_cosine", "semantic_abs_diff",
        "semantic_l2", "semantic_product"
    ]

    model = XGBClassifier(
        n_estimators=900,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED
    )

    model.fit(
        X_train, train_df.is_duplicate.astype(int),
        eval_set=[(X_val, val_df.is_duplicate.astype(int))],
        verbose=False
    )

    p_val = model.predict_proba(X_val)[:,1]
    threshold, val_acc = choose_threshold(
        val_df.is_duplicate.values, p_val
    )
    print(f"\nValidation threshold selected: {threshold:.3f}")
    print(f"Validation accuracy at threshold: {val_acc*100:.2f}%")

    p_test = model.predict_proba(X_test)[:,1]
    test_acc = evaluate(
        "ADVANCED MODEL", test_df.is_duplicate.values, p_test, threshold
    )

    print("\nTarget check:")
    print(f"Target accuracy: 95.00%")
    print(f"Actual test accuracy: {test_acc*100:.2f}%")
    if test_acc >= 0.95:
        print("PASS: test accuracy reached the 95% target.")
    else:
        print("The 95% target was not reached on this held-out test set.")
        print("Do not change the test set or tune the threshold on test data.")

    joblib.dump(model, ART_DIR / "advanced_xgboost.pkl")
    joblib.dump(word_pair, ART_DIR / "word_tfidf.pkl")
    joblib.dump(char_pair, ART_DIR / "char_tfidf.pkl")

    config = {
        "semantic_model": semantic_name,
        "threshold": threshold,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "seed": SEED,
        "basic_threshold": basic_threshold,
        "test_accuracy": float(test_acc)
    }
    with open(ART_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nSaved artifacts:")
    for p in sorted(ART_DIR.iterdir()):
        print(" -", p.name)

if __name__ == "__main__":
    main()


import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from .features import clean_text, build_handcrafted_features
from .train import tfidf_pair_features, encode_semantic

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

class QuestionSimilarity:
    def __init__(self):
        self.model = joblib.load(ART / "advanced_xgboost.pkl")
        self.word = joblib.load(ART / "word_tfidf.pkl")
        self.char = joblib.load(ART / "char_tfidf.pkl")
        with open(ART / "model_config.json", encoding="utf-8") as f:
            self.config = json.load(f)
        self.semantic = SentenceTransformer(self.config["semantic_model"])

    def predict(self, question1, question2):
        df = pd.DataFrame({
            "question1": [clean_text(question1)],
            "question2": [clean_text(question2)]
        })
        handcrafted = build_handcrafted_features(df)
        tfidf = tfidf_pair_features(df, self.word, self.char)
        semantic = encode_semantic(df, self.semantic, batch_size=1)
        X = np.hstack([handcrafted, tfidf, semantic]).astype(np.float32)

        probability = float(self.model.predict_proba(X)[0,1])
        threshold = float(self.config["threshold"])
        duplicate = probability >= threshold

        return {
            "is_duplicate": bool(duplicate),
            "probability": probability,
            "similarity_percent": probability * 100.0,
            "threshold": threshold
        }

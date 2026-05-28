from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
HTML_RE = re.compile(r"<[^>]+>")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
REPEAT_RE = re.compile(r"(.)\1{3,}")

SPAM_WORDS = {
    "spam",
    "free",
    "winner",
    "win",
    "urgent",
    "immediately",
    "offer",
    "discount",
    "bonus",
    "click",
    "buy",
    "cheap",
    "trial",
    "limited",
    "credit",
    "money",
}
PROMO_WORDS = {
    "free",
    "winner",
    "winning",
    "win",
    "urgent",
    "now",
    "immediately",
    "offer",
    "discount",
    "sale",
    "save",
    "deal",
    "cheap",
    "credit",
    "bonus",
    "gift",
    "prize",
}
UNSUB_WORDS = {"unsubscribe", "opt out", "remove", "stop receiving", "manage preferences"}


def _count_terms(text: str, terms: Iterable[str]) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def extract_structured_features(texts: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for value in texts:
        text = "" if value is None else str(value)
        words = len(re.findall(r"\b\w+\b", text))
        chars = len(text)
        alpha = sum(char.isalpha() for char in text)
        upper = sum(char.isupper() for char in text)
        digits = sum(char.isdigit() for char in text)
        urls = len(URL_RE.findall(text))
        html = len(HTML_RE.findall(text))
        punctuation = sum(char in "!?.,;:-_()[]{}" for char in text)
        rows.append(
            {
                "char_len": chars,
                "word_count": words,
                "sentence_count": max(1, len(re.findall(r"[.!?]+", text))),
                "exclamation_count": text.count("!"),
                "question_count": text.count("?"),
                "upper_ratio": upper / max(1, alpha),
                "digit_ratio": digits / max(1, chars),
                "url_count": urls,
                "email_count": len(EMAIL_RE.findall(text)),
                "html_tag_count": html,
                "has_html": int("<html" in text.lower() or "<body" in text.lower() or html > 0),
                "attachment_hint": int(any(key in text.lower() for key in ["attach", "attached", "attachment", "file"])),
                "unsubscribe_count": _count_terms(text, UNSUB_WORDS),
                "money_symbol_count": text.count("$"),
                "phone_pattern_count": len(PHONE_RE.findall(text)),
                "punctuation_density": punctuation / max(1, chars),
                "spam_keyword_count": _count_terms(text, SPAM_WORDS),
                "promo_keyword_count": _count_terms(text, PROMO_WORDS),
                "too_many_links": int(urls >= 3),
                "repeated_char_flag": int(bool(REPEAT_RE.search(text))),
            }
        )
    return pd.DataFrame(rows)


def _safe_svd_components(requested: int, shape: tuple[int, int]) -> int:
    return max(1, min(requested, shape[0] - 1, shape[1] - 1))


@dataclass
class FeatureBundle:
    feature_type: str
    word_vectorizer: TfidfVectorizer | None = None
    char_vectorizer: TfidfVectorizer | None = None
    structured_scaler: StandardScaler | None = None
    word_svd: TruncatedSVD | None = None
    char_svd: TruncatedSVD | None = None
    structured_columns: list[str] | None = None

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        text_values = list(texts)
        parts: list[np.ndarray | sparse.spmatrix] = []

        if self.feature_type in {"word", "hybrid"}:
            if self.word_vectorizer is None:
                raise RuntimeError("word vectorizer is not fitted")
            parts.append(self.word_vectorizer.transform(text_values))

        if self.feature_type in {"char", "hybrid"}:
            if self.char_vectorizer is None:
                raise RuntimeError("char vectorizer is not fitted")
            parts.append(self.char_vectorizer.transform(text_values))

        if self.feature_type in {"structured", "hybrid", "word_svd", "char_svd", "hybrid_svd"}:
            structured = extract_structured_features(text_values)
            if self.structured_columns is not None:
                structured = structured[self.structured_columns]
            if self.structured_scaler is None:
                raise RuntimeError("structured scaler is not fitted")
            structured_values = self.structured_scaler.transform(structured.fillna(0.0).values)
            if self.feature_type in {"structured", "hybrid"}:
                parts.append(sparse.csr_matrix(structured_values))

        if self.feature_type in {"word_svd", "hybrid_svd"}:
            if self.word_vectorizer is None or self.word_svd is None:
                raise RuntimeError("word SVD pipeline is not fitted")
            parts.append(self.word_svd.transform(self.word_vectorizer.transform(text_values)))

        if self.feature_type in {"char_svd", "hybrid_svd"}:
            if self.char_vectorizer is None or self.char_svd is None:
                raise RuntimeError("char SVD pipeline is not fitted")
            parts.append(self.char_svd.transform(self.char_vectorizer.transform(text_values)))

        if self.feature_type in {"word_svd", "char_svd", "hybrid_svd"}:
            parts.append(structured_values)

        if not parts:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        if any(sparse.issparse(part) for part in parts):
            return sparse.hstack(parts).astype(np.float32).toarray()
        return np.hstack(parts).astype(np.float32)


def fit_feature_bundle(
    texts: Iterable[str],
    feature_type: str = "word",
    word_features: int = 30000,
    char_features: int = 30000,
    svd_components: int = 200,
) -> tuple[np.ndarray, FeatureBundle]:
    text_values = list(texts)
    bundle = FeatureBundle(feature_type=feature_type)
    parts: list[np.ndarray | sparse.spmatrix] = []

    if feature_type in {"word", "hybrid", "word_svd", "hybrid_svd"}:
        word_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=word_features)
        word_matrix = word_vectorizer.fit_transform(text_values)
        bundle.word_vectorizer = word_vectorizer
        if feature_type in {"word", "hybrid"}:
            parts.append(word_matrix)
        else:
            word_svd = TruncatedSVD(n_components=_safe_svd_components(svd_components, word_matrix.shape), random_state=42)
            parts.append(word_svd.fit_transform(word_matrix))
            bundle.word_svd = word_svd

    if feature_type in {"char", "hybrid", "char_svd", "hybrid_svd"}:
        char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=char_features,
        )
        char_matrix = char_vectorizer.fit_transform(text_values)
        bundle.char_vectorizer = char_vectorizer
        if feature_type in {"char", "hybrid"}:
            parts.append(char_matrix)
        else:
            char_svd = TruncatedSVD(n_components=_safe_svd_components(svd_components, char_matrix.shape), random_state=42)
            parts.append(char_svd.fit_transform(char_matrix))
            bundle.char_svd = char_svd

    if feature_type in {"structured", "hybrid", "word_svd", "char_svd", "hybrid_svd"}:
        structured = extract_structured_features(text_values)
        scaler = StandardScaler(with_mean=feature_type != "hybrid")
        structured_values = scaler.fit_transform(structured.fillna(0.0).values)
        bundle.structured_scaler = scaler
        bundle.structured_columns = structured.columns.tolist()
        if feature_type in {"structured", "hybrid"}:
            parts.append(sparse.csr_matrix(structured_values))
        else:
            parts.append(structured_values)

    if not parts:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    if any(sparse.issparse(part) for part in parts):
        return sparse.hstack(parts).astype(np.float32).toarray(), bundle
    return np.hstack(parts).astype(np.float32), bundle

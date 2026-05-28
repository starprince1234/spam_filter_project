from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
HTML_RE = re.compile(r"<[^>]+>")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
REPEAT_RE = re.compile(r"(.)\1{3,}")

SPAM_WORDS = {
    "spam", "free", "winner", "win", "urgent", "immediately", "offer", "discount",
    "bonus", "click", "buy", "cheap", "trial", "limited", "credit", "money",
}
PROMO_WORDS = {
    "free", "winner", "winning", "win", "urgent", "now", "immediately", "offer",
    "discount", "sale", "save", "deal", "cheap", "credit", "bonus", "gift", "prize",
}
UNSUB_WORDS = {"unsubscribe", "opt out", "remove", "stop receiving", "manage preferences"}


def _count_terms(text: str, terms: Iterable[str]) -> int:
    lower = text.lower()
    return sum(lower.count(t) for t in terms)


def extract_structured_features(texts: Iterable[str]) -> pd.DataFrame:
    rows = []
    for value in texts:
        text = "" if value is None else str(value)
        words = len(re.findall(r"\b\w+\b", text))
        chars = len(text)
        alpha = sum(c.isalpha() for c in text)
        upper = sum(c.isupper() for c in text)
        digits = sum(c.isdigit() for c in text)
        urls = len(URL_RE.findall(text))
        html = len(HTML_RE.findall(text))
        punctuation = sum(c in "!?.,;:-_()[]{}" for c in text)
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
                "attachment_hint": int(any(k in text.lower() for k in ["attach", "attached", "attachment", "file"])),
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


def _safe_components(requested: int, shape: tuple[int, int]) -> int:
    return max(1, min(requested, shape[0] - 1, shape[1] - 1))


@dataclass
class FeatureBundle:
    feature_type: str
    structured_scaler: StandardScaler | None = None
    word_vectorizer: TfidfVectorizer | None = None
    char_vectorizer: TfidfVectorizer | None = None
    word_svd: TruncatedSVD | None = None
    char_svd: TruncatedSVD | None = None
    structured_columns: list[str] | None = None

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        parts = []
        if self.feature_type in {"structured", "hybrid"}:
            structured = extract_structured_features(texts)
            if self.structured_columns:
                structured = structured[self.structured_columns]
            parts.append(self.structured_scaler.transform(structured.fillna(0.0).values))
        if self.feature_type in {"word_svd", "hybrid"}:
            x_word = self.word_vectorizer.transform(texts)
            parts.append(self.word_svd.transform(x_word))
        if self.feature_type in {"char_svd", "hybrid"}:
            x_char = self.char_vectorizer.transform(texts)
            parts.append(self.char_svd.transform(x_char))
        if not parts:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        return np.hstack(parts).astype("float32")


def fit_feature_bundle(
    texts: Iterable[str],
    feature_type: str,
    word_features: int = 30000,
    char_features: int = 30000,
    svd_components: int = 200,
) -> tuple[np.ndarray, FeatureBundle]:
    texts = list(texts)
    bundle = FeatureBundle(feature_type=feature_type)
    parts = []

    if feature_type in {"structured", "hybrid"}:
        structured = extract_structured_features(texts)
        scaler = StandardScaler()
        parts.append(scaler.fit_transform(structured.fillna(0.0).values))
        bundle.structured_scaler = scaler
        bundle.structured_columns = structured.columns.tolist()

    if feature_type in {"word_svd", "hybrid"}:
        min_df = 1 if len(texts) < 20 else 2
        word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df, max_features=word_features)
        x_word = word_vec.fit_transform(texts)
        word_svd = TruncatedSVD(n_components=_safe_components(svd_components, x_word.shape), random_state=42)
        parts.append(word_svd.fit_transform(x_word))
        bundle.word_vectorizer = word_vec
        bundle.word_svd = word_svd

    if feature_type in {"char_svd", "hybrid"}:
        min_df = 1 if len(texts) < 20 else 2
        char_vec = TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, max_features=char_features
        )
        x_char = char_vec.fit_transform(texts)
        char_svd = TruncatedSVD(n_components=_safe_components(svd_components, x_char.shape), random_state=42)
        parts.append(char_svd.fit_transform(x_char))
        bundle.char_vectorizer = char_vec
        bundle.char_svd = char_svd

    if not parts:
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    return np.hstack(parts).astype("float32"), bundle

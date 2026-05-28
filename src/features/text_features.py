from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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


def _safe_text(x: str) -> str:
    return "" if x is None else str(x)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def _count_terms(text: str, terms: Iterable[str]) -> int:
    lower = text.lower()
    return sum(lower.count(t) for t in terms)


def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for text in df["clean_text"].fillna(""):
        text = _safe_text(text)
        words = _word_count(text)
        chars = len(text)
        alpha = sum(c.isalpha() for c in text)
        upper = sum(c.isupper() for c in text)
        digits = sum(c.isdigit() for c in text)
        exclam = text.count("!")
        quest = text.count("?")
        urls = len(URL_RE.findall(text))
        emails = len(EMAIL_RE.findall(text))
        html = len(HTML_RE.findall(text))
        has_html = int("<html" in text.lower() or "<body" in text.lower() or html > 0)
        attachment_hint = int(any(k in text.lower() for k in ["attach", "attached", "attachment", "file"]))
        unsub_count = _count_terms(text, UNSUB_WORDS)
        money = text.count("$")
        phone = len(PHONE_RE.findall(text))
        punctuation = sum(c in "!?.,;:-_()[]{}" for c in text)
        spam_kw = _count_terms(text, SPAM_WORDS)
        promo_kw = _count_terms(text, PROMO_WORDS)
        too_many_links = int(urls >= 3)
        repeated = int(bool(REPEAT_RE.search(text)))
        rows.append(
            {
                "char_len": chars,
                "word_count": words,
                "sentence_count": _sentence_count(text),
                "exclamation_count": exclam,
                "question_count": quest,
                "upper_ratio": upper / max(1, alpha),
                "digit_ratio": digits / max(1, chars),
                "url_count": urls,
                "email_count": emails,
                "html_tag_count": html,
                "has_html": has_html,
                "attachment_hint": attachment_hint,
                "unsubscribe_count": unsub_count,
                "money_symbol_count": money,
                "phone_pattern_count": phone,
                "punctuation_density": punctuation / max(1, chars),
                "spam_keyword_count": spam_kw,
                "promo_keyword_count": promo_kw,
                "too_many_links": too_many_links,
                "repeated_char_flag": repeated,
            }
        )
    return pd.DataFrame(rows)


def build_tfidf_word(texts: Iterable[str], ngram_range: tuple[int, int] = (1, 2), max_features: int = 30000):
    vec = TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=2, max_features=max_features)
    X = vec.fit_transform(texts)
    return X, vec


def build_tfidf_char(texts: Iterable[str], ngram_range: tuple[int, int] = (3, 5), max_features: int = 30000):
    vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=ngram_range, min_df=2, max_features=max_features)
    X = vec.fit_transform(texts)
    return X, vec


def build_hybrid_features(texts: Iterable[str], structured_df: pd.DataFrame):
    X_word, word_vec = build_tfidf_word(texts)
    X_char, char_vec = build_tfidf_char(texts)
    scaler = StandardScaler(with_mean=False)
    X_struct = scaler.fit_transform(structured_df.fillna(0.0).values)
    X = hstack([X_word, X_char, csr_matrix(X_struct)])
    return X, {"word": word_vec, "char": char_vec, "scaler": scaler}


def build_structured_only(structured_df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(structured_df.fillna(0.0).values)
    return X, scaler


def build_word_only(texts: Iterable[str]):
    return build_tfidf_word(texts)


def build_char_only(texts: Iterable[str]):
    return build_tfidf_char(texts)


def build_text_svd_features(texts: Iterable[str], structured_df: pd.DataFrame, n_components: int = 200):
    X_word, word_vec = build_tfidf_word(texts, max_features=50000)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_word)
    scaler = StandardScaler()
    X_struct = scaler.fit_transform(structured_df.fillna(0.0).values)
    X = np.hstack([X_svd, X_struct])
    return X, {"word": word_vec, "svd": svd, "scaler": scaler}

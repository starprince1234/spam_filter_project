from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


@dataclass
class ParsedEmail:
    raw_text: str
    clean_text: str
    subject: str
    sender: str
    date: str
    message_id: str
    has_html: bool
    html_text: str
    attachment_hint: bool


def _extract_payloads(msg) -> tuple[str, str, bool, bool]:
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachment_hint = False
    has_html = False

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get_content_disposition() or "").lower()
            filename = part.get_filename()
            if filename or disp == "attachment":
                attachment_hint = True
            if ctype == "text/plain":
                try:
                    text_parts.append(part.get_content())
                except Exception:
                    continue
            elif ctype == "text/html":
                has_html = True
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    continue
    else:
        ctype = msg.get_content_type()
        try:
            content = msg.get_content()
        except Exception:
            content = ""
        if ctype == "text/html":
            has_html = True
            html_parts.append(content)
        else:
            text_parts.append(content)

    text = "\n".join([t for t in text_parts if t]).strip()
    html = "\n".join([h for h in html_parts if h]).strip()
    return text, html, has_html, attachment_hint


def normalize_subject(subject: str) -> str:
    subject = (subject or "").strip().lower()
    subject = re.sub(r"^(re|fw|fwd)\s*:\s*", "", subject)
    subject = re.sub(r"\s+", " ", subject)
    return subject


def extract_body_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(" ", strip=True)


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_eml_file(path: str | Path) -> ParsedEmail:
    raw_bytes = Path(path).read_bytes()
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    text, html, has_html, attachment_hint = _extract_payloads(msg)
    html_text = extract_body_from_html(html)
    body = text if text else html_text
    raw_text = clean_text(body)
    subject = clean_text(msg.get("subject", ""))
    sender = clean_text(msg.get("from", ""))
    date = clean_text(msg.get("date", ""))
    message_id = clean_text(msg.get("message-id", "")) or hashlib.sha256(raw_bytes).hexdigest()
    return ParsedEmail(
        raw_text=body or "",
        clean_text=raw_text,
        subject=subject,
        sender=sender,
        date=date,
        message_id=message_id,
        has_html=has_html,
        html_text=html_text,
        attachment_hint=attachment_hint,
    )


def normalize_text_for_dedup(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "<url>", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "<email>", text)
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def body_hash(text: str) -> str:
    return hashlib.sha256(normalize_text_for_dedup(text).encode("utf-8", errors="ignore")).hexdigest()


def subject_hash(subject: str) -> str:
    return hashlib.sha256(normalize_subject(subject).encode("utf-8", errors="ignore")).hexdigest()


def thread_group_id(subject: str, sender: str, body: str, message_id: str) -> str:
    # Keep thread grouping stable across near-duplicate messages.
    # Message-ID is intentionally excluded to avoid making every mail a unique group.
    base = f"{normalize_subject(subject)}|{sender.lower().strip()}|{body_hash(body)}"
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()


def iter_tokens(text: str) -> Iterable[str]:
    return re.findall(r"[A-Za-z']+", text.lower())

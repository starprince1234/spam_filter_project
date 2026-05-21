from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class EmailRecord:
    email_id: str
    source_dataset: str
    source_label: str
    label: int
    raw_path: str
    clean_path: str
    subject: str
    sender: str
    date: str
    message_id: str
    thread_group: str
    sender_group: str
    subject_group: str
    body_group: str
    raw_text: str
    clean_text: str
    has_html: bool
    attachment_hint: bool
    is_corrupted: bool
    duplicate_group: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tarfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.dataset_schema import EmailRecord
from src.data.email_parser import body_hash, clean_text, normalize_subject, parse_eml_file, thread_group_id
from src.utils.io_utils import ensure_dir, load_yaml, sha256_text, write_csv, write_json


ENRON_SPAM_URL = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"
ENRON_SPAM_FALLBACK_URL = "https://github.com/MWiechmann/enron_spam_data/archive/refs/heads/master.zip"
SPAMASSASSIN_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_FILES = [
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_spam.tar.bz2",
    "20050311_spam_2.tar.bz2",
]
TREC_BASE = "https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/"
TREC_FILES = [
    "trec06p.tgz",
    "trec07p.tgz",
]
TREC_PREPROCESSED_CSV = "https://raw.githubusercontent.com/imdeepmind/Preprocessed-TREC-2007-Public-Corpus-Dataset/main/trec07p.csv"


def safe_download(url: str, out_path: Path) -> bool:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    suffixes = "".join(archive_path.suffixes)
    if suffixes.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
    else:
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=out_dir)


def find_email_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.stat().st_size > 0:
            # Keep flexible patterns because corpora often use hashed filenames with dots.
            if (
                p.suffix.lower() in {".eml", ".txt", ""}
                or p.name.isdigit()
                or re.match(r"^\d+\.", p.name) is not None
                or "." in p.name
            ):
                files.append(p)
    return files


def safe_sender_from_path(path: Path) -> str:
    s = str(path).replace("\\", "/").lower()
    for key in ["spam", "ham"]:
        if f"/{key}/" in s or s.endswith(f"/{key}") or f"_{key}" in s:
            return key
    return "unknown"


def infer_label_from_label_file(path: Path) -> tuple[int, str]:
    lower = str(path).lower()
    if "spam" in lower:
        return 1, "spam"
    if "ham" in lower:
        return 0, "ham"
    return -1, "unknown"


def parse_message_record(corpus_name: str, file_path: Path, label: int, source_label: str, raw_store: Path, clean_store: Path) -> EmailRecord | None:
    try:
        parsed = parse_eml_file(file_path)
        raw_body = parsed.raw_text or ""
        clean_body = clean_text(parsed.clean_text or raw_body)
        if len(clean_body) < 10:
            return None
        src_hash = sha256_text(f"{file_path.resolve()}::{parsed.message_id}::{clean_body[:200]}")
        raw_rel = raw_store / corpus_name / f"{src_hash}.txt"
        clean_rel = clean_store / corpus_name / f"{src_hash}.txt"
        ensure_dir(raw_rel.parent)
        ensure_dir(clean_rel.parent)
        raw_rel.write_text(raw_body, encoding="utf-8")
        clean_rel.write_text(clean_body, encoding="utf-8")
        sender = parsed.sender or safe_sender_from_path(file_path)
        subject = parsed.subject or ""
        return EmailRecord(
            email_id=src_hash,
            source_dataset=corpus_name,
            source_label=source_label,
            label=label,
            raw_path=str(raw_rel.as_posix()),
            clean_path=str(clean_rel.as_posix()),
            subject=subject,
            sender=sender,
            date=parsed.date,
            message_id=parsed.message_id,
            thread_group=thread_group_id(subject, sender, clean_body, parsed.message_id),
            sender_group=sha256_text(sender.lower().strip() or "unknown_sender"),
            subject_group=sha256_text(normalize_subject(subject)),
            body_group=body_hash(clean_body),
            raw_text=raw_body,
            clean_text=clean_body,
            has_html=parsed.has_html,
            attachment_hint=parsed.attachment_hint,
            is_corrupted=False,
            duplicate_group=body_hash(clean_body),
        )
    except Exception:
        return None


def parse_trec_index(index_file: Path) -> pd.DataFrame:
    rows = []
    for line in index_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        label_char, rel = parts
        if label_char not in {"0", "1"}:
            continue
        rows.append({"relative_path": rel, "label": int(label_char), "source_label": "spam" if label_char == "1" else "ham"})
    return pd.DataFrame(rows)


def parse_trec_corpus(corpus_name: str, corpus_root: Path, raw_store: Path, clean_store: Path) -> list[EmailRecord]:
    index_file = None
    for cand in corpus_root.rglob("index"):
        index_file = cand
        break
    if index_file is None:
        return []
    index_df = parse_trec_index(index_file)
    rows: list[EmailRecord] = []
    for _, r in tqdm(index_df.iterrows(), total=len(index_df), desc=f"Parsing {corpus_name}"):
        fp = corpus_root / r["relative_path"]
        if not fp.exists():
            continue
        rec = parse_message_record(corpus_name, fp, int(r["label"]), str(r["source_label"]), raw_store, clean_store)
        if rec is not None:
            rows.append(rec)
    return rows


def parse_trec_preprocessed(csv_path: Path, raw_store: Path, clean_store: Path) -> list[EmailRecord]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    rows: list[EmailRecord] = []
    text_col = None
    label_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl in {"text", "body", "message"}:
            text_col = c
        elif cl in {"label", "spam", "spam_ham", "class"}:
            label_col = c
    if text_col is None or label_col is None:
        return []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing TREC preprocessed"):
        label_val = str(row[label_col]).strip().lower()
        if label_val in {"1", "spam", "true", "yes"}:
            label = 1
            source_label = "spam"
        elif label_val in {"0", "ham", "false", "no"}:
            label = 0
            source_label = "ham"
        else:
            continue
        text = clean_text(str(row[text_col]))
        if len(text) < 10:
            continue
        pseudo = hashlib.sha256(f"{csv_path}:{idx}:{text[:200]}".encode("utf-8", errors="ignore")).hexdigest()
        raw_rel = raw_store / "trec_preprocessed" / f"{pseudo}.txt"
        clean_rel = clean_store / "trec_preprocessed" / f"{pseudo}.txt"
        ensure_dir(raw_rel.parent)
        ensure_dir(clean_rel.parent)
        raw_rel.write_text(text, encoding="utf-8")
        clean_rel.write_text(text, encoding="utf-8")
        rows.append(
            EmailRecord(
                email_id=pseudo,
                source_dataset="trec_preprocessed",
                source_label=source_label,
                label=label,
                raw_path=str(raw_rel.as_posix()),
                clean_path=str(clean_rel.as_posix()),
                subject="",
                sender="",
                date="",
                message_id=pseudo,
                thread_group=thread_group_id("", "", text, pseudo),
                sender_group=sha256_text("unknown_sender"),
                subject_group=sha256_text(""),
                body_group=body_hash(text),
                raw_text=text,
                clean_text=text,
                has_html=False,
                attachment_hint=False,
                is_corrupted=False,
                duplicate_group=body_hash(text),
            )
        )
    return rows


def parse_spamassassin_corpus(corpus_root: Path, raw_store: Path, clean_store: Path) -> list[EmailRecord]:
    rows: list[EmailRecord] = []
    for file_path in tqdm(find_email_files(corpus_root), desc="Parsing SpamAssassin"):
        label, source_label = infer_label_from_label_file(file_path)
        if label == -1:
            continue
        rec = parse_message_record("spamassassin", file_path, label, source_label, raw_store, clean_store)
        if rec is not None:
            rows.append(rec)
    return rows


def parse_enron_spam_dataset(enron_root: Path, raw_store: Path, clean_store: Path) -> list[EmailRecord]:
    rows: list[EmailRecord] = []
    # Support common public mirror layouts.
    candidate_files = list(enron_root.rglob("*.csv")) + list(enron_root.rglob("*.tsv"))
    for csv_path in candidate_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        text_col = None
        label_col = None
        subject_col = None
        sender_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl in {"text", "body", "message", "email", "content"}:
                text_col = c
            elif cl in {"spam", "label", "is_spam", "class"}:
                label_col = c
            elif cl in {"spam/ham", "spam_ham"}:
                label_col = c
            elif cl == "subject":
                subject_col = c
            elif cl in {"from", "sender", "author"}:
                sender_col = c
        if text_col is None or label_col is None:
            continue
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing Enron-Spam {csv_path.name}"):
            label_val = str(row[label_col]).strip().lower()
            if label_val in {"1", "spam", "true", "yes"}:
                label = 1
                source_label = "spam"
            elif label_val in {"0", "ham", "false", "no"}:
                label = 0
                source_label = "ham"
            else:
                continue
            text = clean_text(str(row[text_col]))
            if len(text) < 10:
                continue
            subject = clean_text(str(row[subject_col])) if subject_col else ""
            sender = clean_text(str(row[sender_col])) if sender_col else ""
            pseudo = hashlib.sha256(f"{csv_path}:{idx}:{text[:200]}".encode("utf-8", errors="ignore")).hexdigest()
            raw_rel = raw_store / "enron_spam" / f"{pseudo}.txt"
            clean_rel = clean_store / "enron_spam" / f"{pseudo}.txt"
            ensure_dir(raw_rel.parent)
            ensure_dir(clean_rel.parent)
            raw_rel.write_text(text, encoding="utf-8")
            clean_rel.write_text(text, encoding="utf-8")
            rows.append(
                EmailRecord(
                    email_id=pseudo,
                    source_dataset="enron_spam",
                    source_label=source_label,
                    label=label,
                    raw_path=str(raw_rel.as_posix()),
                    clean_path=str(clean_rel.as_posix()),
                    subject=subject,
                    sender=sender,
                    date="",
                    message_id=pseudo,
                    thread_group=thread_group_id(subject, sender, text, pseudo),
                    sender_group=sha256_text(sender.lower().strip() or "unknown_sender"),
                    subject_group=sha256_text(normalize_subject(subject)),
                    body_group=body_hash(text),
                    raw_text=text,
                    clean_text=text,
                    has_html=False,
                    attachment_hint=False,
                    is_corrupted=False,
                    duplicate_group=body_hash(text),
                )
            )
    return rows


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["source_dataset", "email_id"]).copy()
    df = df.drop_duplicates(subset=["duplicate_group"], keep="first")
    return df


def sample_main_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    ham_target = 4000
    spam_target = 1000
    ham_df = df[df["label"] == 0].copy()
    spam_df = df[df["label"] == 1].copy()
    if len(ham_df) < ham_target or len(spam_df) < spam_target:
        raise RuntimeError(f"Not enough samples. ham={len(ham_df)}, spam={len(spam_df)}")
    ham_df = ham_df.sample(n=ham_target, random_state=seed)
    spam_df = spam_df.sample(n=spam_target, random_state=seed)
    out = pd.concat([ham_df, spam_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["group_key"] = out["thread_group"].astype(str)
    return out


def split_by_group_and_label(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split at group level, then stratify labels inside each split."""
    df = df.copy().reset_index(drop=True)
    if "group_key" not in df.columns:
        df["group_key"] = df["thread_group"].astype(str)

    targets = {"train": 3500, "valid": 750, "test": 750}
    if len(df) >= 5000:
        if int((df["label"] == 1).sum()) != 1000 or int((df["label"] == 0).sum()) != 4000:
            raise ValueError("Expected fixed 4000/1000 label distribution before splitting.")

    group_check = df.groupby("group_key")["label"].nunique()
    if int((group_check > 1).sum()) > 0:
        raise ValueError("Mixed-label groups detected; cannot keep leakage-safe split.")

    group_df = df[["group_key", "label"]].drop_duplicates("group_key").copy()
    temp_fraction = min(0.4, (targets["valid"] + targets["test"]) / len(group_df))
    test_fraction = 0.5

    train_groups, temp_groups = train_test_split(
        group_df,
        test_size=temp_fraction,
        random_state=seed,
        stratify=group_df["label"],
    )
    valid_groups, test_groups = train_test_split(
        temp_groups,
        test_size=test_fraction,
        random_state=seed + 1,
        stratify=temp_groups["label"],
    )

    split_map: dict[str, str] = {}
    for split_name, split_groups in [("train", train_groups), ("valid", valid_groups), ("test", test_groups)]:
        for group_key in split_groups["group_key"].astype(str):
            split_map[group_key] = split_name

    df["split"] = df["group_key"].map(split_map)
    if df["split"].isna().any():
        raise RuntimeError("Split assignment failed for some records.")

    if len(df) == sum(targets.values()):
        counts = df["split"].value_counts().to_dict()
        for split_name, expected in targets.items():
            if int(counts.get(split_name, 0)) != expected:
                raise RuntimeError(f"Unexpected split size for {split_name}: {counts.get(split_name, 0)} != {expected}")
    return df


def leakage_rate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    train_groups = set(train_df["group_key"].astype(str))
    test_groups = set(test_df["group_key"].astype(str))
    if not test_groups:
        return 0.0
    return len(train_groups.intersection(test_groups)) / len(test_groups)


def summarize_distribution(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ["train", "valid", "test"]:
        sub = df[df["split"] == split]
        result[split] = {k: int(v) for k, v in sub.groupby("source_dataset").size().to_dict().items()}
    return result


def download_and_extract_all(root: Path, external: Path, logs_dir: Path) -> dict[str, str]:
    logs: dict[str, str] = {}
    enron_zip = external / "enron_spam_data.zip"
    if not safe_download(ENRON_SPAM_URL, enron_zip):
        safe_download(ENRON_SPAM_FALLBACK_URL, enron_zip)
    if enron_zip.exists():
        extract_archive(enron_zip, root / "data" / "raw" / "enron_spam")
        logs["enron_spam"] = "ok"
    else:
        logs["enron_spam"] = "failed"

    for fname in SPAMASSASSIN_FILES:
        url = SPAMASSASSIN_BASE + fname
        dst = external / fname
        ok = safe_download(url, dst)
        logs[f"spamassassin::{fname}"] = "ok" if ok else "failed"
        if ok:
            extract_archive(dst, root / "data" / "raw" / "spamassassin")

    for rel in TREC_FILES:
        url = TREC_BASE + rel
        dst = external / Path(rel).name
        ok = safe_download(url, dst)
        logs[f"trec::{Path(rel).name}"] = "ok" if ok else "failed"
        if ok:
            extract_archive(dst, root / "data" / "raw" / "trec")

    trec_csv = external / "trec07p.csv"
    if safe_download(TREC_PREPROCESSED_CSV, trec_csv):
        logs["trec_preprocessed_csv"] = "ok"
        extract_dir = root / "data" / "raw" / "trec_preprocessed"
        ensure_dir(extract_dir)
        # Keep CSV as-is; parser will read it directly.
    else:
        logs["trec_preprocessed_csv"] = "failed"

    write_json(logs, logs_dir / "download_log.json")
    return logs


def normalize_enron_spam_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "message id":
            rename_map[c] = "message_id"
        elif cl == "spam/ham":
            rename_map[c] = "spam_ham"
        elif cl == "message":
            rename_map[c] = "text"
        elif cl == "subject":
            rename_map[c] = "subject"
        elif cl == "date":
            rename_map[c] = "date"
    if rename_map:
        df = df.rename(columns=rename_map)
        df.to_csv(csv_path, index=False, encoding="utf-8")


def build(config_path: str) -> None:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parents[1]
    raw_data = root / cfg["paths"]["raw_data"]
    cleaned_data = root / cfg["paths"]["cleaned_data"]
    splits_dir = root / cfg["paths"]["splits"]
    external = root / cfg["paths"]["external"]
    tables_dir = root / cfg["paths"]["tables"]
    logs_dir = root / cfg["paths"]["logs"]
    for p in [raw_data, cleaned_data, splits_dir, external, tables_dir, logs_dir]:
        ensure_dir(p)

    download_log = download_and_extract_all(root, external, logs_dir)
    enron_csv = raw_data / "enron_spam" / "enron_spam_data.csv"
    if enron_csv.exists():
        normalize_enron_spam_csv(enron_csv)

    records: list[EmailRecord] = []
    enron_root = raw_data / "enron_spam"
    if enron_root.exists():
        records.extend(parse_enron_spam_dataset(enron_root, raw_data, cleaned_data))
    spam_root = raw_data / "spamassassin"
    if spam_root.exists():
        records.extend(parse_spamassassin_corpus(spam_root, raw_data, cleaned_data))
    trec_root = raw_data / "trec"
    if trec_root.exists():
        records.extend(parse_trec_corpus("trec", trec_root, raw_data, cleaned_data))
    trec_pre = external / "trec07p.csv"
    if trec_pre.exists():
        records.extend(parse_trec_preprocessed(trec_pre, raw_data, cleaned_data))

    if not records:
        raise RuntimeError("No records parsed. Check download sources and corpus layouts.")

    df = pd.DataFrame([r.to_dict() for r in records])
    write_csv(df, cleaned_data / "all_parsed_raw.csv")

    df = df[~df["is_corrupted"]].copy()
    write_csv(df, cleaned_data / "all_parsed_clean_no_corrupt.csv")

    df = deduplicate_records(df)
    write_csv(df, cleaned_data / "all_parsed_clean_dedup.csv")

    seed = int(cfg["project"]["seed"])
    np.random.seed(seed)
    final_df = sample_main_dataset(df, seed=seed)
    write_csv(final_df, cleaned_data / "main_dataset_5000.csv")

    split_df = split_by_group_and_label(final_df, seed=seed)
    split_df["group_key"] = split_df["group_key"].astype(str)
    write_csv(split_df, splits_dir / "main_dataset_split.csv")
    write_csv(split_df[split_df["split"] == "train"], splits_dir / "train.csv")
    write_csv(split_df[split_df["split"] == "valid"], splits_dir / "valid.csv")
    write_csv(split_df[split_df["split"] == "test"], splits_dir / "test.csv")

    train_df = split_df[split_df["split"] == "train"]
    test_df = split_df[split_df["split"] == "test"]
    leak = leakage_rate(train_df, test_df)

    stats = {
        "download_log": download_log,
        "parsed_total": int(len(records)),
        "after_corrupt_filter": int(len(pd.read_csv(cleaned_data / "all_parsed_clean_no_corrupt.csv"))),
        "after_dedup": int(len(pd.read_csv(cleaned_data / "all_parsed_clean_dedup.csv"))),
        "final_size": int(len(final_df)),
        "final_class_distribution": {
            "ham": int((final_df["label"] == 0).sum()),
            "spam": int((final_df["label"] == 1).sum()),
        },
        "split_sizes": {
            "train": int((split_df["split"] == "train").sum()),
            "valid": int((split_df["split"] == "valid").sum()),
            "test": int((split_df["split"] == "test").sum()),
        },
        "source_distribution_by_split": summarize_distribution(split_df),
        "train_test_leakage_rate": leak,
    }
    write_json(stats, tables_dir / "step2_data_stats.json")
    summary = pd.DataFrame(
        [
            ["parsed_total", stats["parsed_total"]],
            ["after_corrupt_filter", stats["after_corrupt_filter"]],
            ["after_dedup", stats["after_dedup"]],
            ["final_size", stats["final_size"]],
            ["ham", stats["final_class_distribution"]["ham"]],
            ["spam", stats["final_class_distribution"]["spam"]],
            ["train", stats["split_sizes"]["train"]],
            ["valid", stats["split_sizes"]["valid"]],
            ["test", stats["split_sizes"]["test"]],
            ["train_test_leakage_rate", stats["train_test_leakage_rate"]],
        ],
        columns=["metric", "value"],
    )
    write_csv(summary, tables_dir / "step2_data_summary_table.csv")
    write_csv(split_df.groupby(["split", "source_dataset"]).size().reset_index(name="count"), tables_dir / "step2_source_split_table.csv")

    with (logs_dir / "data_processing.log").open("w", encoding="utf-8") as f:
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        f.write(str(stats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()

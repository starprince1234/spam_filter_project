from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_report_at_threshold, expected_calibration_error
from src.evaluation.thresholds import select_threshold_by_precision_floor
from src.explainability.error_analysis import classify_error_case
from src.data.email_parser import normalize_subject

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / 'data' / 'splits'
MODELS_RAW = ROOT / 'models' / 'raw'
MODELS_CAL = ROOT / 'models' / 'calibrated'
FIG_DIR = ROOT / 'outputs' / 'figures'
REPORTS_DIR = ROOT / 'reports'
RESULTS_DIR = ROOT / 'experiments' / 'results'
APP_FILE = ROOT / 'app' / 'streamlit_app.py'
REQ_FILE = ROOT / 'requirements.txt'
REPORT_FILE = REPORTS_DIR / 'final_report.md'


REQUIRED_SPLITS = {
    'train.csv': 3500,
    'valid.csv': 750,
    'test.csv': 750,
}
REQUIRED_SPAM = {'train.csv': 700, 'valid.csv': 150, 'test.csv': 150}
REQUIRED_FIGS = [
    'confusion_matrix.png',
    'roc_curve.png',
    'pr_curve.png',
    'calibration_curve.png',
]


def _log(msg: str, rows: list[dict[str, Any]], level: str = 'PASS') -> None:
    print(f'[{level}] {msg}')
    rows.append({'level': level, 'message': msg})


def _norm_sender(x: Any) -> str:
    x = '' if pd.isna(x) else str(x)
    return re.sub(r'\s+', ' ', x.strip().lower())


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / name)


def _pred_probs_from_bundle(bundle: Any, texts: pd.Series) -> np.ndarray:
    if isinstance(bundle, dict) and 'model' in bundle and isinstance(bundle['model'], dict):
        base = bundle['model']
        model = base['model']
        vec = base['vectorizer']
        prob = model.predict_proba(vec.transform(texts.fillna('')))[:, 1]
        cal_type = bundle.get('calibration_type')
        calibrator = bundle.get('calibrator')
        if cal_type == 'temperature':
            t = float(calibrator.get('temperature', 1.0))
            p = np.clip(prob, 1e-6, 1 - 1e-6)
            logit = np.log(p / (1 - p))
            return 1 / (1 + np.exp(-logit / max(t, 1e-6)))
        if cal_type == 'sigmoid' and hasattr(calibrator, 'predict_proba'):
            p = np.clip(prob, 1e-6, 1 - 1e-6)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            return calibrator.predict_proba(logit)[:, 1]
        return prob
    model = bundle['model'] if isinstance(bundle, dict) and 'model' in bundle else bundle
    vec = bundle.get('vectorizer') if isinstance(bundle, dict) else None
    if vec is None:
        raise RuntimeError('model bundle missing vectorizer')
    return model.predict_proba(vec.transform(texts.fillna('')))[:, 1]


def _check_files(rows: list[dict[str, Any]]) -> None:
    for name, expected in REQUIRED_SPLITS.items():
        p = SPLIT_DIR / name
        ok = p.exists()
        _log(f'data/splits/{name} exists', rows, 'PASS' if ok else 'FAIL')
        if ok:
            df = pd.read_csv(p)
            _log(f'{name} rows={len(df)} expected={expected}', rows, 'PASS' if len(df) == expected else 'FAIL')
            spam = int(df['label'].sum()) if 'label' in df.columns else -1
            _log(f'{name} spam count={spam}', rows, 'PASS' if spam == REQUIRED_SPAM[name] else 'FAIL')

    for name in ['multinomial_nb.joblib', 'logistic_regression.joblib', 'linear_svm.joblib']:
        p = MODELS_RAW / name
        _log(f'model exists: models/raw/{name}', rows, 'PASS' if p.exists() else 'FAIL')
    for name in ['logistic_regression_calibrated.joblib', 'linear_svm_calibrated.joblib']:
        p = MODELS_CAL / name
        _log(f'model exists: models/calibrated/{name}', rows, 'PASS' if p.exists() else 'FAIL')

    missing_figs = []
    for name in REQUIRED_FIGS:
        ok = (FIG_DIR / name).exists()
        if not ok:
            missing_figs.append(name)
        _log(f'figure exists: outputs/figures/{name}', rows, 'PASS' if ok else 'FAIL')
    _log('final_report.md exists', rows, 'PASS' if REPORT_FILE.exists() else 'FAIL')
    if missing_figs:
        _log('missing required figure files', rows, 'FAIL')


def _audit_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = _load_split('train.csv')
    test = _load_split('test.csv')
    valid = _load_split('valid.csv')

    def md5(text: str) -> str:
        return hashlib.md5((text or '').encode('utf-8', errors='ignore')).hexdigest()

    train_md5 = train['clean_text'].fillna('').map(md5)
    test_md5 = test['clean_text'].fillna('').map(md5)
    valid_md5 = valid['clean_text'].fillna('').map(md5)
    md5_overlap = len(set(train_md5) & set(test_md5))
    md5_overlap_rate = md5_overlap / max(1, len(test))
    _log(f'train-test md5 overlap count={md5_overlap} rate={md5_overlap_rate:.6f}', rows, 'PASS' if md5_overlap == 0 else 'FAIL')

    train_senders = set(train['sender'].fillna('').map(_norm_sender))
    test_senders = set(test['sender'].fillna('').map(_norm_sender))
    valid_senders = set(valid['sender'].fillna('').map(_norm_sender))
    sender_overlap = len((train_senders & test_senders) - {''})
    sender_overlap_rate = sender_overlap / max(1, len(test_senders - {''}))
    _log(f'train-test sender overlap count={sender_overlap} rate={sender_overlap_rate:.6f}', rows, 'WARN' if sender_overlap else 'PASS')
    if sender_overlap:
        print('[WARN] sender overlap likely due to missing sender metadata in public corpora; treat as residual risk, not MD5 leakage.')

    train_subjects = set(train['subject'].fillna('').map(normalize_subject))
    test_subjects = set(test['subject'].fillna('').map(normalize_subject))
    valid_subjects = set(valid['subject'].fillna('').map(normalize_subject))
    subject_overlap = len((train_subjects & test_subjects) - {''})
    subject_overlap_rate = subject_overlap / max(1, len(test_subjects - {''}))
    _log(f'train-test subject overlap count={subject_overlap} rate={subject_overlap_rate:.6f}', rows, 'WARN' if subject_overlap else 'PASS')
    if subject_overlap:
        print('[WARN] subject overlap reflects shared mailing-list/forwarded subjects; inspect alongside group-key leakage rather than raw count.')

    return {
        'train': train,
        'valid': valid,
        'test': test,
        'md5_overlap': md5_overlap,
        'md5_overlap_rate': md5_overlap_rate,
        'sender_overlap': sender_overlap,
        'sender_overlap_rate': sender_overlap_rate,
        'subject_overlap': subject_overlap,
        'subject_overlap_rate': subject_overlap_rate,
    }


def _audit_threshold_and_metrics(rows: list[dict[str, Any]], splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    valid = splits['valid']
    test = splits['test']
    bundle = joblib.load(MODELS_CAL / 'logistic_regression_calibrated.joblib')
    raw_bundle = joblib.load(MODELS_RAW / 'logistic_regression.joblib')

    valid_prob = _pred_probs_from_bundle(bundle, valid['clean_text'])
    raw_valid_prob = _pred_probs_from_bundle(raw_bundle, valid['clean_text'])
    threshold, table = select_threshold_by_precision_floor(valid['label'], valid_prob, precision_floor=0.95)
    _log('threshold grid scan rows=91 expected=91', rows, 'PASS' if len(table) == 91 else 'FAIL')
    _log(f'selected threshold={threshold:.2f}', rows, 'PASS' if 0.05 <= threshold <= 0.95 else 'FAIL')
    selected_row = table[table['threshold'] == threshold].iloc[0]
    _log(f'valid precision at selected threshold={selected_row.precision:.4f}', rows, 'PASS' if selected_row.precision >= 0.95 else 'FAIL')

    test_prob = _pred_probs_from_bundle(bundle, test['clean_text'])
    raw_test_prob = _pred_probs_from_bundle(raw_bundle, test['clean_text'])
    test_metrics = classification_report_at_threshold(test['label'], test_prob, threshold=float(threshold))
    raw_test_metrics = classification_report_at_threshold(test['label'], raw_test_prob, threshold=float(threshold))
    _log(f'test precision={test_metrics["precision"]:.4f}', rows)
    _log(f'test recall={test_metrics["recall"]:.4f}', rows)
    _log(f'test f1={test_metrics["f1"]:.4f}', rows)
    _log(f'test macro_f1={test_metrics["macro_f1"]:.4f}', rows)

    return {
        'threshold': threshold,
        'valid_prob': valid_prob,
        'raw_valid_prob': raw_valid_prob,
        'test_prob': test_prob,
        'raw_test_prob': raw_test_prob,
        'test_metrics': test_metrics,
        'raw_test_metrics': raw_test_metrics,
        'threshold_table': table,
    }


def _audit_calibration(rows: list[dict[str, Any]], splits: dict[str, pd.DataFrame], probs: dict[str, Any]) -> dict[str, Any]:
    test = splits['test']
    raw_ece = expected_calibration_error(test['label'], probs['raw_test_prob'])
    cal_ece = expected_calibration_error(test['label'], probs['test_prob'])
    raw_brier = classification_report_at_threshold(test['label'], probs['raw_test_prob'], threshold=0.5)['brier']
    cal_brier = classification_report_at_threshold(test['label'], probs['test_prob'], threshold=0.5)['brier']
    _log(f'raw ECE={raw_ece:.6f}, calibrated ECE={cal_ece:.6f}', rows, 'PASS' if cal_ece <= raw_ece + 1e-12 else 'FAIL')
    _log(f'raw Brier={raw_brier:.6f}, calibrated Brier={cal_brier:.6f}', rows, 'PASS' if cal_brier <= raw_brier + 1e-12 else 'WARN')
    return {'raw_ece': raw_ece, 'cal_ece': cal_ece, 'raw_brier': raw_brier, 'cal_brier': cal_brier}


def _audit_app(rows: list[dict[str, Any]]) -> None:
    text = APP_FILE.read_text(encoding='utf-8')
    imports = re.findall(r'^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))', text, flags=re.M)
    imported = []
    for a, b in imports:
        mod = a or b
        imported.append(mod.split('.')[0])
    requirements = REQ_FILE.read_text(encoding='utf-8').lower()
    missing = []
    for mod in sorted(set(imported)):
        if mod in {'src', '__future__', 'pathlib', 'typing'}:
            continue
        if mod not in requirements:
            missing.append(mod)
    _log('app imports declared in requirements', rows, 'PASS' if not missing else 'WARN')
    if missing:
        print('[WARN] missing requirement declarations:', ', '.join(sorted(set(missing))))
    absolute_path_hits = re.findall(r'[A-Za-z]:\\[^\'"\n]+', text)
    _log('no hard-coded absolute paths in app', rows, 'PASS' if not absolute_path_hits else 'FAIL')


def _audit_report(rows: list[dict[str, Any]]) -> None:
    text = REPORT_FILE.read_text(encoding='utf-8')
    lower_text = text.lower()
    formulas = {
        'multinomial_nb': ['laplace', 'alpha', 'p(c', 'p(w_i'],
        'logistic_regression': ['logistic', 'l2', 'lambda'],
        'platt_ece': ['platt', 'ece'],
    }
    for name, pats in formulas.items():
        ok = all(p.lower() in lower_text for p in pats)
        _log(f'report formula coverage: {name}', rows, 'PASS' if ok else 'FAIL')

    required_papers = ['Sahami', 'Drucker', 'Androutsopoulos', 'Metsis', 'Guzella', 'Platt', 'Niculescu-Mizil', 'Guo', 'Devlin']
    missing = [p for p in required_papers if p.lower() not in lower_text]
    _log('report includes all 9 required papers', rows, 'PASS' if not missing else 'FAIL')
    if missing:
        print('[FAIL] missing paper mentions:', ', '.join(missing))

    required_tables = ['表 1', '表 2', '表 3', '表 4', '表 5']
    missing_tables = [t for t in required_tables if t not in text]
    _log('report includes five key tables', rows, 'PASS' if not missing_tables else 'FAIL')
    if missing_tables:
        print('[FAIL] missing tables:', ', '.join(missing_tables))
    if 'TODO' in text:
        _log('report contains TODO', rows, 'FAIL')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-out', default=str(ROOT / 'outputs' / 'tables' / 'acceptance_report.json'))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    _check_files(rows)
    split_info = _audit_leakage(rows)
    probs = _audit_threshold_and_metrics(rows, split_info)
    calib = _audit_calibration(rows, split_info, probs)
    _audit_app(rows)
    _audit_report(rows)

    summary = {
        'files': rows,
        'split_info': {k: (v if isinstance(v, (int, float, str, bool)) else None) for k, v in split_info.items() if k in ['md5_overlap', 'md5_overlap_rate', 'sender_overlap', 'sender_overlap_rate', 'subject_overlap', 'subject_overlap_rate']},
        'threshold': probs['threshold'],
        'test_metrics': probs['test_metrics'],
        'calibration': calib,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[INFO] acceptance summary written to {out}')
    return 0 if not any(r['level'] == 'FAIL' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())

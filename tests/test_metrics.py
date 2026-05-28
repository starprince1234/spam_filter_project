from src.evaluation.metrics import classification_report_at_threshold, expected_calibration_error


def test_expected_calibration_error_perfect():
    assert expected_calibration_error([0, 1], [0.0, 1.0], n_bins=2) == 0.0


def test_classification_report_keys():
    report = classification_report_at_threshold([0, 1], [0.1, 0.9], threshold=0.5)
    for key in ["precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc", "brier", "ece", "confusion_matrix"]:
        assert key in report

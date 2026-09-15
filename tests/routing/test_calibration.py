import pytest

from juniper_encoder.calibration import fit_temperature
from juniper_encoder.evaluation import brier_score, expected_calibration_error


def test_temperature_is_finite_and_positive():
    result = fit_temperature([[3.0, 0.0, -1.0], [0.0, 3.0, -1.0]], [0, 1])
    assert result["status"] == "CALIBRATED"
    assert 0.05 <= result["temperature"] <= 20.0
    assert result["post_nll"] <= result["pre_nll"]
    assert len(result["post_reliability_bins"]) == 15


def test_calibration_metrics_consume_probabilities_not_logits():
    probabilities = [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]]
    assert brier_score(probabilities, [0, 1]) == pytest.approx(0.08)
    result = expected_calibration_error(probabilities, [0, 1], bins=10)
    assert result["support"] == 2
    assert result["ece"] == pytest.approx(0.2)

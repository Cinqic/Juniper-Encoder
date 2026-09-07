from juniper_encoder.calibration import fit_temperature


def test_temperature_is_finite_and_positive():
    result = fit_temperature([[3.0, 0.0, -1.0], [0.0, 3.0, -1.0]], [0, 1])
    assert result["status"] == "CALIBRATED"
    assert 0.05 <= result["temperature"] <= 20.0
    assert result["post_nll"] <= result["pre_nll"]

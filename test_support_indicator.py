from src.modeling.evaluation import support_indicator


def test_support_indicator_uses_documented_thresholds() -> None:
    assert support_indicator(9.9)["priority"] == "High-priority outreach"
    assert support_indicator(10.0)["priority"] == "Targeted support"
    assert support_indicator(12.0)["priority"] == "Monitor and encourage"
    assert support_indicator(14.0)["priority"] == "Continue strengths-based support"

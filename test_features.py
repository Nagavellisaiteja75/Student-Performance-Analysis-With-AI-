import pandas as pd

from src.modeling.features import build_feature_spec


def test_early_intervention_features_exclude_prior_and_final_grades() -> None:
    data = pd.DataFrame(
        {
            "age": [16, 17],
            "school": ["GP", "MS"],
            "subject": ["math", "portuguese"],
            "G1": [12, 13],
            "G2": [13, 14],
            "G3": [14, 15],
        }
    )

    feature_spec = build_feature_spec(data)

    assert feature_spec.feature_columns == ["age", "school", "subject"]
    assert feature_spec.numeric_columns == ["age"]
    assert feature_spec.categorical_columns == ["school", "subject"]

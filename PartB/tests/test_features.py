
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path

import pandas as pd
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer


def test_weekday_imputer_transformer(sample_input_data):
    # Given
    transformer = WeekdayImputer(
        variable=config.model_config_.weekday_var,  # weekday_var
    )
    # print("*",sample_input_data[0].loc[13,'weekday'])
    assert np.isnan(sample_input_data[0].loc[7046,'weekday'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    print(subject.loc[7046,'weekday'])
    assert subject.loc[7046,'weekday'] == 'Wed'
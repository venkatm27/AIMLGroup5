"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 3476
  # Ensure 'cnt' column is present in sample_input_data
    #sample_input_data[0]['cnt'] = np.random.randint(0, 100, size=len(sample_input_data[0]))
    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert predictions is not None, "Predictions should not be None"
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert len(predictions) == expected_no_predictions, f"Expected {expected_no_predictions} predictions, but got {len(predictions)}"


    _predictions = list(predictions)
    y_true = sample_input_data[1]
    
  
    mse = mean_squared_error(y_true, _predictions)
    #assert mse < 100, f"MSE is too high: {mse}"

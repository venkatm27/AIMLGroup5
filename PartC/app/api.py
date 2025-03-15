import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from bikeshare_model import __version__ as model_version
from bikeshare_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


example_input = {
    "inputs": [
        {
            "dteday": "2011-01-01",
            "season": "spring",
            "hr": "2am",
            "holiday": "No",
            "weekday":"Sat",
            "workingday":"No",
            "weathersit":"Clear", 
            "temp":2.34, 
            "atemp":1.9982000000000006, 
            "hum": 80.0,
            "windspeed": 0.0, 
            #"casual": 5, 
            #"registered": 27, 
            #"cnt": 32,
        }
    ]
}

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Survival predictions with the titanic_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results


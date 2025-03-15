from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]



class DataInputSchema(BaseModel):
    dteday: Optional[str]
    season: Optional[str]
    hr: Optional[str]
    holiday: Optional[str]
    weekday: Optional[str]
    workingday: Optional[str]
    weathersit: Optional[str]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]
    #casual: Optional[int]
    #registered: Optional[int]
    #cnt: Optional[int]

"""
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
"""

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
import pickle
from typing import Annotated, List

import pandas as pd
import uvicorn
from fastapi import Body, Depends, FastAPI
from pydantic import BaseModel

app = FastAPI()


# Define the request and response models
class PredictRequest(BaseModel):
    trip_distance: float
    puborough: str
    puzone: str
    puservicezone: str
    doborough: str
    dozone: str
    doservicezone: str
    pickup_hour: str
    pickup_weekday: str


class PredictResponse(BaseModel):
    prediction: List[float]


# Load the model globally
_model = None


def get_model():
    global _model
    if _model is None:
        with open("catboost_model.pkl", "rb") as f:
            _model = pickle.load(f)
    return _model


@app.post("/predict")
def predict(data: List[PredictRequest], model=Depends(get_model)) -> PredictResponse:
    # Convert the list of PredictRequest objects to a DataFrame
    df = pd.DataFrame([{
        "trip_distance": d.trip_distance,
        "puborough": d.puborough,
        "puzone": d.puzone,
        "puservicezone": d.puservicezone,
        "doborough": d.doborough,
        "dozone": d.dozone,
        "doservicezone": d.doservicezone,
        "pickup_hour": d.pickup_hour,
        "pickup_weekday": d.pickup_weekday
    } for d in data])

    # Make predictions
    prediction = model.predict(df)

    # Return predictions as a list
    return PredictResponse(prediction=prediction.tolist())


def main():
    uvicorn.run(app)


if __name__ == '__main__':
    main()

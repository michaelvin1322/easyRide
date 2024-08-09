import pickle
import os
from typing import Annotated, List

import pandas as pd
import uvicorn
from fastapi import Body, Depends, FastAPI
from sqlalchemy import create_engine
from pydantic import BaseModel

app = FastAPI()


# Define the request and response models
class PredictRequest(BaseModel):
    request_datetime: str
    trip_distance: float
    PULocationID: int
    DOLocationID: int
    Airport: int


class PredictResponse(BaseModel):
    prediction: List[float]


# Load the model globally
_model = None


def get_db_engine():
    username = os.getenv('POSTGRES_USERNAME')
    password = os.getenv('POSTGRES_PASSWORD')
    host = os.getenv('POSTGRES_HOST')
    port = os.getenv('POSTGRES_PORT')
    database = os.getenv('POSTGRES_DATABASE')

    database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    engine = create_engine(database_url)
    return engine


def get_model():
    global _model
    if _model is None:
        with open("models/catboost_model.pkl", "rb") as f:
            _model = pickle.load(f)
    return _model


@app.post("/predict")
def predict(data: List[PredictRequest], model=Depends(get_model)) -> PredictResponse:
    # Convert the list of PredictRequest objects to a DataFrame
    df = pd.DataFrame([{
        "request_datetime": d.request_datetime,
        "trip_distance": d.trip_distance,
        "PULocationID": d.PULocationID,
        "DOLocationID": d.DOLocationID,
        "Airport": d.Airport
    } for d in data])

    engine = get_db_engine()

    # Fetch the additional information based on PULocationID and DOLocationID using pd.read_sql
    pu_location_ids = df['PULocationID'].unique().tolist()
    do_location_ids = df['DOLocationID'].unique().tolist()

    pu_query = f"""
    SELECT "LocationID", "Borough", "Zone", "service_zone" 
    FROM public.taxi_zone_lookup 
    WHERE "LocationID" IN ({','.join(map(str, pu_location_ids))})
    """

    do_query = f"""
    SELECT "LocationID", "Borough", "Zone", "service_zone" 
    FROM public.taxi_zone_lookup 
    WHERE "LocationID" IN ({','.join(map(str, do_location_ids))})
    """

    pu_info_df = pd.read_sql(pu_query, engine)
    do_info_df = pd.read_sql(do_query, engine)

    # Map the additional information back to the original DataFrame
    df = df.merge(pu_info_df, left_on='PULocationID', right_on='LocationID', suffixes=('', '_pu'))
    df = df.merge(do_info_df, left_on='DOLocationID', right_on='LocationID', suffixes=('', '_do'))

    df.rename(columns={
        "Borough": "puborough",
        "Zone": "puzone",
        "service_zone": "puservicezone",
        "Borough_do": "doborough",
        "Zone_do": "dozone",
        "service_zone_do": "doservicezone"
    }, inplace=True)

    # Apply the same preprocessing
    df['pickup_hour'] = pd.to_datetime(df['request_datetime']).dt.hour
    df['pickup_weekday'] = pd.to_datetime(df['request_datetime']).dt.weekday

    # Preserve categorical codes (ensure the codes match what was used during training)
    df['puborough'] = df['puborough'].astype('category').cat.codes
    df['puzone'] = df['puzone'].astype('category').cat.codes
    df['puservicezone'] = df['puservicezone'].astype('category').cat.codes
    df['doborough'] = df['doborough'].astype('category').cat.codes
    df['dozone'] = df['dozone'].astype('category').cat.codes
    df['doservicezone'] = df['doservicezone'].astype('category').cat.codes

    # Prepare final dataframe for prediction
    df = df[[
        'trip_distance', 'puborough', 'puzone', 'puservicezone',
        'doborough', 'dozone', 'doservicezone', 'pickup_hour',
        'pickup_weekday'
    ]]

    # Make predictions
    prediction = model.predict(df)

    5 + 6
    # Return predictions as a list
    return PredictResponse(prediction=prediction.tolist())


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == '__main__':
    main()

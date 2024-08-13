import pickle
import os
import json
from typing import Annotated, List
from logger import logger

import pandas as pd
import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException
from sqlalchemy import create_engine
from pydantic import BaseModel

from catboost import CatBoostRegressor

# Set display options to show all columns and full content
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

app = FastAPI()


# Define the request and response models
class PredictRequest(BaseModel):
    trip_id: str
    request_datetime: str
    trip_distance: float
    PULocationID: int
    DOLocationID: int
    Airport: int


class PredictResponse(BaseModel):
    prediction: float


# Load the model globally
_model = None
_categorical_mappings = None


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
        logger.info("Model loaded successfully")
    return _model


def get_mapping():
    global _categorical_mappings
    if _categorical_mappings is None:
        with open("models/categorical_mappings.json", "r") as f:
            _categorical_mappings = json.load(f)
        logger.info("Categorical mappings loaded successfully")
    return _categorical_mappings


@app.post("/predict")
def predict(data: PredictRequest, model=Depends(get_model), mappings=Depends(get_mapping), ) -> PredictResponse:
    # Convert the list of PredictRequest objects to a DataFrame
    df = pd.DataFrame({
        "trip_id": [data.trip_id],
        "request_datetime": [data.request_datetime],
        "trip_distance": [data.trip_distance],
        "PULocationID": [data.PULocationID],
        "DOLocationID": [data.DOLocationID],
        "Airport": [data.Airport],
    })
    logger.info("Converted input to DataFrame:\n%s", df)

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

    if pu_info_df.empty:
        logger.error("PULocationID %s not found TripID %s", data.PULocationID, data.trip_id)
        raise HTTPException(status_code=404, detail=f"PULocationID {data.PULocationID} not found")

    # Check if the DO location is not found
    if do_info_df.empty:
        logger.error("DOLocationID %s not found TripID %s", data.PULocationID, data.trip_id)
        raise HTTPException(status_code=404, detail=f"DOLocationID {data.DOLocationID} not found")

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

    categorical_columns = ['puborough', 'puzone', 'puservicezone', 'doborough', 'dozone', 'doservicezone']
    # Map categorical features using the loaded mappings
    for col in categorical_columns:
        df[col] = df[col].map(mappings[col])

    # Handle NaN values by assigning them the same code as during training (-1 or similar)
    df.fillna(-1, inplace=True)
    df[categorical_columns] = df[categorical_columns].astype(int).astype(str)

    # Prepare final dataframe for prediction
    df = df[[
        'trip_distance', 'puborough', 'puzone', 'puservicezone',
        'doborough', 'dozone', 'doservicezone', 'pickup_hour',
        'pickup_weekday'
    ]]
    logger.info("Final DataFrame for prediction:\n%s", df.to_string(index=False))

    # Make predictions
    prediction = model.predict(df)
    logger.info("Prediction result:\n%s", df.to_string(index=False))

    # Return predictions as a list
    return PredictResponse(prediction=prediction[0])


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == '__main__':
    main()

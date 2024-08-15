import pickle
import json
import re
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from db import engine
from logger import logger

# Set display options to show all columns and full content
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

app = FastAPI()

# Extend the dictionary to include datetime formats without the timezone and with a space between date and time
allowed_date_time_formats_re = {
    r"^\d{4}-\d{2}-\d{2} \d{2}/\d{2}/\d{2}\+\d{2}:\d{2}$": '%Y-%m-%d %H/%M/%S%z',  # Format: 2024-08-15 09/28/25+02:00
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2}$": '%Y-%m-%d %H:%M:%S%z',  # Format: 2024-08-06 14:30:00+02:00
    r"^\d{4}-\d{2}-\d{2}T\d{2}/\d{2}/\d{2}\+\d{4}$": '%Y-%m-%dT%H/%M/%S%z',        # Format: 2024-08-15T09/28/25+0200
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{4}$": '%Y-%m-%dT%H:%M:%S%z',        # Format: 2024-08-06T14:30:00+0200
    r"^\d{4}-\d{2}-\d{2} \d{2}/\d{2}/\d{2}$": '%Y-%m-%d %H/%M/%S',                 # Format: 2024-08-15 09/28/25
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$": '%Y-%m-%d %H:%M:%S',                 # Format: 2024-08-06 14:30:00
    r"^\d{4}-\d{2}-\d{2}T\d{2}/\d{2}/\d{2}$": '%Y-%m-%dT%H/%M/%S',                 # Format: 2024-08-15T09/28/25
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$": '%Y-%m-%dT%H:%M:%S'                  # Format: 2024-08-06T14:30:00
}


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
    if data.trip_distance < 0:
        raise HTTPException(status_code=400, detail="Trip distance cannot be negative")

    logger.info("Converted input to DataFrame:\n%s", df.to_string(index=False))

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

    parsed_datetime = None

    for pattern, dt_format in allowed_date_time_formats_re.items():
        if re.match(pattern, data.request_datetime):
            parsed_datetime = datetime.strptime(data.request_datetime, dt_format)
            break

    if parsed_datetime is None:
        raise HTTPException(status_code=400, detail="Invalid datetime format")

    # Apply the same preprocessing
    df['pickup_hour'] = parsed_datetime.hour
    df['pickup_weekday'] = parsed_datetime.weekday()

    categorical_columns = ['puborough', 'puzone', 'puservicezone', 'doborough', 'dozone', 'doservicezone']
    # Map categorical features using the loaded mappings
    for col in categorical_columns:
        df[col] = df[col].map(mappings[col])

    # Handle NaN values by assigning them the same code as during training (-1 or similar)
    df.fillna(-1, inplace=True)
    df[categorical_columns] = df[categorical_columns].astype(int).astype(str)

    # Prepare final dataframe for prediction
    df['trip_id'] = [data.trip_id]
    df = df[[
        'trip_id', 'trip_distance', 'puborough', 'puzone',
        'puservicezone', 'doborough', 'dozone', 'doservicezone',
        'pickup_hour', 'pickup_weekday',
    ]]
    logger.info("Final DataFrame for prediction:\n%s", df.to_string(index=False))

    # Make predictions
    prediction = model.predict(df)
    df['prediction'] = prediction
    df['trip_id'] = [data.trip_id]
    order = [
        "trip_id", "trip_distance", "puborough", "puzone",
        "puservicezone", "doborough", "dozone", "doservicezone",
        "pickup_hour", "pickup_weekday",  "prediction",
    ]
    df = df[order]
    logger.info("Prediction result:\n%s", df.to_string(index=False))

    # Return predictions as a list
    return PredictResponse(prediction=prediction[0])


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == '__main__':
    main()

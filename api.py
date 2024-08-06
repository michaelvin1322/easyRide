from typing import List, Annotated

import uvicorn
from fastapi import Body, Depends, FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

app = FastAPI()

# Define models for the /hello endpoint
class HelloRequest(BaseModel):
    name: str

class HelloResponse(BaseModel):
    text: str
    name: str

@app.post("/hello")
def hello(data: Annotated[HelloRequest, Body()]) -> HelloResponse:
    return HelloResponse(text=f"Hello, {data.name}", name=data.name)

# Define models and logic for the /predict endpoint
class PredictRequest(BaseModel):
    a: float
    b: float

class PredictResponse(BaseModel):
    prediction: List[float]

_model = None

def get_model():
    global _model
    if _model is None:
        with open("model.pkl", "rb") as f:
            _model = pickle.load(f)
    return _model

@app.post("/predict")
def predict(data: Annotated[List[PredictRequest], Body()], model=Depends(get_model)) -> PredictResponse:
    prediction = model.predict(pd.DataFrame(
        {"a": [o.a for o in data], "b": [o.b for o in data]}
    ))
    return PredictResponse(prediction=prediction.tolist())

# Define database credentials and logic for the /data endpoint
USERNAME = 'students.aeootalfoqhilupvmfac'
PASSWORD = 'students12345'
HOST = 'aws-0-eu-central-1.pooler.supabase.com'
PORT = '6543'
DATABASE = 'postgres'

database_url = f"postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
engine: Engine = create_engine(database_url)

@app.get("/data")
def get_data():
    query = """
    SELECT
      *
    FROM
      public.trips_2024_07_30
    ORDER BY
      tpep_dropoff_datetime DESC
    LIMIT
      10;
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()

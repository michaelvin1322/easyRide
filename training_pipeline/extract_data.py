import pandas as pd
from sqlalchemy import create_engine
import os


def extract_data(
        prefect: bool = False
):
    # Replace the following values with your actual database credentials
    username = os.environ['POSTGRES_USERNAME']
    password = os.environ['POSTGRES_PASSWORD']
    host = os.environ['POSTGRES_HOST']
    port = os.environ['POSTGRES_PORT']
    database = os.environ['POSTGRES_DATABASE']

    # Create the database URL for PostgreSQL
    database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    # Create the SQL Alchemy engine
    engine = create_engine(database_url)
    # SQL query to fetch data with joins and categorize borough, zone, and service zone
    query = """
    SELECT
      trips.*,
      pickup_zone."Borough" as PUBorough,
      pickup_zone."Zone" as PUZone,
      pickup_zone."service_zone" as PUServiceZone,
      dropoff_zone."Borough" as DOBorough,
      dropoff_zone."Zone" as DOZone,
      dropoff_zone."service_zone" as DOServiceZone
    FROM
      public.trips_2024_07_30 trips
    LEFT JOIN
      public.taxi_zone_lookup pickup_zone ON trips."PULocationID" = pickup_zone."LocationID"
    LEFT JOIN
      public.taxi_zone_lookup dropoff_zone ON trips."DOLocationID" = dropoff_zone."LocationID"
    ORDER BY
      tpep_dropoff_datetime DESC
    """
    # Replace with your actual database connection details
    df = pd.read_sql(query, engine)
    df.to_csv('data/dataset.csv', index=False)

    if prefect:
        return df


if __name__ == "__main__":
    extract_data()

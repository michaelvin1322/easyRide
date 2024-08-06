import pandas as pd
from sql_example import engine

# SQL query to fetch 10 samples
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
LIMIT 10
"""

if __name__ == "__main__":

    # Reading data from the database
    df = pd.read_sql(query, engine)

    # Creating the new feature 'trip_duration' (in minutes)
    df['trip_duration'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds() / 60.0

    # Adding hour and weekday of departure
    df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['pickup_weekday'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.weekday

    df['puborough'] = df['puborough'].astype('category').cat.codes
    df['puzone'] = df['puzone'].astype('category').cat.codes
    df['puservicezone'] = df['puservicezone'].astype('category').cat.codes
    df['doborough'] = df['doborough'].astype('category').cat.codes
    df['dozone'] = df['dozone'].astype('category').cat.codes
    df['doservicezone'] = df['doservicezone'].astype('category').cat.codes

    categorical_features = [
        'puborough', 'puzone', 'puservicezone',
        'doborough', 'dozone', 'doservicezone',
        'pickup_hour', 'pickup_weekday'
    ]


    for col in categorical_features:
        df[col] = df[col].astype(str)

    columns_to_keep = [
        'trip_distance', 'puborough', 'puzone', 'puservicezone',
        'doborough', 'dozone', 'doservicezone', 'pickup_hour',
        'pickup_weekday', 'pickup_hour', 'pickup_weekday',
        'trip_duration'
    ]
    df = df[columns_to_keep]

    # Save the DataFrame to a CSV file
    df.to_csv('sample_data.csv', index=False)

    print("Sample data has been saved to 'sample_data.csv'")

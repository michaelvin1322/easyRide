from sql_example import engine
import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

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

if __name__ == "__main__":
    # Reading data from the database
    df = pd.read_sql(query, engine)

    # Verify the column names
    print("DataFrame columns:")
    print(df.columns)

    # Creating the new feature 'trip_duration' (in minutes)
    df['trip_duration'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds() / 60.0

    # Adding hour and weekday of departure
    df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['pickup_weekday'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.weekday

    # Drop rows with missing values for simplicity
    df = df.dropna(subset=['trip_duration', 'trip_distance', 'PULocationID', 'DOLocationID'])

    # Encode categorical variables
    # df['PULocationID'] = df['PULocationID'].astype('category').cat.codes
    # df['DOLocationID'] = df['DOLocationID'].astype('category').cat.codes
    df['puborough'] = df['puborough'].astype('category').cat.codes
    df['puzone'] = df['puzone'].astype('category').cat.codes
    df['puservicezone'] = df['puservicezone'].astype('category').cat.codes
    df['doborough'] = df['doborough'].astype('category').cat.codes
    df['dozone'] = df['dozone'].astype('category').cat.codes
    df['doservicezone'] = df['doservicezone'].astype('category').cat.codes

    # Define features and target variable
    X = df[['trip_distance', 'puborough', 'puzone', 'puservicezone', 'doborough', 'dozone', 'doservicezone', 'pickup_hour', 'pickup_weekday']]
    y = df['trip_duration']

    categorical_features = ['puborough', 'puzone', 'puservicezone', 'doborough', 'dozone', 'doservicezone', 'pickup_hour', 'pickup_weekday']


    for col in categorical_features:
        df[col] = df[col].astype(str)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the linear regression model
    model = CatBoostRegressor(iterations=10000, learning_rate=0.01, depth=8, cat_features=categorical_features, verbose=100)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate and print accuracy and determination coefficient
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    # TODO: ADD MAPE

    # Save the model as a pickle file
    with open('catboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model has been trained and saved as 'catboost_model.pkl'")

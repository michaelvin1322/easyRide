import pandas as pd
import json
import numpy as np


def modify_data(
        prefect: bool = False,
        df: pd.DataFrame = pd.DataFrame(),
):
    if not prefect:
        df = pd.read_csv('data/dataset.csv')

    # Creating the new feature 'trip_duration' (in minutes)
    df['trip_duration'] = (
            (pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime']))
            .dt.total_seconds() / 60.0)

    # Adding hour and weekday of departure
    df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['pickup_weekday'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.weekday

    # Drop rows with missing values for simplicity
    df = df.dropna(subset=['trip_duration', 'trip_distance', 'PULocationID', 'DOLocationID'])

    # Encode categorical variables
    # df['PULocationID'] = df['PULocationID'].astype('category').cat.codes
    # df['DOLocationID'] = df['DOLocationID'].astype('category').cat.codes
    # df['puborough'] = df['puborough'].astype('category').cat.codes
    # df['puzone'] = df['puzone'].astype('category').cat.codes
    # df['puservicezone'] = df['puservicezone'].astype('category').cat.codes
    # df['doborough'] = df['doborough'].astype('category').cat.codes
    # df['dozone'] = df['dozone'].astype('category').cat.codes
    # df['doservicezone'] = df['doservicezone'].astype('category').cat.codes

    # Define features and target variable
    categorical_features = [
        'puborough', 'puzone', 'puservicezone', 'doborough',
        'dozone', 'doservicezone'
    ]

    # Create a dictionary to store the mappings
    categorical_mappings = {}

    for column in categorical_features:
        df[column] = df[column].astype('category')

        # Create a mapping that includes NaN as a category (if necessary)
        mapping = {category: int(code) for code, category in enumerate(df[column].cat.categories)}

        # If NaN exists, map it to a specific code, e.g., -1
        if pd.isnull(df[column]).any():
            mapping[np.nan] = -1

        categorical_mappings[column] = mapping

        # Apply the mapping, replacing NaN with the code
        df[column] = df[column].map(mapping)

    with open('models/categorical_mappings.json', 'w') as f:
        json.dump(categorical_mappings, f)

    for col in categorical_features:
        df[col] = df[col].astype(str)

    df = df[
        [
            'trip_distance', 'puborough', 'puzone', 'puservicezone',
            'doborough', 'dozone', 'doservicezone', 'pickup_hour',
            'pickup_weekday', 'trip_duration'
         ]
    ]

    df.to_csv('data/modified_dataset.csv', index=False)

    if prefect:
        return df


if __name__ == "__main__":
    modify_data()

import pandas as pd


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
    df['puborough'] = df['puborough'].astype('category').cat.codes
    df['puzone'] = df['puzone'].astype('category').cat.codes
    df['puservicezone'] = df['puservicezone'].astype('category').cat.codes
    df['doborough'] = df['doborough'].astype('category').cat.codes
    df['dozone'] = df['dozone'].astype('category').cat.codes
    df['doservicezone'] = df['doservicezone'].astype('category').cat.codes

    # Define features and target variable
    categorical_features = [
        'puborough', 'puzone', 'puservicezone', 'doborough',
        'dozone', 'doservicezone', 'pickup_hour', 'pickup_weekday',
    ]

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

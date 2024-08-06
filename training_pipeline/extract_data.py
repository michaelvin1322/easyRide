import pandas as pd
from sql_example import engine
from train import query


def extract_data():
    # Replace with your actual database connection details
    df = pd.read_sql(query, engine)
    df.to_csv('data/dataset.csv', index=False)


if __name__ == "__main__":
    extract_data()

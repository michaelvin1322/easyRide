from sqlalchemy import create_engine
import pandas as pd
import os

# Replace the following values with your actual database credentials
USERNAME = os.environ['POSTGRES_USERNAME']
PASSWORD = os.environ['POSTGRES_PASSWORD']
HOST = os.environ['POSTGRES_HOST']
PORT = os.environ['POSTGRES_PORT']
DATABASE = os.environ['POSTGRES_DATABASE']

# Create the database URL for PostgreSQL
database_url = f"postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
# Create the SQL Alchemy engine
engine = create_engine(database_url)

# SQL query to execute
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

if __name__ == "__main__":
    # Use pandas to load sql query result into a DataFrame
    df = pd.read_sql(query, engine)

    # Show the DataFrame
    print(df.columns)

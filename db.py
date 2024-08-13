from sqlalchemy import create_engine
import os

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

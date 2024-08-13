from prefect import task, flow
from sqlalchemy import create_engine
import pandas as pd
import boto3
import json
from datetime import datetime, timedelta
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


# Define your S3 bucket details and database connection
S3_BUCKET_NAME = "your-bucket-name"
DATABASE_URL = "your-database-url"


@task
def extract_new_data():
    # Task to extract new data from the database for the last hour
    engine = create_engine(DATABASE_URL)
    last_hour = datetime.now() - timedelta(hours=1)
    query = f"""
    SELECT *
    FROM your_table
    WHERE request_datetime >= '{last_hour.strftime('%Y-%m-%d %H:%M:%S')}'
    """
    df = pd.read_sql(query, engine)
    return df


@task
def fetch_logs_from_s3():
    # Task to fetch logs from the S3 bucket
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="logs/")
    log_data = []
    for obj in response.get('Contents', []):
        log_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
        log_data.append(json.loads(log_obj['Body'].read().decode('utf-8')))
    return pd.DataFrame(log_data)


@task
def calculate_metrics(new_data, logs_data):
    # Task to compare data and calculate metrics using evidently ai
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(new_data, logs_data)
    return dashboard


@flow(name="Data Monitoring Pipeline", log_prints=True)
def data_monitoring_flow():
    # Define the flow
    new_data = extract_new_data()
    logs_data = fetch_logs_from_s3()
    metrics_dashboard = calculate_metrics(new_data, logs_data)
    metrics_dashboard.save("data_monitoring_dashboard.html")
    return metrics_dashboard


if __name__ == "__main__":
    data_monitoring_flow()

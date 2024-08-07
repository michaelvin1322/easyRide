# prefect_flow.py
from prefect import task, flow
from training_pipeline.extract_data import extract_data
from training_pipeline.modify_data import modify_data
from training_pipeline.train_model import train_model

extract_data_task = task(extract_data)
modify_data_task = task(modify_data)
train_model_task = task(train_model)


@flow(
    name="Model training pipeline",
    log_prints=True,
)
def train_model_flow():
    data = extract_data_task(prefect=True)
    data_modified = modify_data_task(prefect=True, df=data)
    model = train_model_task(prefect=True, df=data_modified)
    return model


if __name__ == "__main__":
    train_model_flow()

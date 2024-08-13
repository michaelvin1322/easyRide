from prefect import flow

# Source for the code to deploy (here, a GitHub repo)
SOURCE_REPO = "https://github.com/michaelvin1322/easyRide.git"

if __name__ == "__main__":
    flow.from_source(
        source=SOURCE_REPO,
        entrypoint="prefect_train_pipeline.py:train_model_flow",  # Specific flow to run
    ).deploy(
        name="initial-deploy",
        work_pool_name="my-work-pool",  # Work pool target
        cron="0 1 * * *",  # Cron schedule (1am every day)
    )

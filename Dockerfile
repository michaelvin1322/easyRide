# Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Make sure the DVC remote storage is accessible
# RUN dvc remote modify myremote --local access_key_id ${S3_ACCESS_KEY_ID}
# RUN dvc remote modify myremote --local secret_access_key $S3_SECRET_ACCESS_KEY
#
# RUN --mount=type=secret,id=S3_ACCESS_KEY_ID \
#     S3_ACCESS_KEY_ID="$(cat /run/secrets/S3_ACCESS_KEY_ID)" dvc remote modify myremote --local access_key_id ${S3_ACCESS_KEY_ID}
# # Pull the DVC data
# RUN dvc pull

# Make port 80 available to the world outside this container
# EXPOSE 80

# Run the application
CMD ["python", "app.py"]
import os
import logging
from typing import Optional

import joblib
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
import google.auth
from google.cloud import aiplatform, storage

from models import RequestFeatures

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Manually set on the app deployment as of now
LOCAL_MODEL_DIR = "./model"
MODEL_ID = os.getenv("MODEL_ID")
MODEL_VERSION = os.getenv("MODEL_VERSION")

def _get_model_uri_from_version() -> Optional[str]:
    model_registry = aiplatform.Model.list(filter=f'display_name="{MODEL_ID}"')

    for model in model_registry:
        if model.version_id == MODEL_VERSION:
            return model.uri

def download_model():
    """Downloads the model from Vertex AI Model Registry to the current directory."""
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    credentials, project_id = google.auth.default()

    aiplatform.init(
        project=project_id,
        location="us-central1",
        credentials=credentials
    )

    logging.info(f"Fetching model {MODEL_ID} from Vertex AI...")
    model_uri = _get_model_uri_from_version()

    if not model_uri:
        raise ValueError("Model URI is not available")

    if not model_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// path but got {model_uri}")

    bucket_name = model_uri.split("/")[2]
    prefix = "/".join(model_uri.split("/")[3:])

    # Initialize GCS client
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    # List objects to find the .pkl file
    blobs = bucket.list_blobs(prefix=prefix)
    model_files = [blob for blob in blobs if blob.name.endswith('.pkl')]

    if not model_files:
        raise ValueError("No .pkl file found in model artifacts")

    for model in model_files:
        if model.name.endswith("model.pkl"):
            model.download_to_filename(f"{LOCAL_MODEL_DIR}/model.pkl")

    logging.info("Model download complete!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function executes on container startup, and it downloads the model.pkl locally to the container directory, so
    it is not bound to the GCS network call at prediction time.
    A point for enhancement would be to have the model in memory, but have to test it in future versions.
    """
    download_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "running", "model_downloaded": os.path.exists(LOCAL_MODEL_DIR)}

@app.post("/predict")
def predict(features: RequestFeatures) -> dict:
    instances_list = [instance.model_dump(by_alias=True) for instance in features.instances]

    req = pd.DataFrame(instances_list)

    model = joblib.load(f"{LOCAL_MODEL_DIR}/model.pkl")

    return {"predictions": model.predict(req).tolist()}
from kfp.v2.google.client import AIPlatformClient
from src.main import regression_pipeline

aiplatform_client = AIPlatformClient(
    project_id="my-gcp-project",
    region="us-central1"
)

pipeline_params = {
    "file_path": "gs://my-bucket/data/train.csv",
    "model_name": "my-regression-model",
    "gs_bucket": "my-bucket",
    "serving_container_uri": "gcr.io/cloud-aiplatform/prediction/xgboost2-cpu:latest"
}

aiplatform_client.create_run_from_pipeline_func(
    regression_pipeline,
    pipeline_root="gs://my-bucket/kfp-runs",
    parameter_values=pipeline_params
)

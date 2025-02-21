import google.auth
from google.cloud import aiplatform
credentials, project_id = google.auth.load_credentials_from_file("./level-agent-451514-c0-fc48725fe84a.json")

aiplatform.init(
    project=project_id,
    location="us-central1",
    credentials=credentials
)

pipeline_params = {
    "file_path": "gs://house-pricing-tryolabs/data/train.csv",
    "model_name": "my-first-model",
    "serving_container_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest",
    "pipeline_root": "gs://house-pricing-tryolabs/artifacts"
}

pipeline_job = aiplatform.PipelineJob(
    display_name="my-first-model",
    pipeline_root=pipeline_params["pipeline_root"],
    template_path="./regression_pipeline_v2.json",
    parameter_values=pipeline_params
)

pipeline_job.submit()
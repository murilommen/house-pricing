from typing import Dict

import google.auth
from google.cloud import aiplatform
from kfp.compiler import compiler

from src.pipeline import regression_pipeline

credentials, project_id = google.auth.default()

def compile_pipeline(file_name: str) -> None:
    compiler.Compiler().compile(
        pipeline_func=regression_pipeline,
        package_path=file_name
    )

    print(f"Compiled pipeline to: {file_name}")

def submit_pipeline(file_name: str, pipeline_parameters: Dict[str, str]) -> None:
    aiplatform.init(
        project=project_id,
        location="us-central1",
        credentials=credentials
    )

    pipeline_job = aiplatform.PipelineJob(
        display_name=pipeline_parameters["model_name"],
        pipeline_root=pipeline_parameters["pipeline_root"],
        template_path=f"./{file_name}",
        parameter_values=pipeline_parameters
    )

    pipeline_job.submit()

if __name__ == "__main__":
    pipeline_filename = 'regression_pipeline.json'
    pipeline_params = {
        "file_path": "gs://house-pricing-tryolabs/data/train.csv",
        "model_name": "my-first-model",
        "pipeline_root": "gs://house-pricing-tryolabs/artifacts"
    }

    compile_pipeline(pipeline_filename)
    submit_pipeline(pipeline_filename, pipeline_params)
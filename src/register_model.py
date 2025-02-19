from typing import Optional

from google.cloud import aiplatform
from google.cloud.aiplatform import Model

GS_ARTIFACT_URI = "gs://your-bucket-path"
SERVING_CONTAINER_URI = "gcr.io/cloud-aiplatform/prediction/xgboost2-cpu:latest"


def register_model_to_vertex(model_name: str, rmse: float) -> Optional[Model]:
    model_registry = aiplatform.Model.list(project=model_name)
    
    # No model found, so register the first version
    if not model_registry:  
        print(f"Registering the first version of {model_name}...")
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=GS_ARTIFACT_URI,
            serving_container_image_uri=SERVING_CONTAINER_URI,
        )
        model.deploy(machine_type="n1-standard-4")
        print(f"Model {model_name} registered as version 1.")
        return model

    else:
        existing_model = model_registry[0]
        existing_model_rmse = existing_model.metadata.get("rmse", None)
       
        if existing_model_rmse is None:
            # TODO this should be a warning and throw
            print(f"Existing model {model_name} doesn't have an RMSE value. Skipping registration.")
            return None

        print(f"Comparing RMSEs: Existing model RMSE = {existing_model_rmse}, New model RMSE = {rmse}")
       
        # Register the new model only if its RMSE is lower than the existing one
        if rmse < existing_model_rmse:
            print(f"New model has a lower RMSE. Registering a new version of {model_name}...")
            model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=GS_ARTIFACT_URI,
                serving_container_image_uri=SERVING_CONTAINER_URI,
            )
            model.deploy(machine_type="n1-standard-4")
            print(f"Model {model_name} registered as a new version.")
            return model
        else:
            print("New model does not have a lower RMSE. Skipping registration.")
            return None

from kfp import dsl
from kfp.dsl import Output, Artifact, Input, Model, Dataset


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "numpy", "scikit-learn", "gcsfs"])
def load_data(file_path: str,
                        x_train_output: Output[Dataset],
                        x_test_output: Output[Dataset],
                        y_train_output: Output[Dataset],
                        y_test_output: Output[Dataset]):
    from typing import Tuple
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split

    def get_features_and_target(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(file_path)
        df.drop(columns=["Id"], inplace=True)
        y = np.log1p(df["SalePrice"])
        X = df.drop(columns=["SalePrice"])
        return X, y

    X, y = get_features_and_target(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pd.DataFrame(X_train).to_csv(x_train_output.path + ".csv", index=False)
    pd.DataFrame(X_test).to_csv(x_test_output.path + ".csv", index=False)
    pd.Series(y_train).to_csv(y_train_output.path + ".csv", index=False)
    pd.Series(y_test).to_csv(y_test_output.path + ".csv", index=False)


@dsl.component(base_image="python:3.10", packages_to_install=["pandas", "numpy", "scikit-learn", "gcsfs"])
def train_model(x_train_input: Input[Dataset],
                y_train_input: Input[Dataset],
                model_output: Output[Model]):
    import joblib

    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    def preprocess_data(X: pd.DataFrame) -> ColumnTransformer:
        num_features = X.select_dtypes(include=["int64", "float64"]).columns
        cat_features = X.select_dtypes(include=["object"]).columns

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("power_transform", PowerTransformer(method="yeo-johnson"))
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ])
        return preprocessor

    X_train = pd.read_csv(x_train_input.path + ".csv")
    y_train = pd.read_csv(y_train_input.path + ".csv").squeeze("columns")

    preprocessor = preprocess_data(X_train)

    model = LinearRegression()
    # TODO work with xgboost for a better performant model
    # model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    with open(model_output.path + ".pkl", 'wb') as f:
        joblib.dump(pipeline, f)
    model_output.metadata['framework'] = 'sklearn'
    model_output.metadata['type'] = 'model'
    model_output.metadata["model_filename"] = "model.pkl"


@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-storage"])
def copy_artifacts_to_gcs(
        model_input: Input[Model],
        pipeline_root: str,
):
    from google.cloud import storage

    storage_client = storage.Client()

    model_filename = model_input.metadata["model_filename"]

    bucket_name = pipeline_root.split("/")[2]

    bucket = storage_client.bucket(bucket_name)

    model_blob = bucket.blob("artifacts/" + model_filename)
    model_blob.upload_from_filename(model_input.path + ".pkl")


@dsl.component(base_image="python:3.10", packages_to_install=["pandas", "numpy", "scikit-learn", "gcsfs", "google-cloud-aiplatform"])
def evaluate_and_register(x_test_input: Input[Dataset],
                          y_test_input: Input[Dataset],
                          model_input: Input[Model],
                          pipeline_root: str,
                          model_name: str):
    import joblib
    from typing import Optional
    from math import sqrt
    import logging
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from google.cloud import aiplatform
    from google.api_core.exceptions import PermissionDenied, NotFound

    def register_model_to_vertex(model_name: str, rmse: float, pipeline_root: str) -> Optional[aiplatform.Model]:
        try:
            model_registry = aiplatform.Model.list(filter=f'display_name="{model_name}"')
        except (PermissionDenied, NotFound) as e:
            logging.error(f"Error listing models: {e}")
            return None

        if not model_registry:
            print(f"Registering the first version of {model_name}...")
            try:
                model = aiplatform.Model.upload(
                    display_name=model_name,
                    artifact_uri=pipeline_root,
                    serving_container_image_uri="us-central1-docker.pkg.dev/level-agent-451514-c0/ml-housing-repo/housing-model",
                )
                print(f"Model {model_name} registered as version 1.")

                model.update(labels={"rmse": str(rmse)})
                return model
            except Exception as e:
                logging.error(f"Error registering model: {e}")
                raise

        else:
            # Find existing model (highest version)
            existing_model = None
            highest_version = -1
            for model in model_registry:
                try:
                    version_id = int(model.version_id)
                    if version_id > highest_version:
                        highest_version = version_id
                        existing_model = model
                except (ValueError, TypeError):
                    logging.warning(f"Invalid version ID for model: {model.resource_name}")
                    continue

            if not existing_model:
                logging.warning("No valid existing model found.")
                return None

            existing_rmse = float(existing_model.labels.get("rmse", float("inf")))

            if rmse < existing_rmse:
                logging.info(
                    f"Registering a new version of {model_name} with RMSE {rmse} (previous: {existing_rmse})...")
                model = aiplatform.Model.upload(
                    display_name=model_name,
                    artifact_uri=pipeline_root,
                    serving_container_image_uri="us-central1-docker.pkg.dev/level-agent-451514-c0/ml-housing-repo/housing-model",
                )
                logging.info(f"Model {model_name} registered as a new version.")

                model.update(labels={"rmse": str(rmse)})
                return model
            else:
                logging.info(
                    f"New model RMSE {rmse} is not lower than existing RMSE {existing_rmse}. Skipping registration.")
                return None

    X_test = pd.read_csv(x_test_input.path + ".csv")
    y_test = pd.read_csv(y_test_input.path + ".csv").squeeze("columns")

    logging.info(f"X_test columns: {X_test.columns}")

    with open(model_input.path + ".pkl", 'rb') as f:
        model = joblib.load(f)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"New RMSE: {rmse}")

    register_model_to_vertex(model_name, rmse, pipeline_root)


@dsl.pipeline(
    description="A pipeline for training and registering a regression model"
)
def regression_pipeline(file_path: str, model_name: str,
                        pipeline_root: str):  # Added pipeline_root
    load_data_task = load_data(file_path=file_path)
    train_model_task = train_model(
        x_train_input=load_data_task.outputs["x_train_output"],
        y_train_input=load_data_task.outputs["y_train_output"]
    )

    copy_artifacts_task = copy_artifacts_to_gcs(
        model_input=train_model_task.outputs["model_output"],
        pipeline_root=pipeline_root,
    )

    evaluate_and_register_task = evaluate_and_register(
        x_test_input=load_data_task.outputs["x_test_output"],
        y_test_input=load_data_task.outputs["y_test_output"],
        model_input=train_model_task.outputs["model_output"],
        pipeline_root=pipeline_root,
        model_name=model_name
    )
    evaluate_and_register_task.after(copy_artifacts_task)
from kfp import dsl
from kfp.dsl import Output, Artifact, Input, Model, Dataset


@dsl.component(packages_to_install=["pandas", "numpy", "scikit-learn", "gcsfs"])
def load_and_preprocess(file_path: str,
                        x_train_output: Output[Dataset],
                        x_test_output: Output[Dataset],
                        y_train_output: Output[Dataset],
                        y_test_output: Output[Dataset],
                        preprocessor_output: Output[Artifact]):
    import pickle
    from typing import Tuple
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split

    def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(file_path)
        df.drop(columns=["Id"], inplace=True)
        y = np.log1p(df["SalePrice"])
        X = df.drop(columns=["SalePrice"])
        return X, y

    # def preprocess_data(X: pd.DataFrame) -> ColumnTransformer:
    #     num_features = X.select_dtypes(include=["int64", "float64"]).columns
    #     cat_features = X.select_dtypes(include=["object"]).columns
    #
    #     num_pipeline = Pipeline([
    #         ("imputer", SimpleImputer(strategy="median")),
    #         ("scaler", StandardScaler()),
    #         ("power_transform", PowerTransformer(method="yeo-johnson"))
    #     ])
    #     cat_pipeline = Pipeline([
    #         ("imputer", SimpleImputer(strategy="most_frequent")),
    #         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    #     ])
    #     preprocessor = ColumnTransformer([
    #         ("num", num_pipeline, num_features),
    #         ("cat", cat_pipeline, cat_features)
    #     ])
    #     return preprocessor

    def preprocess_data(X: np.ndarray) -> ColumnTransformer:
        num_indices = [i for i, col in enumerate(X.T) if np.issubdtype(col.dtype, np.number)]
        cat_indices = [i for i, col in enumerate(X.T) if not np.issubdtype(col.dtype, np.number)]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("power_transform", PowerTransformer(method="yeo-johnson"))
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        return ColumnTransformer([
            ("num", num_pipeline, num_indices),
            ("cat", cat_pipeline, cat_indices)
        ])

    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = preprocess_data(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)

    pd.DataFrame(X_train_processed).to_csv(x_train_output.path + ".csv", index=False)

    # will not need to preprocess X_test, as it will be pickled to do that on the prediction time
    pd.DataFrame(X_test).to_csv(x_test_output.path + ".csv", index=False)
    pd.Series(y_train).to_csv(y_train_output.path + ".csv", index=False)
    pd.Series(y_test).to_csv(y_test_output.path + ".csv", index=False)
    with open(preprocessor_output.path + ".pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    preprocessor_output.metadata['type'] = 'preprocessor'


@dsl.component(packages_to_install=["pandas", "numpy", "scikit-learn", "gcsfs", "xgboost"])
def train_model(x_train_input: Input[Dataset],
                y_train_input: Input[Dataset],
                preprocessor_input: Input[Artifact],
                model_output: Output[Model]):
    import pickle

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    with open(preprocessor_input.path + ".pkl", 'rb') as f:
        preprocessor = pickle.load(f)

    X_train = pd.read_csv(x_train_input.path + ".csv").values
    y_train = pd.read_csv(y_train_input.path + ".csv").squeeze("columns").values
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    with open(model_output.path + ".pkl", 'wb') as f:
        pickle.dump(pipeline, f)
    model_output.metadata['framework'] = 'xgboost'
    model_output.metadata['type'] = 'model'
    model_output.metadata["model_filename"] = "model.pkl"


@dsl.component(packages_to_install=["google-cloud-storage"])
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


@dsl.component(packages_to_install=["pandas", "numpy", "scikit-learn", "gcsfs", "google-cloud-aiplatform", "xgboost"])
def evaluate_and_register(x_test_input: Input[Dataset],
                          y_test_input: Input[Dataset],
                          model_input: Input[Model],
                          pipeline_root: str,
                          model_name: str,
                          serving_container_uri: str):
    import pickle
    from typing import Optional
    from math import sqrt
    import logging
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from google.cloud import aiplatform
    from google.cloud.aiplatform import Model
    from google.api_core.exceptions import PermissionDenied, NotFound

    def register_model_to_vertex(model_name: str, rmse: float, pipeline_root: str,
                                 serving_container_uri: str) -> \
            Optional[Model]:
        try:
            # Use a filter to find models with the exact display name.
            model_registry = aiplatform.Model.list(filter=f'display_name="{model_name}"')
        except (PermissionDenied, NotFound) as e:
            logging.error(f"Error listing models: {e}")
            return None

        if not model_registry:
            print(f"Registering the first version of {model_name}...")
            try:
                model = aiplatform.Model.upload(
                    display_name=model_name,
                    serving_container_image_uri=serving_container_uri,
                    artifact_uri=pipeline_root
                )
                model.deploy(machine_type="n1-standard-4")
                print(f"Model {model_name} registered as version 1.")
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


            # TODO sub fixed threshold to existing model rmse
            if rmse < 100.5:
                logging.info(f"Registering a new version of {model_name}...")
                model = aiplatform.Model.upload(
                    display_name=model_name,
                    serving_container_image_uri=serving_container_uri,
                    artifact_uri=pipeline_root
                )
                model.deploy(machine_type="n1-standard-4")
                logging.info(f"Model {model_name} registered as a new version.")
                return model
            else:
                logging.info("New model does not have a lower RMSE. Skipping.")
                return None

    X_test = pd.read_csv(x_test_input.path + ".csv").values
    y_test = pd.read_csv(y_test_input.path + ".csv").squeeze("columns").values

    with open(model_input.path + ".pkl", 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"New RMSE: {rmse}")

    register_model_to_vertex(model_name, rmse, pipeline_root, serving_container_uri)


@dsl.pipeline(
    description="A pipeline for training and registering a regression model"
)
def regression_pipeline(file_path: str, model_name: str, serving_container_uri: str,
                        pipeline_root: str):  # Added pipeline_root
    load_and_preprocess_task = load_and_preprocess(file_path=file_path)
    train_model_task = train_model(
        x_train_input=load_and_preprocess_task.outputs["x_train_output"],
        y_train_input=load_and_preprocess_task.outputs["y_train_output"],
        preprocessor_input=load_and_preprocess_task.outputs["preprocessor_output"]
    )
    # Copy artifacts *before* evaluation/registration
    copy_artifacts_task = copy_artifacts_to_gcs(
        model_input=train_model_task.outputs["model_output"],
        pipeline_root=pipeline_root,  # Pass pipeline_root
    )

    evaluate_and_register_task = evaluate_and_register(
        x_test_input=load_and_preprocess_task.outputs["x_test_output"],
        y_test_input=load_and_preprocess_task.outputs["y_test_output"],
        model_input=train_model_task.outputs["model_output"],
        pipeline_root=pipeline_root,
        model_name=model_name,
        serving_container_uri=serving_container_uri
    )
    evaluate_and_register_task.after(copy_artifacts_task)
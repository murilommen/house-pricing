from kfp import dsl
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.feature_engineering import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.register_model import register_model_to_vertex


@dsl.component(packages_to_install=["pandas", "numpy"])
def load_data_component(file_path: str, data_output: Output[Artifact]):
    return load_data(file_path)

@dsl.component
def preprocess_data_component(df: pd.DataFrame):
    return preprocess_data(df)

@dsl.component
def train_model_component(X_train: pd.DataFrame, y_train: pd.DataFrame, preprocessor: ColumnTransformer):
    return train_model(X_train, y_train, preprocessor)

@dsl.component
def evaluate_model_component(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame):
    return evaluate_model(model_pipeline, X_test, y_test)

@dsl.component
def register_model_component(model_pipeline, model_name: str, gs_artifact_uri: str, serving_container_uri: str):
    return register_model_to_vertex(model_pipeline, model_name, gs_artifact_uri, serving_container_uri)

@dsl.pipeline(
    name="Regression Pipeline",
    description="A pipeline for training and registering a regression model"
)
def regression_pipeline(file_path: str, model_name: str, gs_artifact_uri: str, serving_container_uri: str):
    X, y = load_data_component(file_path)
    preprocessor = preprocess_data_component(X)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model_pipeline = train_model_component(X_train, y_train, preprocessor)
    rmse = evaluate_model_component(model_pipeline, X_test, y_test)
   
    register_model_component(model_name, rmse, gs_artifact_uri, serving_container_uri)

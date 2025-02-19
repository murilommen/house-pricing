from kfp import dsl
from kfp.components import create_component_from_func
from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.feature_engineering import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.register_model import register_model_to_vertex


@create_component_from_func
def load_data_component(file_path: str):
    return load_data(file_path)

@create_component_from_func
def preprocess_data_component(X):
    return preprocess_data(X)

@create_component_from_func
def train_model_component(X_train, y_train, preprocessor):
    return train_model(X_train, y_train, preprocessor)

@create_component_from_func
def evaluate_model_component(model_pipeline, X_test, y_test):
    return evaluate_model(model_pipeline, X_test, y_test)

@create_component_from_func
def register_model_component(model_pipeline, model_name: str):
    return register_model_to_vertex(model_pipeline, model_name)

@dsl.pipeline(
    name="Regression Pipeline",
    description="A pipeline for training and registering a regression model"
)
def regression_pipeline(file_path: str, model_name: str):
    X, y = load_data_component(file_path)
    preprocessor = preprocess_data_component(X)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model_pipeline = train_model_component(X_train, y_train, preprocessor)
    rmse = evaluate_model_component(model_pipeline, X_test, y_test)
   
    register_model_component(model_name, rmse)

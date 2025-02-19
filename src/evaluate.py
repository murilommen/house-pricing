from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(model_pipeline, X_test, y_test):
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def train_model(X_train, y_train, preprocessor):
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.9, 1.0]
    }
    
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", grid_search)
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline

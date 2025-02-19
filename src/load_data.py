from typing import Tuple

import pandas as pd


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path)
    df.drop(columns=["Id"], inplace=True)
    y = np.log1p(train_data["SalePrice"])
    X = train_data.drop(columns=["SalePrice"])
    return X, y
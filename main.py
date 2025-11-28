import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# ------------------------------

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1, errors='ignore')


class HousePriceModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.pipeline = Pipeline([
            ("drop_columns", ColumnDropper(columns_to_drop=["id", "date", "zipcode"])),
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
        ])

    # train
    def fit(self, df: pd.DataFrame):
        X = df.drop("price", axis=1)
        y = df["price"]
        self.pipeline.fit(X, y)

    # new data
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop("price", axis=1, errors='ignore') if "price" in df.columns else df
        return self.pipeline.predict(X)

    # metrics
    def evaluate(self, df: pd.DataFrame) -> dict:
        X = df.drop("price", axis=1)
        y = df["price"]
        preds = self.pipeline.predict(X)
        return {
            "MAE": mean_absolute_error(y, preds),
            "MSE": mean_squared_error(y, preds),
            "RMSE": np.sqrt(mean_squared_error(y, preds)),
            "R2": r2_score(y, preds),
        }

    # cross-validation
    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> dict:
        X = df.drop("price", axis=1)
        y = df["price"]
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        mse_scorer = make_scorer(mean_squared_error)
        r2_scorer = make_scorer(r2_score)
        mse_scores = cross_val_score(self.pipeline, X, y, cv=kf, scoring=mse_scorer)
        r2_scores = cross_val_score(self.pipeline, X, y, cv=kf, scoring=r2_scorer)
        return {
            "MSE_per_fold": mse_scores,
            "Mean_MSE": mse_scores.mean(),
            "R2_per_fold": r2_scores,
            "Mean_R2": r2_scores.mean()
        }

    def plot_predictions(self, df: pd.DataFrame):
        X = df.drop("price", axis=1)
        y = df["price"]
        preds = self.pipeline.predict(X)
        plt.scatter(y, preds)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted")
        plt.show()


df = pd.read_csv("libraries/kc_house_data.csv")

house_model = HousePriceModel()

# train
house_model.fit(df)

# predict
metrics = house_model.evaluate(df)
print("Evaluation metrics:", metrics)

# cross-validation
cv_metrics = house_model.cross_validate(df, cv=5)
print("Cross-validation results:", cv_metrics)

# visual
house_model.plot_predictions(df)

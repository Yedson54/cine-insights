"""Module to train the model and evaluate the model performance."""

# pylint: disable=locally-disabled, multiple-statements, fixme, invalid-name, too-many-arguments, too-many-instance-attributes

import warnings
from time import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

import src.features.preprocess_train_data as Preprocess_trainData
import src.features.preprocessing_predicted_data as Preprocessing_PredictedData

parameters_: Dict[str, List[Optional[float]]] = {
    "objective": ["reg:squarederror"],
    "learning_rate": [0.03, 0.05, 0.07],
    "max_depth": [5, 6, 7],
    "min_child_weight": [3, 4],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7],
    "n_estimators": [500, 1000, 2800],
}
WITH_DURATION: bool = False


class XGBRegressorWrapper:
    """XGBRegressorWrapper."""

    def __init__(self, Raw_data: pd.DataFrame, with_duration: bool = WITH_DURATION):
        """
        Initialize the XGBRegressorWrapper.

        Args:
            Raw_data: The raw data for training the model.
            with_duration: A flag indicating whether to include duration in the preprocessing.

        Returns:
            None
        """
        (X_train, y_train), (X_test, y_test), (x_val, y_val), ct = self._get_clean_df(
            Raw_data, with_duration
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params_: Dict[str, Optional[float]] = {}
        self.model: Optional[XGBRegressor] = None
        self.ct_data_transformer = ct
        self.train_time: Optional[float] = None
        self.train_rmsle: Optional[float] = None
        self.test_rmsle: Optional[float] = None
        self.test_mape: Optional[float] = None
        self.param_Combination: List[Dict[str, Optional[float]]] = list(
            ParameterGrid(parameters_)
        )
        warnings.filterwarnings("ignore")

    def _get_clean_df(self, raw_df: pd.DataFrame, with_duration: bool) -> Tuple:
        """
        Preprocess the raw data.

        Args:
            raw_df: The raw data for preprocessing.
            with_duration: A flag indicating whether to include duration in the preprocessing.

        Returns:
            Tuple containing the preprocessed data and transformers.
        """
        return Preprocess_trainData.preprocessing_data(
            raw_df, with_duration=with_duration
        )

    def _train_model(self) -> None:
        """
        Train the XGBoost model.

        Returns:
            None
        """
        start = time()
        xgb = XGBRegressor(**self.best_params_) if self.best_params_ else XGBRegressor()
        xgb.fit(self.X_train, self.y_train)
        self.model = xgb
        self.train_time = np.round(time() - start, 4)

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """
        Evaluate the model performance.

        Args:
            X: The input features for evaluation.
            y: The target variable for evaluation.

        Returns:
            Tuple containing the RMSLE and MAPE scores.
        """
        y_pred = self.model.predict(X)
        rmsle_score = rmsle(y, y_pred)
        mape_score = mean_absolute_percentage_error(y, y_pred)
        return rmsle_score, mape_score

    def train_model(self, params: Dict[str, Optional[float]]) -> None:
        """
        Train the XGBoost model with the given parameters.

        Args:
            params: The parameters for training the model.

        Returns:
            None
        """
        xgb = XGBRegressor(**params)
        xgb.fit(self.X_train, self.y_train)
        self.model = xgb

    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the model performance on training and test data.

        Returns:
            Dictionary containing the evaluation metrics.
        """
        train_rmsle, _ = self._evaluate_model(self.X_train, self.y_train)
        test_rmsle, test_mape = self._evaluate_model(self.X_test, self.y_test)
        return {
            "Training RMSLE": train_rmsle,
            "Test RMSLE": test_rmsle,
            "Test MAPE": test_mape,
        }

    def display_results(self) -> pd.DataFrame:
        """
        Display the results of the model training and evaluation.

        Returns:
            DataFrame containing the results.
        """
        data = {
            "Best parameters": [self.best_params_],
            "Training RMSLE": [self.train_rmsle],
            "Test RMSLE": [self.test_rmsle],
            "Test MAPE": [self.test_mape],
            "Training time": [self.train_time],
        }
        return pd.DataFrame(data)

    def save_best_model(self, filename: str) -> None:
        """
        Save the best model to a file.

        Args:
            filename: The name of the file to save the model.

        Returns:
            None
        """
        joblib.dump(self.model, filename)

    def predict_(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: The input features for prediction.

        Returns:
            Array of predictions.
        """
        X_transform = Preprocessing_PredictedData.preprocessing_pipeline(
            X, self.ct_data_transformer, with_duration=WITH_DURATION
        )
        return self.model.predict(X_transform)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: The input features for prediction.

        Returns:
            Array of predictions.
        """
        return self.predict_(X)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true: The true values.
        y_pred: The predicted values.

    Returns:
        The RMSLE score.
    """
    return np.sqrt(((np.log1p(y_true) - np.log1p(y_pred)) ** 2).mean())


def mean_absolute_percentage_error(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Args:
        y_test: The true values.
        y_pred: The predicted values.

    Returns:
        The MAPE score.
    """
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


scorer = make_scorer(rmsle)

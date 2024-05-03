import sys
sys.path.append('/home/onyxia/work/cine-insights/src/features')
from sklearn.preprocessing import MinMaxScaler
from time import time
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import warnings
from sklearn.metrics import make_scorer

import Import_data
from  Import_data import *

import Preprocess_trainData
from  Preprocess_trainData import *

import Preprocessing_PredictedData
from Preprocessing_PredictedData import *

from sklearn.model_selection import ParameterGrid

parameters_ = {
            'objective': ['reg:squarederror'],
            'learning_rate': [0.03, 0.05, 0.07],
            'max_depth': [5, 6, 7],
            'min_child_weight': [3, 4],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7],
            'n_estimators': [500, 1000, 2800]
        }


with_duration=False

def get_clean_df(raw_df,with_duration):
    (x_train, y_train), (x_test, y_test), (x_val, y_val), ct = Preprocess_trainData.preprocessing_data(raw_df,with_duration=with_duration)
    return (x_train, y_train), (x_test, y_test), (x_val, y_val), ct


# define custom scorer
def rmsle(y_true, y_pred):
    rmsle = np.sqrt(((np.log1p(y_true)-np.log1p(y_pred))**2).mean())
    #rmsle = np.sqrt((y_true-y_pred)**2).mean()
    return rmsle


scorer = make_scorer(rmsle)


def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


class XGBRegressorWrapper:
    def __init__(self, Raw_data, with_duration):
        (X_train, y_train), (X_test, y_test), (x_val, y_val), ct = get_clean_df(Raw_data, with_duration)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params_ = None
        self.model = None
        self.ct_data_transformer = ct  # un attribut important lors de la prédiction de nouvelles données
        self.train_time = None
        self.train_rmsle = None
        self.test_rmsle = None
        self.test_mape = None
        self.param_Combination = list(ParameterGrid(parameters_))

        # Ignore all warnings
        warnings.filterwarnings('ignore')

    def _train_model(self):
        start = time()
        xgb = XGBRegressor(**self.best_params_) if self.best_params_ else XGBRegressor()
        xgb.fit(self.X_train, self.y_train)
        self.model = xgb
        self.train_time = np.round(time() - start, 4)

    def _evaluate_model(self, X, y):
        y_pred = self.model.predict(X)
        rmsle_score = rmsle(y, y_pred)
        mape_score = mean_absolute_percentage_error(y, y_pred)
        return rmsle_score, mape_score

    def train_model(self, params):
        xgb = XGBRegressor(**params)
        xgb.fit(self.X_train, self.y_train)
        self.model = xgb

    def evaluate_model(self):
        train_rmsle, _ = self._evaluate_model(self.X_train, self.y_train)
        test_rmsle, test_mape = self._evaluate_model(self.X_test, self.y_test)
        return {
            "Training RMSLE": train_rmsle,
            "Test RMSLE": test_rmsle,
            "Test MAPE": test_mape
        }
    def display_results(self):
        data = {
            "Best parameters": [self.best_params_],
            "Training RMSLE": [self.train_rmsle],
            "Test RMSLE": [self.test_rmsle],
            "Test MAPE": [self.test_mape],
            "Training time": [self.train_time]
        }
        return pd.DataFrame(data)

    def save_best_model(self, filename):
        import joblib
        joblib.dump(self.model, filename)

    def predict_(self, X):
        X_transform = Preprocessing_PredictedData.preprocessing_pipeline(X, self.ct_data_transformer, with_duration)
        return self.model.predict(X_transform)

    def predict(self, X):
        return self.predict_(X)
  
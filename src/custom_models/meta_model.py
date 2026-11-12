import torch
import torchvision

import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split


from torchvision import transforms

from torchvision.transforms import v2

#################################################################
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from sklearn.utils import check_random_state

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

import pandas as pd
import mlflow
from mlflow import MlflowClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.base import clone

#GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
import optuna

from sklearn.preprocessing import StandardScaler #уточнить
from sklearn.base import BaseEstimator,TransformerMixin
#Metrics
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import check_is_fitted
#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from catboost import Pool

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap
import graphviz

import joblib

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

import mlflow
from mlflow import MlflowClient

import time
import subprocess
import tempfile
from pathlib import Path
from IPython.display import display
import json

import sys
import importlib

import src.modeling.train as train
from src.feature.features import FeatureTransformer
from src.feature.features import encoder_transformer
from config import Config
import src.utils.drawings as drawer
from src.utils.mymlflow import Mlflow_logger
from src.custom_models.myMLP import MyModel
from src.feature.feature_selection import count_off_shap
from src.feature.features import DNN_encoder
from src.utils.model_saving import ModelSaver
from src.modeling.predict import Predictor


class MyStackingRidge:
    """Объединяет прогнозы базовых моделей с помощью метамодели.

    Базовые модели формируют OOF-признаки для обучения метамодели.
    groups позволяет дополнительно усреднять выбранные столбцы прогнозов,
    а meta_cols определяет итоговый набор и порядок её входов.

    На новых данных сначала усредняются прогнозы фолдовых базовых моделей,
    затем применяются те же группировки и ансамбль метамоделей.
    В в бинарной классификации используются вероятности класса 1.
    """
    def __init__(self,meta_cols = None, groups = {}):

        def standard_pipe(model):
            return Pipeline([
                ("feature_transform", FeatureTransformer()),
                ("encoder", clone(encoder_transformer)),
                ("scaler", StandardScaler()),
                ("model", model)
            ])

        self.pipes = {
            "xgb": standard_pipe(XGBClassifier()),

            "catboost": Pipeline([
                ("feature_transform", FeatureTransformer(cat_boosting=True)),
                ("model", CatBoostClassifier(allow_writing_files=False))
            ]),

            "lightgbm": Pipeline([
                ("feature_transform", FeatureTransformer(cat_boosting=True)),
                ("model", LGBMClassifier())
            ]),

            "rf": standard_pipe(RandomForestClassifier()),

            "dnn": Pipeline([
                ("feature_transform", FeatureTransformer()),
                ("encoder", DNN_encoder()),
                ("model", MyModel())
            ]),

            "logebase": standard_pipe(LogisticRegression()),
        }

        self.meta_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ])

        self.ensembles = {}
        self.oof = {}

        # Настройка дополнительных комбинаций прогнозов
        # 
        self.groups = groups
        self.meta_cols = meta_cols if meta_cols else list(self.pipes)
        
        # self.groups = {"avg_boost": ["xgb", "catboost", "lightgbm"]}
        # self.meta_cols = ["avg_boost", "rf", "dnn", "logebase"]

        # self.groups = {"avg_boost": ["xgb", "catboost", "lightgbm"]}
        # self.meta_cols = ["xgb", "catboost", "lightgbm", "rf", "dnn", "logebase", "avg_boost"]


    def fit_sub_models(
        self, X, y,
        rf_params, dnn_params, cat_params,
        lgbm_params, xgb_params, ridge_params,
        cv_folds, random_state
    ):

        configs = {
            "xgb": xgb_params,
            "catboost": cat_params,
            "lightgbm": lgbm_params,
            "rf": rf_params,
            "dnn": dnn_params,
            "logebase": ridge_params
        }

        for name, config in configs.items():

            ensemble, results = train.Trainer.cross_validate(
                X=X,
                y=y,
                estimator=self.pipes[name],
                params=config["parameters"],
                fit_params=config["fit_param"],
                cv_folds=cv_folds,
                random_state=random_state
            )

            self.ensembles[name] = ensemble
            self.oof[name] = results["oof_pred"]
            print(f"Обучился {name}")

        return self


    def _meta_features(self, predictions):

        predictions = predictions.copy()

        for name, models in self.groups.items():
            predictions[name] = predictions[list(models)].mean(axis=1)

        return predictions[self.meta_cols]


    def fit(self, X, y, ridge_meta_params, cv_folds, random_state):

        oof_df = pd.DataFrame(self.oof, index=X.index)
        self.meta_data = self._meta_features(oof_df)

        self.meta_ensemble, meta_res = train.Trainer.cross_validate(
            X=self.meta_data,
            y=y,
            estimator=self.meta_pipe,
            params=ridge_meta_params["parameters"],
            fit_params=ridge_meta_params["fit_param"],
            cv_folds=cv_folds,
            random_state=random_state
        )

        return meta_res

    def predict_proba(self, X):
        predictions = pd.DataFrame({
            name: np.mean(
                [pipe.predict_proba(X)[:, 1] for pipe in ensemble],
                axis=0,
            )
            for name, ensemble in self.ensembles.items()
        }, index=X.index)

        meta_data = self._meta_features(predictions)

        proba = np.mean(
            [pipe.predict_proba(meta_data)[:, 1]
            for pipe in self.meta_ensemble],
            axis=0,
        )

        return np.column_stack([1 - proba, proba])


    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= Config.decision_threshold).astype(int)

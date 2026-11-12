import sys

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

# Models
from sklearn.linear_model import LogisticRegression

#################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Metrics
# Pipeline
from sklearn.pipeline import Pipeline

# GridSearch
from sklearn.preprocessing import StandardScaler  # уточнить
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Обучение моделей Titanic на CV и оценка на отложенной выборке.
# run_inference скаичивает все модели и проводит инференс
from config import Config
from src.custom_models.myMLP import CheckpointManager, EarlyStopping, MyModel
from src.feature.features import DNN_encoder, FeatureTransformer, encoder_transformer
from src.modeling.predict import Predictor
from src.utils.model_saving import ModelSaver
from src.utils.mymlflow import Mlflow_logger

if "src.core_funcs" in sys.modules:
    del sys.modules["src.core_funcs"]
from src.core_funcs import run_experiment, setup_validation
from src.custom_models.meta_model import MyStackingRidge


def run_meta_model(X_train, y_train):
    ridge_params = {
        "parameters": {
                          'model__C': 1.0,
                          'model__penalty': 'l2',
                          'model__solver': 'lbfgs',
                          'model__max_iter': 2000,
                          'model__random_state': Config.seed,
                      },
        "fit_param": {},
    }

    xgb_params = {
        "parameters": {
                          'model__device': 'cpu',
                          'model__tree_method': 'hist',
                          'model__n_estimators': 300,
                          'model__learning_rate': 0.03,
                          'model__max_depth': 3,
                          'model__min_child_weight': 1,
                          'model__subsample': 0.8,
                          'model__colsample_bytree': 0.8,
                          'model__reg_lambda': 1.0,
                          'model__objective': 'binary:logistic',
                          'model__eval_metric': 'logloss',
                          'model__random_state': Config.seed,
                          'model__n_jobs': -1,
                      },
        "fit_param": {},
    }

    rf_params = {
        "parameters": {
                          'model__n_estimators': 500,
                          'model__max_depth': 6,
                          'model__criterion': 'gini',
                          'model__min_samples_split': 10,
                          'model__min_samples_leaf': 3,
                          'model__max_features': 'sqrt',
                          'model__random_state': Config.seed,
                          'model__n_jobs': -1,
                      },
        "fit_param": {},
    }

    lgbm_params = {
        "parameters": {
                          'model__n_estimators': 1000,
                          'model__learning_rate': 0.03,
                          'model__max_depth': 4,
                          'model__num_leaves': 15,
                          'model__min_child_samples': 20,
                          'model__objective': 'binary',
                          'model__metric': 'binary_logloss',
                          'model__verbosity': -1,
                          'model__random_state': Config.seed,
                          'model__n_jobs': -1,
                          'model__early_stopping_round': 50,
                      },
        "fit_param": {},
    }

    cat_params = {
        "parameters": {
                          'model__iterations': 1000,
                          'model__learning_rate': 0.03,
                          'model__depth': 4,
                          'model__loss_function': 'Logloss',
                          'model__eval_metric': 'Accuracy',
                          'model__l2_leaf_reg': 3.0,
                          'model__random_seed': Config.seed,
                          'model__verbose': False,
                          'model__early_stopping_rounds': 50,
                          'model__thread_count': -1,
                      },
        "fit_param": {},
    }

    dnn_params = {
        "parameters": {
                          'model__cat_cols': FeatureTransformer.X_cat_cols,
                          'model__num_cols': FeatureTransformer.X_num_cols,
                          'model__epochs': 1000,
                          'model__device': Config.device,
                          'model__emb_dim': 12,
                          'model__inner_n': 128,
                          'model__drop_n': 0.2,
                          'model__n_layers': 2,
                          'model__use_batchnorm': False,
                          'model__loss_cls': torch.nn.CrossEntropyLoss,
                          'model__l1_lambda': 0.0,
                          'model__optim': torch.optim.Adam,
                          'model__optim_lr': 0.001,
                          'model__scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                          'model__lr_scheduler': True,
                          'model__factor': 0.5,
                          'model__patience': 10,
                          'model__threshold': 0.001,
                          'model__min_lr': 1e-8,
                          'model__early_stopping_patience': 40,
                          'model__early_stopping_threshold': 0.001,
                          'model__batch_size': 32,
                          'model__saving_threshold': 0.001,
                          'model__save_path': 'checkpoints',
                      },
        "fit_param": {"model__callbacks": [CheckpointManager(), EarlyStopping()]},
    }

    ridge_meta_params = {
        "parameters": {
                          'model__C': 1.0,
                          'model__penalty': 'l2',
                          'model__solver': 'lbfgs',
                          'model__max_iter': 2000,
                          'model__random_state': Config.seed,
                      },
        "fit_param": {},
    }
    # ------------------------------------------------------------------------------------------------------------------------------------
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.seed)
    cv_folds = list(kf.split(X_train, y_train))

    meta_ridge = MyStackingRidge(
        groups={"avg_boost": ["xgb", "catboost", "lightgbm"]},
        meta_cols=["avg_boost", "rf", "dnn", "logebase"],
    )
    meta_ridge.fit_sub_models(
        X_train,
        y_train,
        rf_params,
        dnn_params,
        cat_params,
        lgbm_params,
        xgb_params,
        ridge_params,
        cv_folds,
        Config.seed,
    )

    meta_res = meta_ridge.fit(
        X_train, y_train, ridge_meta_params, cv_folds, Config.seed
    )
    ModelSaver.save_ensemble([meta_ridge], "meta_logreg")
    return meta_res["oof_accuracy"]


def run_cv_all_models():
    """Обучает классификаторы и стекинг, выводит OOF и hold-out accuracy."""
    # Стартовые параметры Titanic; в приложенном EDA нет результатов их подбора.
    all_pipelines = {
        "logreg": {
            "logreg": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression()),
                    ]
                ),
                "parameters": {
                                  'model__C': [1.0],
                                  'model__penalty': ['l2'],
                                  'model__solver': ['lbfgs'],
                                  'model__max_iter': [2000],
                                  'model__random_state': [Config.seed],
                              },
                "search_type": "grid",
                "fit_param": {},
                "n_iter": 1,
                "plot_param": 'model__C',
                "heatmap_params": ('model__C', 'model__max_iter'),
                "shap_mode": "linear",
                "cv": 5,
                "n_jobs": -1,
            }
        },
        "dnn": {
            "dnn": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", DNN_encoder()),
                        ("model", MyModel()),
                    ]
                ),
                "parameters": {
                                  'model__cat_cols': [FeatureTransformer.X_cat_cols],
                                  'model__num_cols': [FeatureTransformer.X_num_cols],
                                  'model__epochs': [1000],
                                  'model__device': [Config.device],
                                  'model__emb_dim': [12],
                                  'model__inner_n': [128],
                                  'model__drop_n': [0.2],
                                  'model__n_layers': [2],
                                  'model__use_batchnorm': [False],
                                  'model__loss_cls': [torch.nn.CrossEntropyLoss],
                                  'model__l1_lambda': [0.0],
                                  'model__optim': [torch.optim.Adam],
                                  'model__optim_lr': [0.001],
                                  'model__scheduler': [torch.optim.lr_scheduler.ReduceLROnPlateau],
                                  'model__lr_scheduler': [True],
                                  'model__factor': [0.5],
                                  'model__patience': [10],
                                  'model__threshold': [0.001],
                                  'model__min_lr': [1e-8],
                                  'model__early_stopping_patience': [40],
                                  'model__early_stopping_threshold': [0.001],
                                  'model__batch_size': [32],
                                  'model__saving_threshold': [0.001],
                                  'model__save_path': ['checkpoints'],
                              },
                "search_type": "grid",
                "fit_param": {
                    "model__callbacks": [CheckpointManager(), EarlyStopping()]
                },  # ,MLflow_logger()
                "n_iter": 1,  # для рандом серча
                "plot_param": "model__optim_lr",
                "heatmap_params": ("model__optim_lr", "model__n_layers"),
                "shap_mode": "dnn",
                "cv": 5,
                "n_jobs": 1,
            }
        },
        "catboost": {
            "catboost": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer(cat_boosting=True)),
                        ("model", CatBoostClassifier(allow_writing_files=False)),
                    ]
                ),
                "parameters": {
                                  'model__iterations': [1000],
                                  'model__learning_rate': [0.03],
                                  'model__depth': [4],
                                  'model__loss_function': ['Logloss'],
                                  'model__eval_metric': ['Accuracy'],
                                  'model__l2_leaf_reg': [3.0],
                                  'model__random_seed': [Config.seed],
                                  'model__verbose': [False],
                                  'model__early_stopping_rounds': [50],
                                  'model__thread_count': [-1],
                              },
                "fit_param": {},
                "n_iter": 1,
                "plot_param": "model__learning_rate",
                "heatmap_params": ('model__learning_rate', 'model__l2_leaf_reg'),
                "shap_mode": "catboost",
                "cv": 5,
                "n_jobs": 1,
            }
        },
        "lightgbm": {
            "lightgbm": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer(cat_boosting=True)),
                        ("model", LGBMClassifier()),
                    ]
                ),
                "parameters": {
                                  'model__n_estimators': [1000],
                                  'model__learning_rate': [0.03],
                                  'model__max_depth': [4],
                                  'model__num_leaves': [15],
                                  'model__min_child_samples': [20],
                                  'model__objective': ['binary'],
                                  'model__metric': ['binary_logloss'],
                                  'model__verbosity': [-1],
                                  'model__random_state': [Config.seed],
                                  'model__n_jobs': [-1],
                                  'model__early_stopping_round': [50],
                              },
                "fit_param": {},
                "n_iter": 1,
                "plot_param": "model__learning_rate",
                "heatmap_params": ("model__learning_rate", "model__max_depth"),
                "shap_mode": "lightgbm",
                "cv": 5,
                "n_jobs": 1,
            }
        },
        "knn": {
            "knn": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", KNeighborsClassifier()),
                    ]
                ),
                "parameters": {
                                  'model__n_neighbors': [9],
                                  'model__weights': ['distance'],
                                  'model__p': [1],
                                  'model__n_jobs': [-1],
                              },
                "search_type": "grid",
                "fit_param": {},
                "n_iter": 1,
                "plot_param": "model__n_neighbors",
                "heatmap_params": ("model__n_neighbors", "model__p"),
                "shap_mode": "dnn",
                "cv": 5,
                "n_jobs": -1,
            }
        },
        "decision_tree": {
            "decision_tree": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", DecisionTreeClassifier()),
                    ]
                ),
                "parameters": {
                                  'model__max_depth': [5],
                                  'model__criterion': ['gini'],
                                  'model__min_samples_split': [10],
                                  'model__min_samples_leaf': [5],
                                  'model__random_state': [Config.seed],
                              },
                "search_type": "grid",
                "fit_param": {},
                "n_iter": 1,
                "plot_param": "model__max_depth",
                "heatmap_params": ('model__max_depth', 'model__min_samples_leaf'),
                "shap_mode": "tree",
                "cv": 5,
                "n_jobs": -1,
            }
        },
        "random_forest": {
            "random_forest": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", RandomForestClassifier()),
                    ]
                ),
                "parameters": {
                                  'model__n_estimators': [500],
                                  'model__max_depth': [6],
                                  'model__criterion': ['gini'],
                                  'model__min_samples_split': [10],
                                  'model__min_samples_leaf': [3],
                                  'model__max_features': ['sqrt'],
                                  'model__random_state': [Config.seed],
                                  'model__n_jobs': [-1],
                              },
                "search_type": "grid",
                "fit_param": {},
                "n_iter": 1,
                "plot_param": 'model__n_estimators',
                "heatmap_params": ("model__n_estimators", "model__max_depth"),
                "shap_mode": "tree",
                "cv": 5,
                "n_jobs": 1,
            }
        },
        "xgboost": {
            "xgboost": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", XGBClassifier()),
                    ]
                ),
                "parameters": {
                                  'model__device': ['cpu'],
                                  'model__tree_method': ['hist'],
                                  'model__n_estimators': [300],
                                  'model__learning_rate': [0.03],
                                  'model__max_depth': [3],
                                  'model__min_child_weight': [1],
                                  'model__subsample': [0.8],
                                  'model__colsample_bytree': [0.8],
                                  'model__reg_lambda': [1.0],
                                  'model__objective': ['binary:logistic'],
                                  'model__eval_metric': ['logloss'],
                                  'model__random_state': [Config.seed],
                                  'model__n_jobs': [-1],
                              },
                "search_type": "grid",
                "fit_param": {},
                "n_iter": 1,
                "plot_param": "model__learning_rate",
                "heatmap_params": ("model__learning_rate", "model__n_estimators"),
                "shap_mode": "tree",
                "cv": 5,
                "n_jobs": 1,
            }
        },
    }
    # cmd - python -m mlflow server --port 5000
    logger = Mlflow_logger("Titanic")
    logger.setup_connection()

    if "Survived" not in pd.read_csv(Config.path_train_data, nrows=0).columns:
        raise ValueError("Config.path_train_data должен указывать на train.csv Titanic с Survived.")

    X_train, X_holdout, y_train, y_holdout = setup_validation(
        Config.path_train_data, test_size=0.2
    )

    all_results = {"model": [], "oof_accuracy": []}

    for model_name, pipeline in all_pipelines.items():
        model_results = run_experiment(X_train, y_train, model_name, pipeline, logger)
        all_results["model"].append(model_name)
        all_results["oof_accuracy"].append(model_results["best_value"])
        ModelSaver.save_ensemble(model_results["best_ensemble"], model_name)

    stacking_oof = run_meta_model(X_train, y_train)
    all_results["model"].append("meta_logreg")
    all_results["oof_accuracy"].append(stacking_oof)

    results = pd.DataFrame(all_results)

    saved_models = ModelSaver.upload_saved_models(Config.path_to_save_models)
    hold_out_res = Predictor.run_test(saved_models, X_holdout, y_holdout)

    results["holdout_accuracy"] = results["model"].map(hold_out_res.set_index("model")["holdout_accuracy"])
    results = results.sort_values("oof_accuracy", ascending=False)

    print("\nРезультаты моделей\n")
    print(
        results.to_markdown(
            index=False,
            tablefmt="rounded_outline",
            floatfmt=".5f",
        )
)



if __name__ == "__main__":
    run_cv_all_models()

import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import joblib
from config import Config
from sklearn.metrics import accuracy_score

class Predictor:    
    @staticmethod
    def run_test(pipelines, X_holdout, y_holdout):
        """Оценивает сохранённые ансамбли на отложенной выборке.

        В бинарной классификации усредняет вероятности класса 1,
        после чего применяет порог Config.decision_threshold.

        Args:
            pipelines: Словарь ансамблей с обученными объектами в поле models.
            X_holdout: Признаки отложенной выборки.
            y_holdout: Таргет в той же шкале и порядке строк, что при обучении.

        Returns:
            DataFrame: Названия ансамблей и значения holdout-метрики.
        """
        scores = {}

        for model_name, fold_pipelines in pipelines.items():
            fold_pipelines = fold_pipelines['models']
            mean_proba = np.mean(
                [pipe.predict_proba(X_holdout)[:, 1]
                for pipe in fold_pipelines],
                axis=0,
            )

            pred = (mean_proba >= Config.decision_threshold).astype(int)
            scores[model_name] = accuracy_score(y_holdout, pred)

        results = (
            pd.DataFrame(scores.items(), columns=["model", "holdout_accuracy"])
            .sort_values("holdout_accuracy", ascending=False)
            .reset_index(drop=True)
        )

        print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        return results


    @staticmethod
    # Предсказания на Kaggle test и отдельный CSV для каждой модели
    def run_inference(pipelines, path_to_test = Config.path_test_data, output_dir=Config.path_to_submission):
        """Создаёт и сохраняет отдельный файл предсказаний для каждого ансамбля.

        Загружает тестовые данные и объединяет прогнозы фолдовых моделей.
        Преобразование выхода модели и формат CSV определяются задачей:
        для классификации к усреднённым вероятностям применяется порог решения.

        Args:
            pipelines: Словарь ансамблей с обученными объектами в поле models.
            path_to_test: Путь к CSV с тестовыми признаками.
            output_dir: Папка для файлов submission_<название>.csv.

        Результат работы метода — сохранённые CSV-файлы.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_test = pd.read_csv(path_to_test)

        predictions = {}

        for model_name, fold_pipelines in pipelines.items():
            fold_pipelines = fold_pipelines['models']

            mean_proba = np.mean(
                [pipe.predict_proba(data_test)[:, 1] for pipe in fold_pipelines],
                axis=0,
            )

            submission = pd.DataFrame({
                "PassengerId": data_test["PassengerId"].to_numpy(),
                "Survived": (mean_proba >= Config.decision_threshold).astype(int),
            })
            
            path = output_dir / f"submission_{model_name}.csv"
            submission.to_csv(path, index=False)

            predictions[model_name] = submission
            print(f"{model_name}: {len(submission)} строк → {path}")

        return predictions
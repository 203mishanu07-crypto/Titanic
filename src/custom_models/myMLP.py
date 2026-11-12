from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.custom_datasets.dataset import transform_data
from config import Config

class Callback:
    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass


class CheckpointManager(Callback):
    """Хранит лучшие веса сети в памяти и восстанавливает их после обучения.

    Копия state_dict сохраняется на CPU при достаточном улучшении
    контролируемой метрики. Порог улучшения берётся из saving_threshold
    обучаемой модели.
    """
    def __init__(self):
        self.best_state = None

    def on_train_begin(self, trainer):
        self.best_state = None

    def is_better(self, trainer):
        return (
            trainer.best_epoch is None
            or trainer.val_loss
            < trainer.best_score - trainer.saving_threshold
        )

    def on_epoch_end(self, trainer):
        if self.is_better(trainer):

            trainer.best_score = float(trainer.val_loss)
            trainer.best_epoch = trainer.epoch

            # Копируем лучшие веса в RAM (на CPU)
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in trainer.state_dict().items()
            }

    def on_train_end(self, trainer):
        if self.best_state is not None:
            trainer.load_state_dict(self.best_state)


class EarlyStopping(Callback):
    """Останавливает обучение после заданного числа эпох без улучшения.

    Улучшением считается изменение контролируемой метрики не меньше
    early_stopping_threshold. Допустимое число эпох без такого улучшения
    задаётся early_stopping_patience.
    """
    def on_train_begin(self, trainer):
        self.patience_ = trainer.early_stopping_patience
        self.threshold_ = trainer.early_stopping_threshold

        self.best_score = float("inf")
        self.count = 0

    def changed_better(self, current, best):
        return current < best - self.threshold_

    def on_epoch_end(self, trainer):
        current = float(trainer.val_loss)
        if self.changed_better(current, self.best_score):
            self.best_score = current
            self.count = 0
        else:
            self.count += 1
        if self.count >= self.patience_:
            trainer.stop_train = True



class MyModel(ClassifierMixin, nn.Module, BaseEstimator):
    """Обёртка над табличной нейронной сетью для работы в sklearn-пайплайне.

    Объединяет эмбеддинги категориальных признаков с числовыми входами
    и передаёт их в полносвязную сеть. Ожидает данные, уже подготовленные
    внешним encoder.

    Управляет обучением по батчам, scheduler, callbacks и историей метрик.
    Выход сети, функция потерь и методы предсказания соответствуют
    конкретной реализации задачи.
    """
    def __init__(
        self,
        cat_cols=None,
        num_cols=None,
        epochs=250,
        device="cuda",

        # Архитектура
        emb_dim=12,
        inner_n=128,
        drop_n=0.2,
        n_layers=2,
        use_batchnorm=False,

        # Loss
        loss_cls=nn.CrossEntropyLoss,

        # Reg
        l1_lambda=0.0,

        # Optimizer
        optim=torch.optim.Adam,
        optim_lr=0.005,

        # Scheduler
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler=True,
        factor=0.9,
        patience=10,
        threshold=0.01,
        min_lr=1e-8,

        # Early Stopping
        early_stopping_patience=60,
        early_stopping_threshold=1e-5,

        # DataLoader
        batch_size=32,

        # Checkpoint
        saving_threshold=0.001,
        save_path="checkpoints",

        callbacks=None,
    ):
        super().__init__()

        # Данные
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.epochs = epochs
        self.device = device

        # Архитектура
        self.emb_dim = emb_dim
        self.inner_n = inner_n
        self.drop_n = drop_n
        self.n_layers = n_layers
        self.use_batchnorm = use_batchnorm

        # Loss
        self.loss_cls = loss_cls

        # l1 reg
        self.l1_lambda=l1_lambda

        # Optimizer
        self.optim = optim
        self.optim_lr = optim_lr

        # Scheduler
        self.scheduler = scheduler
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        # Early Stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

        # DataLoader
        self.batch_size = batch_size

        # Checkpoint
        self.saving_threshold = saving_threshold
        self.save_path = save_path

        # Callbacks
        self.callbacks = callbacks

    def forward(self, x_num, x_cat):
        cat_embeddings = []
        for i, embedding in enumerate(self.embeddings):
            category_indexes = x_cat[:, i]
            cat_embeddings.append(embedding(category_indexes))
        x = torch.cat([x_num] + cat_embeddings, dim=1)
        return self.network(x)

    def _init_training_objects(self):
        # loss_cls — экземпляр nn.Module или класс nn.Module.
        self.loss_fn = self.loss_cls().to(self.device)

        self.opt = self.optim(
            self.parameters(),
            lr=self.optim_lr,
        )

        self.lr_scheduler_ = None

        if self.lr_scheduler and self.scheduler is not None:
            self.lr_scheduler_ = self.scheduler(
                self.opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                min_lr=self.min_lr,
            )
        
    def fit(self, X, y, eval_set,fold_idx):
        """Создаёт сеть и обучает её с оценкой на переданном eval_set.

        При каждом вызове заново создаёт архитектуру и объекты обучения.
        После каждой эпохи сохраняет метрики и learning rate, выполняет шаг
        scheduler и вызывает callbacks.

        Args:
            X: DataFrame с подготовленными признаками. Категории представлены
                целочисленными кодами, числовые признаки уже масштабированы.
            y: Целевая переменная в порядке строк X.
            eval_set: Пара (X_val, y_val), подготовленная тем же preprocessor.
            fold_idx: Номер фолда для истории обучения и callbacks.

        Returns:
            MyModel: Обученный экземпляр. Если подключён CheckpointManager,
                после завершения восстанавливаются сохранённые лучшие веса.
        """
        self.fold_idx = fold_idx

        self.cat_cols_ = [
            col for col in self.cat_cols
            if col in X.columns
        ]

        self.num_cols_ = [
            col for col in self.num_cols
            if col in X.columns
        ]
        #Распаковываем X
        self.cat_num = [
            int(X[col].max()) + 1
            for col in self.cat_cols_]
        
        self.num_num =  X[self.num_cols_].shape[1]

        X_train_cat = X[self.cat_cols_]
        X_train_num = X[self.num_cols_]

        # Проверяем, что все входные колонки учтены
        used_columns = set(self.cat_cols_ + self.num_cols_)
        unknown = set(X.columns) - used_columns
        y_array = np.asarray(y).reshape(-1)

        if unknown:
            raise ValueError(
                f"Входные признаки не отнесены к cat/num: {unknown}"
            )
    
        if len(y_array) != len(X):
            raise ValueError("Количество строк X и y различается.")

        if not np.isin(y_array, [0, 1]).all():
            raise ValueError("Ожидаются метки классов 0 и 1.")

        X_val, y_val = eval_set

        X_val_cat = X_val[self.cat_cols_]
        X_val_num = X_val[self.num_cols_]

        (
            train_loader,
            val_loader,
        ) = transform_data(
            X_train_cat, X_train_num, 
            X_val_cat,X_val_num, y, y_val,
            self.cat_cols_,
            self.num_cols_,
            self.batch_size,
        )

        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        
        self.build_model()
        self.to(self.device)
        self._init_training_objects()

        self.learning_rate_epoch = []
        self.epochs_train_loss = []
        self.epochs_val_loss = []
        self.epochs_train_accuracy = []
        self.epochs_val_accuracy = []
        self.best_score = float("inf")
        self.best_epoch = None
        self.stop_train = False
        self.epoch = -1

        self.callbacks_ = (
            deepcopy(self.callbacks)
            if self.callbacks is not None
            else [CheckpointManager(), EarlyStopping()]
        )

        for callback in self.callbacks_:
            callback.on_train_begin(self)

        for epoch in tqdm(range(self.epochs), desc="Training", leave=False):
            self.epoch = epoch
            self.current_lr = self.opt.param_groups[0]["lr"]
            self.learning_rate_epoch.append(self.current_lr)

            self.train_loss, self.train_accuracy = self.run_train_epoch(
                train_loader
            )
            self.val_loss, self.val_accuracy = self.run_val_epoch(
                val_loader
            )

            self.epochs_train_loss.append(self.train_loss)
            self.epochs_val_loss.append(self.val_loss)
            self.epochs_train_accuracy.append(self.train_accuracy)
            self.epochs_val_accuracy.append(self.val_accuracy)

            if self.lr_scheduler_ is not None:
                if isinstance(self.lr_scheduler_, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler_.step(self.val_loss)
                else:
                    self.lr_scheduler_.step()

            for callback in self.callbacks_:
                callback.on_epoch_end(self)

            if self.stop_train:
                print(f"Обучение остановлено на эпохе {epoch + 1}")
                break

        for callback in self.callbacks_:
            callback.on_train_end(self)

        self.is_fitted_ = True
        return self

    def build_model(self):
        embeddings = []
        for n_cat in self.cat_num:
            embeddings.append(nn.Embedding(
                num_embeddings=n_cat,
                embedding_dim=self.emb_dim,
                padding_idx=0,
            ))
        self.embeddings = nn.ModuleList(embeddings)
        n_embeddings = sum(embedding.embedding_dim for embedding in self.embeddings)
        n_features = self.num_num + n_embeddings

        layers = []
        for _ in range(self.n_layers):
            layers.append(nn.Linear(n_features, self.inner_n))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.inner_n))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.drop_n))
            n_features = self.inner_n
        layers.append(nn.Linear(n_features, 2))
        self.network = nn.Sequential(*layers)

    def run_train_epoch(self, train_loader):
        self.train()

        loss_sum = 0.0
        correct = 0
        total_samples = 0

        for x_num, x_cat, targets in train_loader:
            x_num = x_num.to(self.device, dtype=torch.float32)
            x_cat = x_cat.to(self.device, dtype=torch.long)
            targets = targets.to(
                self.device, dtype=torch.long
            ).reshape(-1)

            self.opt.zero_grad()

            logits = self(x_num, x_cat)
            data_loss = self.loss_fn(logits, targets)

            # L1-регуляризация весов линейных слоёв
            l1_penalty = sum(
                layer.weight.abs().sum()
                for layer in self.network
                if isinstance(layer, nn.Linear)
            )

            loss = data_loss + self.l1_lambda * l1_penalty
            loss.backward()
            self.opt.step()

            batch_size = targets.numel()

            # В историю записываем cross entropy без L1
            loss_sum += data_loss.detach().item() * batch_size

            proba = torch.softmax(logits.detach(), dim=1)[:, 1]
            predicted_classes = (
                proba >= Config.decision_threshold).long()

            correct += (predicted_classes == targets).sum().item()
            total_samples += batch_size

        return loss_sum / total_samples, correct / total_samples


    def run_val_epoch(self, val_loader):
        self.eval()

        loss_sum = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_num, x_cat, targets in val_loader:
                x_num = x_num.to(self.device, dtype=torch.float32)
                x_cat = x_cat.to(self.device, dtype=torch.long)
                targets = targets.to(
                    self.device, dtype=torch.long
                ).reshape(-1)

                logits = self(x_num, x_cat)
                loss = self.loss_fn(logits, targets)

                batch_size = targets.numel()
                loss_sum += loss.item() * batch_size

                
                proba = torch.softmax(logits.detach(), dim=1)[:, 1]
                predicted_classes = (
                    proba >= Config.decision_threshold).long()

                correct += (predicted_classes == targets).sum().item()
                total_samples += batch_size

        return loss_sum / total_samples, correct / total_samples


    def _predict_logits(self, X):
        check_is_fitted(self, "is_fitted_")
        self.eval()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        x_cat = torch.as_tensor(
            X[self.cat_cols_].to_numpy(),
            dtype=torch.long,
            device=self.device,
        )
        x_num = torch.as_tensor(
            X[self.num_cols_].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            logits = self(x_num, x_cat)

        return logits.cpu().numpy()  # Без reshape(-1).


    def predict_proba(self, X):
        logits = self._predict_logits(X)

        # Стабильный softmax.
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)

        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return self.classes_[
            (proba >= Config.decision_threshold).astype(int)
        ]

    #Возвращает оценку уверенности в классе 1 до преобразования в вероятность
    def decision_function(self, X):
        logits = self._predict_logits(X)
        return logits[:, 1] - logits[:, 0]
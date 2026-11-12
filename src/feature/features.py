import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


from pathlib import Path
from tqdm import tqdm

#Добавить управление объемом признаков через подачу колонок
class FeatureTransformer(TransformerMixin, BaseEstimator):
    """Подготавливает исходные данные для обучения и предсказания.

    Заполняет пропуски, кодирует признаки и создаёт производные колонки.
    Статистики, зависящие от обучающих данных, сохраняются в fit и затем
    используются при преобразовании новых данных.

    selected_columns задаёт итоговый набор и порядок признаков.
    cat_boosting включает представление категорий для моделей,
    которые обрабатывают их самостоятельно.
    """
    X_cols_origin = [
        "PassengerId",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked"]

    X_cols = [
        "Family_TE",
        "Pclass",
        "Sex",
        "Embarked",
        "Initial",
        "Age_band",
        "Rich_Woman",
        'Age',
        "Alone",
        "Family_size",
        "Fare",
        "Fare_cat",
        "Age*Class"]

    X_cat_cols = [
        "Embarked",
        "Initial"]
    
    X_num_cols = [
        "Family_TE",
        'Pclass',
        "Sex",
        'Age',
        "Age_band",
        "Rich_Woman",
        "Alone",
        "Family_size",
        "Fare",
        "Fare_cat", 
        "Age*Class"]

    def __init__(self,m=5, cat_boosting = False, selected_columns = None):
        self.m = m
        self.initial_mapping = {
            "Mlle": "Miss",
            "Mme": "Miss",
            "Ms": "Miss",
            "Dr": "Mr",
            "Major": "Mr",
            "Lady": "Mrs",
            "Countess": "Mrs",
            "Jonkheer": "Other",
            "Col": "Other",
            "Rev": "Other",
            "Capt": "Mr",
            "Sir": "Mr",
            "Don": "Mr",
            "Dona": "Mrs",}
        self.selected_columns = selected_columns
        self.cat_boosting = cat_boosting
        self.X_cat_cols = type(self).X_cat_cols.copy()
        
    def fit(self, data, y=None):
        if y is None:
            raise ValueError("Для семейного кодирования нужен y.")

        if self.m < 0:
            raise ValueError("m должен быть положительным.")

        data = data.copy()

        self.median_fare_ = data["Fare"].median()
        self.embarked_mode_ = data["Embarked"].mode().iloc[0]

        data = self.create_features(data)

        data["Survived"] = np.asarray(y).reshape(-1)
        self.y_ = data["Survived"].mean()

        family_stats = (
            data.groupby("FamilyID")["Survived"]
            .agg(["sum", "count"])
        )

        self.stat = (
            family_stats["sum"] + self.m * self.y_
        ) / (
            family_stats["count"] + self.m
        )

        if self.selected_columns is not None:
            unknown = set(self.selected_columns) - set(self.X_cols)

            if unknown:
                raise ValueError(f"Неизвестные признаки: {unknown}")
            
        self.is_fitted_ = True
        return self

    def transform(self,data):    
        check_is_fitted(self, "is_fitted_")
        data_transformed = data.copy()
        data_transformed = self.create_features(data_transformed)

        if self.cat_boosting:
            data_transformed["Family_TE"] = (data_transformed["FamilyID"])
        else:

            data_transformed["Family_TE"] = (
                data_transformed["FamilyID"]
                .map(self.stat)
                .fillna(self.y_)
            )

        data_transformed = data_transformed.drop(
            columns=["FamilyID"]
        )
        columns = (
            self.selected_columns
            if self.selected_columns is not None
            else self.X_cols
        )

        result = data_transformed.loc[:, columns].copy()

        if self.cat_boosting:
            cat_cols = self.get_existing_cat_cols(result)

            for col in cat_cols:
                result[col] = result[col].astype("category")

        return result

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        data_transformed = self.create_features(data)

        if self.cat_boosting:
            data_transformed["Family_TE"] = data_transformed["FamilyID"]

        else:
            data_transformed["Survived"] = np.asarray(y).reshape(-1)

            groups = data_transformed.groupby("FamilyID")["Survived"]

            family_sum = groups.transform("sum")
            family_count = groups.transform("count")

            data_transformed["Family_TE"] = (
                family_sum
                - data_transformed["Survived"]
                + self.m * self.y_
            ) / (
                family_count - 1 + self.m
            )

        columns = (
            self.selected_columns
            if self.selected_columns is not None
            else self.X_cols
        )
        
        result = data_transformed.loc[:, columns].copy()

        if self.cat_boosting:
            cat_cols = self.get_existing_cat_cols(result)

            for col in cat_cols:
                result[col] = result[col].astype("category")

        return result


    def create_features(self,data):
        data = data.copy()
        #Initial
        data["Initial"] = 0

        data["Initial"] = (
            data["Name"]
            .str.extract(r"([A-Za-z]+)\.", expand=False)
            .replace(self.initial_mapping)
        )

        valid_titles = ["Mr", "Mrs", "Miss", "Master", "Other"]

        data.loc[~data["Initial"].isin(valid_titles), "Initial"] = "Other"

        data = self.fill_missing(data)
        #Rich woman
        data["Rich_Woman"] = ((data["Sex"] == "female") & (data["Pclass"] == 1)).astype(int)

        #Family_size/Alone
        data["Alone"] = 0
        data["Family_size"] = data["SibSp"] + data["Parch"]
        data.loc[data["Family_size"] == 0, "Alone"] = 1
        data["Alone"] = (data["Family_size"] == 0).astype(int)

        #Fare cat
        data["Fare_cat"] = 0
        data.loc[data["Fare"] <= 7.91, "Fare_cat"] = 0
        data.loc[
            (data["Fare"] > 7.91) & (data["Fare"] <= 14.454), "Fare_cat"
        ] = 1
        data.loc[
            (data["Fare"] > 14.454) & (data["Fare"] <= 31), "Fare_cat"
        ] = 2
        data.loc[data["Fare"] > 31, "Fare_cat"] = 3

        #Age band 
        data["Age_band"] = 0
        data.loc[data["Age"] <= 16, "Age_band"] = 0
        data.loc[(data["Age"] > 16) & (data["Age"] <= 35), "Age_band"] = 1
        data.loc[(data["Age"] > 35) & (data["Age"] <= 48), "Age_band"] = 2
        data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age_band"] = 3
        data.loc[data["Age"] > 64, "Age_band"] = 4

        #Age*Class
        data["Age*Class"] = data["Age"] * data["Pclass"]

        #Кодируем пол
        data["Sex"] = (
            data["Sex"]
            .map({"male": 0, "female": 1})
            .astype("int64")
        )

        #Famile ID
        data["FamilyID"] = (
            data["Name"].str.split(",").str[0].str.strip() + "_" + data["Family_size"].astype(str))

        return data


    def fill_missing(self, data):
        data['Fare'] = data['Fare'].fillna(self.median_fare_)
        data['Embarked'] = data['Embarked'].fillna(self.embarked_mode_)
        
        data.loc[
            data["Age"].isna() & (data["Initial"] == "Mr"), "Age"
        ] = 33
        data.loc[
            data["Age"].isna() & (data["Initial"] == "Mrs"), "Age"
        ] = 36
        data.loc[
            data["Age"].isna() & (data["Initial"] == "Master"), "Age"
        ] = 5
        data.loc[
            data["Age"].isna() & (data["Initial"] == "Miss"), "Age"
        ] = 22
        data.loc[
            data["Age"].isna() & (data["Initial"] == "Other"), "Age"
        ] = 46
        return data

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")

        columns = (
            self.selected_columns
            if self.selected_columns is not None
            else self.X_cols
        )

        return np.asarray(columns, dtype=object)
    
    @staticmethod
    def get_existing_cat_cols(X):
        columns = [
            col for col in FeatureTransformer.X_cat_cols
            if col in X.columns
        ]

        if (
            "Family_TE" in X.columns
            and not pd.api.types.is_numeric_dtype(X["Family_TE"])
        ):
            columns.append("Family_TE")

        return columns

    @staticmethod
    def get_existing_num_cols(X):
        return [
            col for col in FeatureTransformer.X_num_cols
            if col in X.columns
            and pd.api.types.is_numeric_dtype(X[col])
        ]


encoder_transformer = ColumnTransformer(
    transformers=[
        (
            "one_hot",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            ),
            FeatureTransformer.get_existing_cat_cols,
        )
    ],
    remainder="passthrough",
)


class DNN_encoder(BaseEstimator, TransformerMixin):
    """Подготавливает категориальные и числовые признаки для нейронной сети.

    Категории кодируются целыми числами для подачи в эмбеддинги:
    известные значения получают коды от 1, неизвестные — код 0.
    Числовые признаки стандартизируются по статистикам обучающей выборки.

    Возвращает DataFrame: сначала категориальные колонки, затем числовые.
    Порядок строк и их индексы сохраняются.
    """
    def __init__(self, x_cat = FeatureTransformer.get_existing_cat_cols, x_num = FeatureTransformer.get_existing_num_cols):
        self.x_cat = x_cat
        self.x_num = x_num
        self.is_fitted_ = False

    def fit(self,X, y = None):
        self.encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1)
        self.scaler = StandardScaler()

        self.num_cols = self.x_num(X)
        self.cat_cols = self.x_cat(X)

        self.X_cat = X[self.cat_cols]
        self.X_num = X[self.num_cols]

        self.encoder.fit(self.X_cat)

        self.scaler.fit(self.X_num)

        self.cat_num = [
            len(categories) + 1
            for categories in self.encoder.categories_]

        self.n_num = self.X_num.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self,X):
        check_is_fitted(self, "is_fitted_")
    # Категориальные признаки
        X_transformed = pd.DataFrame(
            self.encoder.transform(X[self.cat_cols]).astype("int64") + 1,
            columns=self.cat_cols,
            index=X.index
        )
        # Числовые признаки
        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.num_cols]),
            columns=self.num_cols,
            index=X.index
        )

        return pd.concat([X_transformed,X_scaled], axis=1)

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.cat_cols + self.num_cols, dtype=object)


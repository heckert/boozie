from dataclasses import dataclass

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class ModelTrainer:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        *,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.X = X
        self.y = y

        self.features = X.columns

        self.test_size = test_size
        self.random_state = random_state

        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        self.model = None

    def _get_hyperopt_grid(self) -> Pipeline:
        pipeline = make_pipeline(StandardScaler(), RandomForestRegressor())

        param_grid = {
            "randomforestregressor__n_estimators": [50, 100, 200],
            "randomforestregressor__max_depth": [None, 10, 20, 30],
            "randomforestregressor__min_samples_split": [2, 5, 10],
            "randomforestregressor__min_samples_leaf": [1, 2, 4],
            "randomforestregressor__max_features": [None, "sqrt", "log2"],
            "randomforestregressor__random_state": [42],
        }

        grid = RandomizedSearchCV(
            pipeline,
            n_iter=50,
            param_distributions=param_grid,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=True,
            random_state=self.random_state,
        )

        return grid

    def _get_model(self) -> Pipeline:
        params = {
            "random_state": 42,
            "n_estimators": 200,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "log2",
            "max_depth": None,
        }

        return make_pipeline(StandardScaler(), RandomForestRegressor(**params))

    def train(self, optimize_hyperparams: bool = False) -> Pipeline:
        if optimize_hyperparams:
            model = self._get_hyperopt_grid()

        else:
            model = self._get_model()

        self.model = model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self) -> float:
        y_test_pred = self.model.predict(self.X_test)
        return mean_squared_error(self.y_test, y_test_pred)


@dataclass
class TrainingResult:
    mse: float
    model: Pipeline
    features: list


def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> TrainingResult:
    trainer = ModelTrainer(X, y)
    trainer.train()

    return TrainingResult(trainer.evaluate(), trainer.model, trainer.features)

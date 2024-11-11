from sklearn.base import BaseEstimator
from pptoolbox.platform.automl_v4 import AutoML_v4

import numpy as np
import pandas as pd

class HierarchicalClassifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.model_A = AutoML_v4(
            task_type = "classify", 
            exploration_runs = 5, 
            exploitation_runs = 5
        )
        self.model_B = AutoML_v4(
            task_type = "classify", 
            exploration_runs = 5, 
            exploitation_runs = 5
        )

        self.pipeline_A = None if "pipeline_A" not in kwargs else kwargs["pipeline_A"]
        self.pipeline_B = None if "pipeline_B" not in kwargs else kwargs["pipeline_B"]

        self.automl_fitted = False if "automl_fitted" not in kwargs else kwargs["automl_fitted"]

    def split_data(self, X, y):
        y_A = y[y <= 1]
        y_B = y[y > 1]

        X_A = X[y <= 1]
        X_B = X[y > 1]

        return X_A, y_A, X_B, y_B

    def _fit(self, X, y):
        X_A, y_A, X_B, y_B = self.split_data(X, y)

        self.model_A.fit(X_A, y_A)
        self.model_B.fit(X_B, y_B)

        self.pipeline_A = self.model_A.get_pipeline()
        self.pipeline_B = self.model_B.get_pipeline()

        self.automl_fitted = True

    def fit(self, X, y):
        if not self.automl_fitted:
            self._fit(X, y)

        X_A, y_A, X_B, y_B = self.split_data(X, y)

        self.pipeline_A.fit(X_A, y_A)
        self.pipeline_B.fit(X_B, y_B)

        return self

    def predict(self, X, threshold = 0.7):
        A_probs = self.pipeline_A.predict_proba(X)
        A_preds = np.where(A_probs > 0.7, 1, 0).max(axis = 1)

        A_preds = pd.Series(A_preds, index = X.index, name = "A pred")

        X_B = X[A_preds == 1]
        B_preds = pd.Series(self.pipeline_B.predict(X_B), index = X_B.index, name = "B pred")

        AB_pred = pd.merge(
            A_preds,
            B_preds,
            left_on = A_preds.index,
            right_on = B_preds.index
        ).set_index("key_0")

        return (AB_pred.apply(
            lambda row: 
            row["A pred"] 
            if row["A pred"] == 0 
            else row["B pred"],
            axis = 1 
        ))

    def get_params(self, deep = False):
        return {
            "pipeline_A": self.pipeline_A,
            "pipeline_B": self.pipeline_B,
            "automl_fitted": self.automl_fitted
        }

    def set_params(self, params):
        for param, value in params.items():
            self.__setattr__(param, value)
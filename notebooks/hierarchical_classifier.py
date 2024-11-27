import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold

from pptoolbox.platform.automl_v4 import AutoML_v4

class HierarchicalClassifier(BaseEstimator):
    def __init__(self, mapping, **kwargs):
        """
        Initialize the hierarchical classifier with two models and label encoder.
        
        Parameters:
        check_reject_model: A custom model or scikit-learn model to predict 'Check' or 'Reject'.
        pass_mild_model: A custom model or scikit-learn model to predict '1-Pass' or '2-Mild' if 'Check'.
        mapping: mapping from encoder that is created outside and should be passed to the class object
            use the following:
                mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

        """

        # self.check_reject_model = AutoML_v4(
        #     task_type = "classify", 
        #     criteria = "test_precision_micro", 
        #     exploration_runs=100, 
        #     exploitation_runs=200
        #     )

        # self.pass_mild_model = AutoML_v4(
        #     task_type = "classify", 
        #     criteria = "test_balanced_accuracy", 
        #     exploration_runs=100, 
        #     exploitation_runs=200
        #     )

        ### testing
        self.check_reject_model = AutoML_v4(
            task_type = "classify", 
            criteria = "test_precision_micro", 
            exploration_runs=5, 
            exploitation_runs=5
            )

        self.pass_mild_model = AutoML_v4(
            task_type = "classify", 
            criteria = "test_balanced_accuracy", 
            exploration_runs=5, 
            exploitation_runs=5
            )
        ###

        self.mapping = mapping # created outside

        self.check_reject_pipeline = None if "check_reject_pipeline" not in kwargs else kwargs["check_reject_pipeline"]
        self.pass_mild_pipeline = None if "pass_mild_pipeline" not in kwargs else kwargs["pass_mild_pipeline"]

        self.check_reject_trainer = None if "check_reject_trainer" not in kwargs else kwargs["check_reject_trainer"]
        self.pass_mild_trainer = None if "pass_mild_trainer" not in kwargs else kwargs["pass_mild_trainer"]

        self.automl_fitted = False if "automl_fitted" not in kwargs else kwargs["automl_fitted"]

    def relabel_data(self, X, y):
        # Convert y-labels into binary labels (0 = Check, 1 = Reject) for first model and fit model
        reject_label = self.mapping['3-Rancid']
        y_check_reject = pd.Series(np.where(y == reject_label, 1, 0),
                                  index = y.index)  # 0 = Check, 1 = Reject
        
        # Fit the second model (1-Pass/2-Mild) on 'Check' samples
        check_indices = y.index[np.where(y_check_reject == 0)[0]]
        X_check = X.loc[check_indices]
        y_check = y.loc[check_indices]  # These labels are still the original encoded (numeric) labels

        return y_check_reject, X_check, y_check
    
    def _fit(self, X, y):

        """
        Fit the check_reject model on the dataset and the Pass/Mild model on 'Check' samples.
        
        Parameters:
        X: Features for training.
        y: Labels for training. Should have been encoded.
        """

        y_check_reject, X_check, y_check = self.relabel_data(X, y)

        self.check_reject_model.fit(X, y_check_reject, kfold=StratifiedGroupKFold(n_splits=5))     
        self.pass_mild_model.fit(X_check, y_check, kfold=StratifiedGroupKFold(n_splits=5))
        
        self.check_reject_trainer = self.check_reject_model
        self.pass_mild_trainer = self.pass_mild_model

        self.check_reject_pipeline = self.check_reject_model.get_pipeline()
        self.pass_mild_pipeline = self.pass_mild_model.get_pipeline()

        self.automl_fitted = True

    def fit(self, X, y):
        """
        Fit the models only once. If fitted already, fit pipeline.
        """

        if not self.automl_fitted:
            self._fit(X, y)

        y_check_reject, X_check, y_check = self.relabel_data(X, y)

        self.check_reject_pipeline.fit(X, y_check_reject)
        self.pass_mild_pipeline.fit(X_check, y_check)

        return self

    def predict(self, X, threshold = 0.5):

        # Adjusted threshold for Reject
        y_probs = self.check_reject_pipeline.predict_proba(X)[:, 1]  # Probabilities for the 'Reject' class
        new_threshold = threshold  # taken as input
        check_reject_pred = np.where(y_probs >= new_threshold, 1, 0) # 0 = Check, 1 = Reject
        
        # Record Reject values and pass Check to second model
        final_preds = []
        for i, pred in enumerate(check_reject_pred):
            if pred == 1:
                final_preds.append(2)  # encoded 3-Rancid
            else:
                # If Check (0), use the second model to predict between 1-Pass and 2-Mild
                pass_mild_pred = self.pass_mild_pipeline.predict(X.reset_index().drop(columns=['lot_id']).iloc[[i]])
                final_preds.append(pass_mild_pred[0])
        
        return np.array(final_preds)
    
    def predict_proba(self, X):
        # Predict probabilities
        y_probs = self.check_reject_pipeline.predict_proba(X)[:, 1]  # Probabilities for the 'Reject' class
        
        return np.array(y_probs)
    
    def get_params(self, deep = False):
        return {
            "check_reject_trainer": self.check_reject_trainer,
            "pass_mild_trainer": self.pass_mild_trainer,
            "check_reject_pipeline": self.check_reject_pipeline,
            "pass_mild_pipeline": self.pass_mild_pipeline,
            "automl_fitted": self.automl_fitted,
            "mapping" : self.mapping
        }

    def set_params(self, params):
        for param, value in params.items():
            self.__setattr__(param, value)
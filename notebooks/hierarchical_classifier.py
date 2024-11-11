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
        pass_defect_model: A custom model or scikit-learn model to predict 'Pass' or 'Defect'.
        mild_rancid_model: A custom model or scikit-learn model to predict '2Mild' or '3Rancid' if 'Defect'.
        mapping: mapping from encoder that is created outside and should be passed to the class object
            use the following:
                mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

        """

        self.pass_defect_model = AutoML_v4(
            task_type = "classify", 
            criteria = "test_balanced_accuracy", 
            exploration_runs=100, 
            exploitation_runs=200
            )

        self.mild_rancid_model = AutoML_v4(
            task_type = "classify", 
            criteria = "test_balanced_accuracy", 
            exploration_runs=100, 
            exploitation_runs=200
            )

        self.mapping = mapping # created outside

        self.pass_defect_pipeline = None if "pass_defect_pipeline" not in kwargs else kwargs["pass_defect_pipeline"]
        self.mild_rancid_pipeline = None if "mild_rancid_pipeline" not in kwargs else kwargs["mild_rancid_pipeline"]

        self.automl_fitted = False if "automl_fitted" not in kwargs else kwargs["automl_fitted"]

    def relabel_data(self, X, y):
        # Convert y-labels into binary labels (0 = Pass, 1 = Defect) for first model and fit model
        pass_label = self.mapping['1Pass']
        y_pass_defect = pd.Series(np.where(y == pass_label, 0, 1),
                                  index = y.index)  # 0 = Pass, 1 = Defect
        
        # Fit the second model (2Mild/3Rancid) on 'Defect' samples
        defect_indices = y.index[np.where(y_pass_defect == 1)[0]]
        X_defect = X.loc[defect_indices]
        y_defect = y.loc[defect_indices]  # These labels are still the original encoded (numeric) labels

        return y_pass_defect, X_defect, y_defect
    
    def _fit(self, X, y):

        """
        Fit the pass/defect model on the dataset and the mild/rancid model on 'Defect' samples.
        
        Parameters:
        X: Features for training.
        y: Labels for training. Should have been encoded.
        """

        y_pass_defect, X_defect, y_defect = self.relabel_data(X, y)

        self.pass_defect_model.fit(X, y_pass_defect, kfold=StratifiedGroupKFold(n_splits=5))     
        self.mild_rancid_model.fit(X_defect, y_defect, kfold=StratifiedGroupKFold(n_splits=5))
        
        self.pass_defect_pipeline = self.pass_defect_model.get_pipeline()
        self.mild_rancid_pipeline = self.mild_rancid_model.get_pipeline()

        self.automl_fitted = True

    def fit(self, X, y):
        """
        Fit the models only once. If fitted already, fit pipeline.
        """

        if not self.automl_fitted:
            self._fit(X, y)

        y_pass_defect, X_defect, y_defect = self.relabel_data(X, y)

        self.pass_defect_pipeline.fit(X, y_pass_defect)
        self.mild_rancid_pipeline.fit(X_defect, y_defect)

        return self

    def predict(self, X, threshold = 0.5):

        # Adjusted threshold for Pass
        y_probs = self.pass_defect_pipeline.predict_proba(X)[:, 0]  # Probabilities for the 'Pass' class
        new_threshold = threshold  # taken as input
        pass_defect_pred = np.where(y_probs >= new_threshold, 0, 1) # 0 = Pass, 1 = Defect
        
        # Record Pass values and pass Defect to second model
        final_preds = []
        for i, pred in enumerate(pass_defect_pred):
            if pred == 0:
                final_preds.append(pred)  # encoded 1Pass
            else:
                # If Defect (1), use the second model to predict between 2Mild and 3Rancid
                mild_rancid_pred = self.mild_rancid_pipeline.predict(X.reset_index().drop(columns=['lot_id']).iloc[[i]])
                final_preds.append(mild_rancid_pred[0])
        
        return np.array(final_preds)
    
    def predict_proba(self, X):
        # Predict probabilities
        y_probs = self.pass_defect_pipeline.predict_proba(X)[:, 0]  # Probabilities for the 'Pass' class
        
        return np.array(y_probs)
    
    def get_params(self, deep = False):
        return {
            "pass_defect_pipeline": self.pass_defect_pipeline,
            "mild_rancid_pipeline": self.mild_rancid_pipeline,
            "automl_fitted": self.automl_fitted,
            "mapping" : self.mapping
        }

    def set_params(self, params):
        for param, value in params.items():
            self.__setattr__(param, value)
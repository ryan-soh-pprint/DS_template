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
        mild_rancid_model: A custom model or scikit-learn model to predict '3-Rancid' or '2-Mild' if 'Defect'.
        mapping: mapping from encoder that is created outside and should be passed to the class object
            use the following:
                mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

        """

        self.pass_defect_model = AutoML_v4(
            task_type = "classify", 
            )

        self.mild_rancid_model = AutoML_v4(
            task_type = "classify", 
            )

        # ## testing
        # self.pass_defect_model = AutoML_v4(
        #     task_type = "classify", 
        #     exploration = 5, 
        #     exploitation = 5
        #     )

        # self.mild_rancid_model = AutoML_v4(
        #     task_type = "classify", 
        #     exploration = 5, 
        #     exploitation = 5
        #     )
        # ##

        self.mapping = mapping # created outside

        self.pass_defect_pipeline = None if "pass_defect_pipeline" not in kwargs else kwargs["pass_defect_pipeline"]
        self.mild_rancid_pipeline = None if "mild_rancid_pipeline" not in kwargs else kwargs["mild_rancid_pipeline"]

        self.pass_defect_trainer = None if "pass_defect_trainer" not in kwargs else kwargs["pass_defect_trainer"]
        self.mild_rancid_trainer = None if "mild_rancid_trainer" not in kwargs else kwargs["mild_rancid_trainer"]

        self.automl_fitted = False if "automl_fitted" not in kwargs else kwargs["automl_fitted"]

    def relabel_data(self, X, y):
        # Convert y-labels into binary labels (0 = Pass, 1 = Defect) for first model and fit model
        pass_label = self.mapping['1-Pass']
        y_pass_defect = pd.Series(np.where(y == pass_label, 0, 1),
                                  index = y.index)  # 0 = Pass, 1 = Defect
        
        # Fit the second model (2-Mild/3-Rancid) on 'Defect' samples
        defect_indices = y.index[np.where(y_pass_defect == 1)[0]]
        X_check = X.loc[defect_indices]
        y_check = y.loc[defect_indices]  # These labels are still the original encoded (numeric) labels

        return y_pass_defect, X_check, y_check
    
    def _fit(self, X, y):

        """
        Fit the pass_defect model on the dataset and the Pass/Mild model on 'Check' samples.
        
        Parameters:
        X: Features for training.
        y: Labels for training. Should have been encoded.
        """

        y_pass_defect, X_check, y_check = self.relabel_data(X, y)

        self.pass_defect_model.fit(X, y_pass_defect)     
        self.mild_rancid_model.fit(X_check, y_check)
        
        self.pass_defect_trainer = self.pass_defect_model
        self.mild_rancid_trainer = self.mild_rancid_model

        self.pass_defect_pipeline = self.pass_defect_model.get_pipeline()
        self.mild_rancid_pipeline = self.mild_rancid_model.get_pipeline()

        self.automl_fitted = True

    def fit(self, X, y):
        """
        Fit the models only once. If fitted already, fit pipeline.
        """

        if not self.automl_fitted:
            self._fit(X, y)

        y_pass_defect, X_check, y_check = self.relabel_data(X, y)

        self.pass_defect_pipeline.fit(X, y_pass_defect)
        self.mild_rancid_pipeline.fit(X_check, y_check)

        return self

    def predict(self, X, threshold = 0.5):

        # Adjusted threshold for Reject
        y_probs = self.pass_defect_pipeline.predict_proba(X)[:, 0]  # Probabilities for the 'Pass' class
        new_threshold = threshold  # taken as input
        pass_defect_pred = np.where(y_probs >= new_threshold, 0, 1) # 0 = Pass, 1 = Defect
        
        # Record Reject values and pass Check to second model
        final_preds = []
        for i, pred in enumerate(pass_defect_pred):
            if pred == 0:
                final_preds.append(0)  # encoded 1-Pass
            else:
                # If Defect (1), use the second model to predict between 2-Mild/3-Rancid
                mild_rancid_pred = self.mild_rancid_pipeline.predict(X.reset_index().drop(columns=['lot_id']).iloc[[i]])
                final_preds.append(mild_rancid_pred[0])
        
        return np.array(final_preds)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for each label: Pass, Mild, Rancid.
        """
        # Step 1: Get probability estimates from pass_defect_pipeline
        pass_defect_probs = self.pass_defect_pipeline.predict_proba(X)
        P_Pass = pass_defect_probs[:, 0]  # Probability of Pass
        P_Defect = pass_defect_probs[:, 1]  # Probability of Defect

        # Step 2: Identify Defect samples and get probabilities from mild_rancid_pipeline
        defect_indices = np.where(P_Defect > 0.5)[0]  # Consider samples where Defect is likely
        P_Mild = np.zeros_like(P_Pass)
        P_Rancid = np.zeros_like(P_Pass)

        P_Mild = self.mild_rancid_pipeline.predict_proba(X)[:, 0]
        P_Rancid = self.mild_rancid_pipeline.predict_proba(X)[:, 1]

        # Step 3: Compute final probabilities
        P_Mild = P_Defect * P_Mild  # P(Mild) = P(Defect) * P(Mild | Defect)
        P_Rancid = P_Defect * P_Rancid  # P(Rancid) = P(Defect) * P(Rancid | Defect)
        
        return np.column_stack((P_Pass, P_Mild, P_Rancid))
    
    def get_params(self, deep = False):
        return {
            "pass_defect_trainer": self.pass_defect_trainer,
            "mild_rancid_trainer": self.mild_rancid_trainer,
            "pass_defect_pipeline": self.pass_defect_pipeline,
            "mild_rancid_pipeline": self.mild_rancid_pipeline,
            "automl_fitted": self.automl_fitted,
            "mapping" : self.mapping
        }

    def set_params(self, params):
        for param, value in params.items():
            self.__setattr__(param, value)

    class PipelineWrapper:
        def __init__(self, pass_defect_pipeline, mild_rancid_pipeline, mapping, **kwargs):
            self.pass_defect_pipeline = pass_defect_pipeline
            self.mild_rancid_pipeline = mild_rancid_pipeline
            self.mapping = mapping

        def relabel_data(self, X, y):
            # Convert y-labels into binary labels (0 = Pass, 1 = Defect) for first model and fit model
            pass_label = self.mapping['1-Pass']
            y_pass_defect = pd.Series(np.where(y == pass_label, 0, 1),
                                    index = y.index)  # 0 = Pass, 1 = Defect
            
            # Fit the second model (2-Mild/3-Rancid) on 'Defect' samples
            defect_indices = y.index[np.where(y_pass_defect == 1)[0]]
            X_check = X.loc[defect_indices]
            y_check = y.loc[defect_indices]  # These labels are still the original encoded (numeric) labels

            return y_pass_defect, X_check, y_check

        def fit(self, X, y):
            y_pass_defect, X_check, y_check = self.relabel_data(X, y)

            self.pass_defect_pipeline.fit(X, y_pass_defect)
            self.mild_rancid_pipeline.fit(X_check, y_check)

            return self

        def predict(self, X, threshold = 0.5):

            # Adjusted threshold for Reject
            y_probs = self.pass_defect_pipeline.predict_proba(X)[:, 0]  # Probabilities for the 'Pass' class
            new_threshold = threshold  # taken as input
            pass_defect_pred = np.where(y_probs >= new_threshold, 0, 1) # 0 = Pass, 1 = Defect
            
            # Record Reject values and pass Check to second model
            final_preds = []
            for i, pred in enumerate(pass_defect_pred):
                if pred == 0:
                    final_preds.append(0)  # encoded 1-Pass
                else:
                    # If Defect (1), use the second model to predict between 2-Mild/3-Rancid
                    mild_rancid_pred = self.mild_rancid_pipeline.predict(X.reset_index().drop(columns=['lot_id']).iloc[[i]])
                    final_preds.append(mild_rancid_pred[0])
            
            return np.array(final_preds)
        
        def predict_proba(self, X):
            """
            Predict class probabilities for each label: Pass, Mild, Rancid.
            """
            # Step 1: Get probability estimates from pass_defect_pipeline
            pass_defect_probs = self.pass_defect_pipeline.predict_proba(X)
            P_Pass = pass_defect_probs[:, 0]  # Probability of Pass
            P_Defect = pass_defect_probs[:, 1]  # Probability of Defect

            # Step 2: Identify Defect samples and get probabilities from mild_rancid_pipeline
            defect_indices = np.where(P_Defect > 0.5)[0]  # Consider samples where Defect is likely
            P_Mild = np.zeros_like(P_Pass)
            P_Rancid = np.zeros_like(P_Pass)

            P_Mild = self.mild_rancid_pipeline.predict_proba(X)[:, 0]
            P_Rancid = self.mild_rancid_pipeline.predict_proba(X)[:, 1]

            # Step 3: Compute final probabilities
            P_Mild = P_Defect * P_Mild  # P(Mild) = P(Defect) * P(Mild | Defect)
            P_Rancid = P_Defect * P_Rancid  # P(Rancid) = P(Defect) * P(Rancid | Defect)
            
            return np.column_stack((P_Pass, P_Mild, P_Rancid))

        def get_params(self, deep = False):
            return {
                "pass_defect_pipeline": self.pass_defect_pipeline,
                "mild_rancid_pipeline": self.mild_rancid_pipeline,
                "mapping" : self.mapping
            }

        def set_params(self, params):
            for param, value in params.items():
                self.__setattr__(param, value)

    def get_pipeline(self):
        return self.PipelineWrapper(self.pass_defect_pipeline, self.mild_rancid_pipeline, self.mapping)
from utilities import get_models_dict, IND_COLS
from balanced_data_ensemble import BalancedDataEnsemble
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_processing.imputation import get_imputer, fill_cat_null_values
from feature_selection.feature_selection import get_feature_selection
import pandas as pd

class ABXModel:
    """
    The ABXModel object represents a general ML model for predicting ABX appropriateness, capturing the pre-training steps.
    """
    def __init__(self, model_name, features_choice, K, param_dict, imp, norm, k=5, data_ensemble=False):
        # Pre-processing parameters
        self.features_choice = features_choice
        self.K = K
        self.normalizer = MinMaxScaler() if norm == 'min_max' else StandardScaler()
        self.imputer = get_imputer(imp, k)
        self.isolation_forest = IsolationForest(random_state=42)

        # Model parameters
        self.model_name = model_name
        self.param_dict = param_dict
        self.clf = get_models_dict([model_name], param_dict, get_funcs=True)[model_name]
        self.clf = BalancedDataEnsemble(self.clf) if data_ensemble else self.clf()
        self.selected_features = []


    def fit(self, X_train, y_train):
        """ Train model """
        print(f"Train {self.model_name}.\nTraining set size: {len(X_train)}")

        cat_cols = list(X_train.columns[X_train.dtypes == 'bool']) # TODO deal with this - maybe change type to categorical?
        cont_cols = [col for col in X_train.columns if col not in cat_cols]

        # TODO add filteration of null features and so on? check original func
        # Standardization
        X_train = self.normalizer.fit_transform(X_train[cont_cols])

        # Data imputation
        X_train = fill_cat_null_values(X_train, cat_cols)
        X_train[cont_cols] = self.imputer.fit_transform(X_train[cont_cols])

        # Getting Isolation Forest Feature (anomaly score)
        self.isolation_forest.fit(X_train)
        X_train.loc[:, 'IsolationForest'] = self.isolation_forest.predict(X_train)

        # Feature selection
        # TODO add correlation filteration (+stats?)
        self.selected_features = get_feature_selection(X_train, y_train, self.model_name, self.features_choice, self.K, self.param_dict)
        X_train = X_train[self.selected_features]

        # Train model
        self.clf.fit(X_train, y_train)


    def evaluation(self, X_test, y_test): # TODO update according to changes in fit function
        """ Predict and evaluate model """
        print(f"Evaluate {self.model_name}.\nTest set size: {len(X_test)}")

        cat_cols = list(X_test.columns[X_test.dtypes == 'bool'])
        cont_cols = [col for col in X_test.columns if col not in cat_cols]

        # Standardization
        X_test = self.normalizer.transform(X_test[cont_cols])

        # Data imputation
        X_test = fill_cat_null_values(X_test, cat_cols)
        X_test[cont_cols] = self.imputer.transform(X_test[cont_cols])

        # Getting Isolation Forest Feature (anomaly score)
        X_test.loc[:, 'IsolationForest'] = self.isolation_forest.predict(X_test)

        # Feature selection
        X_test = X_test[self.selected_features]

        # Predict
        probs = self.clf.predict_proba(X_test)[:, 1]
        model_results = {self.model_name: probs, 'target': y_test}
        res_df = pd.DataFrame.from_dict(model_results)
        return res_df
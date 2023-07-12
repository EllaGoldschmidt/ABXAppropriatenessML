from utilities import get_models_dict, filter_cols_by_null_percentage
from balanced_data_ensemble import BalancedDataEnsemble
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_processing.imputation import get_imputer, fill_cat_null_values
from feature_selection.feature_selection import get_feature_selection
from feature_selection.correlation_filteration import filter_correlated_features_from_stats
import pandas as pd


class ABXModel:
    """
    The ABXModel object represents a general ML model for predicting ABX appropriateness, capturing the pre-training steps.
    """
    def __init__(self, model_name, features_choice, K, param_dict, imp, norm, k=5, data_ensemble=False, null_thresh=0.7, corr_threshold=0.7, n_keep=1):
        # Pre-processing parameters
        self.features_choice = features_choice
        self.K = K
        self.corr_threshold = corr_threshold
        self.n_keep = n_keep
        self.normalizer = MinMaxScaler() if norm == 'min_max' else StandardScaler()
        self.imputer = get_imputer(imp, k)
        self.isolation_forest = IsolationForest(random_state=42)
        self.null_threshold = null_thresh

        # Model parameters
        self.model_name = model_name
        self.param_dict = param_dict
        self.clf = get_models_dict([model_name], param_dict, get_funcs=True)[model_name]
        self.clf = BalancedDataEnsemble(self.clf) if data_ensemble else self.clf()
        self.cont_cols = []
        self.cat_cols = []
        self.selected_features = []

    def fit(self, X_train, y_train):
        """ Train model """
        print(f"Train {self.model_name}.\nTraining set size: {len(X_train)}")

        cat_cols = list(X_train.columns[X_train.dtypes == 'bool']) # TODO deal with this - maybe change type to "category"?
        cont_cols = [col for col in X_train.columns if col not in cat_cols]

        # filtering Features
        cont_cols = filter_cols_by_null_percentage(X_train, cont_cols, self.null_threshold)
        var_threshold = VarianceThreshold(threshold=0.005)
        var_threshold.fit(X_train[cont_cols])
        cont_cols = X_train[cont_cols].loc[:, var_threshold.get_support()].columns.tolist()
        X_train = X_train[sorted(cont_cols + cat_cols)]
        self.cont_cols, self.cat_cols = cont_cols, cat_cols  # Saving for evaluation

        # Standardization
        X_train[cont_cols] = self.normalizer.fit_transform(X_train[cont_cols])

        # Data imputation
        X_train = fill_cat_null_values(X_train, cat_cols)
        X_train[cont_cols] = self.imputer.fit_transform(X_train[cont_cols])

        # Getting Isolation Forest Feature (anomaly score)
        self.isolation_forest.fit(X_train)
        X_train.loc[:, 'IsolationForest'] = self.isolation_forest.predict(X_train)

        # Redundant features removal + feature selection
        filtered_cols = filter_correlated_features_from_stats(X_train, cont_cols, cat_cols, self.corr_threshold, self.n_keep)
        self.selected_features = get_feature_selection(X_train[filtered_cols], y_train, self.model_name, self.features_choice, self.K, self.param_dict)
        X_train = X_train[self.selected_features]

        # Train model
        self.clf.fit(X_train, y_train)

    def evaluation(self, X_test, y_test):
        """ Predict and evaluate model """
        print(f"Evaluate {self.model_name}.\nTest set size: {len(X_test)}")

        # adding cols from train that are missing in test
        missing_cols = list(set([col for col in self.cont_cols + self.cat_cols if col not in X_test.columns]))
        X_test = X_test.reindex(columns=X_test.columns.tolist() + missing_cols)
        X_test = X_test[sorted(self.cat_cols + self.cont_cols)]

        # Standardization
        X_test[self.cont_cols] = self.normalizer.transform(X_test[self.cont_cols])

        # Data imputation
        X_test = fill_cat_null_values(X_test, self.cat_cols)
        X_test[self.cont_cols] = self.imputer.transform(X_test[self.cont_cols])

        # Getting Isolation Forest Feature (anomaly score)
        X_test.loc[:, 'IsolationForest'] = self.isolation_forest.predict(X_test)

        # Feature selection
        X_test = X_test[self.selected_features]

        # Predict
        probs = self.clf.predict_proba(X_test)[:, 1]
        model_results = {self.model_name: probs, 'target': y_test}
        res_df = pd.DataFrame.from_dict(model_results)
        return res_df
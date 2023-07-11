import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import xgboost as xgb
from lightgbm import LGBMClassifier


IND_COLS = ("identifier", "subject_id", "hadm_id")
ID_COL = "identifier"
LABEL = 'target'


def parse_col_name(name):
    name = re.sub("\s\([0-9]+\)", "", name.replace("_", " "))
    name = re.sub(r'[^A-Za-z0-9_]+', '', name.replace(" ", "_"))
    return name

def get_org_col(col):
    """Return the name of the raw measurement that the col was created from"""
    if "_days_" in col:
        return re.search(".+(?=_all_days_)", col).group() if "_all_days" in col else re.search(".+(?=_[2345]_days_)", col).group()
    if "_over_time" in col:
        return re.search("(?<=count_).+(?=_over_time)", col).group()
    else:
        return col.replace("_unique_percentage", "").replace("time_from_last_","").replace("_12_hours", "").replace("_median_diff", "").replace("_coef_ratio", "").replace("_is_imputed", "").replace("_after_before_BC_ratio", "")


def get_models_dict(models, param_dict, default_iter_num, get_funcs=False):
    param_dict = {} if param_dict is None else param_dict
    rf = (lambda: RandomForestClassifier(random_state=42, **param_dict["Random Forest"])) if "Random Forest" in param_dict else (lambda: RandomForestClassifier(random_state=42))
    knn = (lambda: KNeighborsClassifier(**param_dict["KNN"])) if "KNN" in param_dict else (lambda: KNeighborsClassifier())
    adb = (lambda: AdaBoostClassifier(random_state=42, **param_dict["Adaboost"])) if "Adaboost" in param_dict else (lambda: AdaBoostClassifier(random_state=42))
    lr = (lambda: LogisticRegression(random_state=42, **param_dict["Logistic Regression"])) if "Logistic Regression" in param_dict else (lambda: LogisticRegression(random_state=42, max_iter=default_iter_num))
    svm = (lambda: SVC(probability=True, random_state=42, kernel='linear', **param_dict["SVM"])) if "SVM" in param_dict else (lambda: SVC(probability=True, random_state=42, kernel='linear'))
    sgd = (lambda: SGDClassifier(**param_dict["SGD"], random_state=42)) if "SGD" in param_dict else (lambda: SGDClassifier(loss="log", random_state=42))
    xgb_model = (lambda: xgb.XGBClassifier(objective="binary:logistic", eval_metric=['auc', 'aucpr'], random_state=42, booster='dart', rate_drop=0.1, **param_dict["xgboost"])) if "xgboost" in param_dict else (lambda: xgb.XGBClassifier(objective="binary:logistic", eval_metric=['auc', 'aucpr'], booster='dart', rate_drop=0.1, random_state=42))
    grad = (lambda: GradientBoostingClassifier(random_state=42, **param_dict["GradientBoosting"])) if "GradientBoosting" in param_dict else (lambda: GradientBoostingClassifier(random_state=42))
    lgbm = (lambda: LGBMClassifier(random_state=42, **param_dict["LightGBM"])) if "LightGBM" in param_dict else (lambda: LGBMClassifier(random_state=42))


    models_dict = {"Random Forest": rf, "KNN": knn, "Adaboost": adb, "Logistic Regression": lr, "SVM": svm, "SGD":sgd,
                   "xgboost": xgb_model, "GradientBoosting":grad, "LightGBM": lgbm}
    models_dict = {key: v for (key, v) in models_dict.items() if key in models} if models else models_dict

    if not get_funcs:
        models_dict = {key: v() for key, v in models_dict.items()}
    return models_dict


def split_train_test_by_indices(df, train_ids, cols, id_col=ID_COL, ind_cols=IND_COLS, label_col=LABEL):
    train = df[df[id_col].isin(train_ids)].set_index(ind_cols)
    test = df[~df[id_col].isin(train_ids)].set_index(ind_cols)
    X_train, y_train = train[cols], train[label_col]
    X_test, y_test = test[cols], test[label_col]
    return X_train, y_train, X_test, y_test


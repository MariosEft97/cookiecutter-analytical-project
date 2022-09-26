# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier


### FEATURE SELECTION FUNCTION ###
def feature_selection(df: pd.DataFrame, identifier: list, target: str, method: str, **kwargs) -> pd.DataFrame:

    """
    The function performs feature selection.
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        target (str): target feature
        identifier (list): identifier features of the dataset
        method (str): method of feature selection (RFE: Recursive Feature Ellimination, Boruta)
        
        **kwargs:
            REF:
                estimator (str): estimator that assigns weights to the features (accepts: tree, logistic, svc, knn, bagging, forest, adaboost) 
                selected_features_number (int): number of features to select, if None (default) selects half of the features

    
    Returns:
        feature_selection_df (pd.DataFrame): data structure with the only the selected features

    """
    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(method, str):
        error_message = "method must be specified as a string\noptions: RFE (Recursive Feature Ellimination), Boruta"
        raise TypeError(error_message)

    else:

        # split data set into predictor features and target feature
        X = df.drop(columns=[target, identifier[0]])
        y = df[target]

        if method == "RFE":
            estimator = str(list(kwargs.values())[0])
            selected_features_number = list(kwargs.values())[1]

            # estimators
            tree = DecisionTreeClassifier()
            logistic = LogisticRegression()
            svc = SVC()
            knn = KNeighborsClassifier()
            bagging = BaggingClassifier()
            forest = RandomForestClassifier()
            adaboost = AdaBoostClassifier()

            # estimators' dictionary
            estimators = {"tree": tree, "logistic": logistic, "svc": svc, "knn": knn, "bagging": bagging, "forest": forest, "adaboost": adaboost}

            rfe = RFE(estimator=estimators[estimator], n_features_to_select=selected_features_number)
            rfe.fit(X, y)
            X_new = rfe.transform(X)
            selected_features = rfe.get_feature_names_out(X.columns)
            X_new_df = pd.DataFrame(X_new, columns=selected_features)
            feature_selection_df = pd.concat([df[identifier[0]].reset_index(drop=True), X_new_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            return feature_selection_df



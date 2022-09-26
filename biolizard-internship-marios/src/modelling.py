# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.tree import DecisionTreeClassifier


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
                estimator (str): estimator that assigns weights to the features
                selected_features_number (int): number of features to select, if None (default) selects half of the features

    
    Returns:
        selected_df (pd.DataFrame): data structure with the only the selected features

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

            # estimators' dictionary
            estimators = {"tree": tree}

            rfe = RFE(estimator=estimators[estimator], n_features_to_select=selected_features_number)
            X_new = rfe.fit_transform(X, y)

            return X_new



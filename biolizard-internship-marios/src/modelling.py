# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
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

        # estimators
        tree = DecisionTreeClassifier()
        logistic = LogisticRegression()
        svc = SVC()
        linearSVC = LinearSVC()
        knn = KNeighborsClassifier()
        bagging = BaggingClassifier()
        forest = RandomForestClassifier()
        adaboost = AdaBoostClassifier()

        # estimators' dictionary
        estimators = {"Tree": tree, "Logistic": logistic, "SVC": svc, "linearSVC": linearSVC, "KNN": knn, "Bagging": bagging, "Forest": forest, "AdaBoost": adaboost}

        if method == "RFE":
            # define RFE hyperparameters
            estimator = str(list(kwargs.values())[0])
            selected_features_number = list(kwargs.values())[1]

            rfe = RFE(estimator=estimators[estimator], n_features_to_select=selected_features_number)
            rfe.fit(X, y)
            X_new = rfe.transform(X)
            selected_features = rfe.get_feature_names_out(X.columns)
            X_new_df = pd.DataFrame(X_new, columns=selected_features)
            feature_selection_df = pd.concat([df[identifier[0]].reset_index(drop=True), X_new_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            return feature_selection_df

        elif method == "Boruta":
            # BorutaPy accepts only numpy arrays
            X_array = X.values
            y_array = y.values

            # define BorutaPy hyperparameters
            estimator = str(list(kwargs.values())[0])
            random_state = list(kwargs.values())[1]

            boruta = BorutaPy(estimator=estimators[estimator], n_estimators="auto", random_state=random_state)
            boruta.fit(X_array, y_array)
            X_new = boruta.transform(X_array)
            selected_features = X.loc[:, boruta.support_].columns
            X_new_df = pd.DataFrame(X_new, columns=selected_features)
            feature_selection_df = pd.concat([df[identifier[0]].reset_index(drop=True), X_new_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            return feature_selection_df
        
        elif method == "L1":
            # estimators
            logistic = LogisticRegression(penalty="l1", solver="saga")
            linearSVC = LinearSVC(penalty="l1", loss="squared_hinge", dual=False)

            # estimators' dictionary
            estimators = {"Logistic": logistic, "linearSVC": linearSVC}

            # define L1-based selection model hyperparameters
            estimator = str(list(kwargs.values())[0])

            l1 = SelectFromModel(estimator=estimators[estimator])
            l1.fit(X, y)
            X_new = l1.transform(X)
            selected_features = X.loc[:, l1.get_support()].columns
            X_new_df = pd.DataFrame(X_new, columns=selected_features)
            feature_selection_df = pd.concat([df[identifier[0]].reset_index(drop=True), X_new_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            return feature_selection_df


            


        



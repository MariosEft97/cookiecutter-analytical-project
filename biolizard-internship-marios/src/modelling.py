# LOAD PACKAGES
import sys
from turtle import color
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
from IPython.display import display
import joblib
import re
from tabulate import tabulate
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.feature_selection import RFE, SelectFromModel
from boruta import BorutaPy
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from yellowbrick.classifier import ROCAUC
from typing import Union, Any
from sklearn.inspection import permutation_importance
import shap
# from BorutaShap import BorutaShap
from tpot import TPOTClassifier

### GLOBAL VARIABLES ###
# define the class of sklearn classifiers
ClassifierModel = Union[LogisticRegression, SVC, LinearSVC, SGDClassifier, KNeighborsClassifier, GaussianNB, DecisionTreeClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, XGBClassifier]

### FEATURE SELECTION FUNCTION ###
def feature_selection(df: pd.DataFrame, identifier: list, target: str, method: str, **kwargs) -> pd.DataFrame:

    """
    The function performs feature selection.
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        target (str): target feature
        identifier (list): identifier features of the dataset
        method (str): method of feature selection (RFE: Recursive Feature Ellimination, Boruta, L1)
        
        **kwargs:
            REF:
                estimator (str): estimator that assigns weights to the features (accepts: Tree, Logistic, linearSVC, Forest, AdaBoost) 
                selected_features_number (int): number of features to select, if None (default) selects half of the features
            Boruta:
                estimator (str): estimator that assigns weights to the features (accepts: Forest)
                random_state (int): determines random number generation (default=None)


    
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
        error_message = "method must be specified as a string\noptions: RFE (Recursive Feature Ellimination), Boruta, L1"
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


        print(f"Initial dataset dimensions:")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        print("\n")
        print(f"Dataset dimensions after feature selection:")
        print(f"Rows: {feature_selection_df.shape[0]}")
        print(f"Columns: {feature_selection_df.shape[1]}")

        return feature_selection_df

### BINARY CLASSIFICATION FUNCTION ###
def binary_classification(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, k_fold: int=10, metric: str="accuracy", save_cv_results: bool=True) -> ClassifierModel:
    
    '''
    The function fits different binary classification models, performs hyperparameter tuning and returns the best model.

    Parameters:
        train_df (Pandas DataFrame): data structure with loaded data (train sample)
        test_df (Pandas DataFrame): data structure with loaded data (test sample)
        identifier (list): identifier features of the dataset
        target (str): target feature
        metric (str): metric to be used for the hyperparameter tuning (accuracy, recall, precision, default=accuracy)
        k_fold (int): number of k-folds for cross-validation (default=10)
        save_cv_results (bool): whether the hyperparameter tuning results should be saved (True or False, default=True)
            
    Returns:
        classifier (ClassifierModel): best performing sklearn classifier model
    '''

    if not isinstance(train_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(test_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(k_fold, int):
        error_message = "k_fold must be specified as an integer value"
        raise TypeError(error_message)
    
    elif not isinstance(metric, str):
        error_message = "metric must be specified as a string\noptions: accuracy, recall or presicion"
        raise TypeError(error_message)
    
    elif not isinstance(save_cv_results, bool):
        error_message = "save_cv_results must be specified as a boolean value (True or False)"
        raise TypeError(error_message)

    else:

        # encode the target variables
        lb = LabelBinarizer()
        encoded_train_target = lb.fit_transform(train_df[target])
        encoded_test_target = lb.transform(test_df[target])

        # create train set
        X_train = train_df.drop(columns=[identifier[0], target])
        y_train = pd.DataFrame(encoded_train_target, columns=[target])

        # create test set
        X_test = test_df[X_train.columns]
        y_test = pd.DataFrame(encoded_test_target, columns=[target])

        # data structure with classifiers
        models = []
        models.append(['LogisticRegression', LogisticRegression(random_state=0)])
        models.append(['SVM', SVC(random_state=0)])
        models.append(['LinearSVM', LinearSVC(random_state=0)])
        models.append(['SGD', SGDClassifier(random_state=0)])
        models.append(['K-NearestNeigbors', KNeighborsClassifier()])
        models.append(['GaussianNB', GaussianNB()])
        models.append(['DecisionTree', DecisionTreeClassifier(random_state=0)])
        models.append(['Bagging', BaggingClassifier(random_state=0)])
        models.append(['RandomForest', RandomForestClassifier(random_state=0)])
        models.append(['AdaBoost', AdaBoostClassifier(random_state=0)])
        models.append(['GradientBoosting', GradientBoostingClassifier(random_state=0)])
        models.append(['XGBoost', XGBClassifier(random_state=0)])
        # models.append(['MLP', MLPClassifier(random_state=0)])

        # check baseline performance of classifiers
        # data structure to append results for each classifier
        lst_1 = []

        for m in range(len(models)):
            lst_2 = []
            model = models[m][1]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=sorted(train_df[target].unique()), columns=sorted(train_df[target].unique()))
            
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            lst_2.append(models[m][0])
            lst_2.append(cm_df.iloc[0][0])
            lst_2.append(cm_df.iloc[0][1])
            lst_2.append(cm_df.iloc[1][0])
            lst_2.append(cm_df.iloc[1][1])
            lst_2.append(accuracy)
            lst_2.append(recall)
            lst_2.append(precision)
            lst_2.append(f1)
            lst_1.append(lst_2)
        
        baseline_performance_df = pd.DataFrame(lst_1, columns=['Algorithm','TN', 'FN', 'FP', 'TP', 'Accuracy','Recall', 'Precision', 'F1'])
        print('\nBaseline Performance of Classifiers:')
        print(tabulate(round(baseline_performance_df, 3), headers='keys', tablefmt='psql'))
        
        # define hyperparameter search space
        search_space = [
            (LogisticRegression(), [{'penalty':['l1', 'l2', 'elasticnet', 'none'], 'class_weight':['balanced', None], 'solver':["saga"], 'random_state':[0]}]),
            (SVC(), [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'class_weight':['balanced', None], 'probability': [True], 'random_state':[0]}]),
            (LinearSVC(), [{'penalty':['l1', 'l2'], 'class_weight':['balanced', None], 'random_state':[0]}]),
            (SGDClassifier(), [{'loss': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty':['l1', 'l2', 'elasticnet'], 'learning_rate':['optimal', 'constant', 'invscaling'], 'random_state':[0]}]),
            (KNeighborsClassifier(), [{'n_neighbors':[5, 10, 15], 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]),
            (GaussianNB(), [{'var_smoothing':[1e-8, 1e-9, 1e-10]}]),
            (DecisionTreeClassifier(), [{'criterion':['gini', 'entropy', 'log_loss'], 'splitter': ['best', 'random'], 'class_weight':['balanced', None], 'random_state':[0]}]),
            (BaggingClassifier(), [{'n_estimators':[5, 10, 15],  'random_state':[0]}]),
            (RandomForestClassifier(), [{'n_estimators':[50, 100, 150], 'criterion':['gini', 'entropy', 'log_loss'], 'class_weight':['balanced', None], 'random_state':[0]}]),
            (AdaBoostClassifier(), [{'n_estimators':[25, 50, 75], 'learning_rate':[0.5, 1.0, 1.5], 'algorithm':['SAMME', 'SAMME.R'], 'random_state':[0]}]),
            (GradientBoostingClassifier(), [{'learning_rate':[0.05, 0.1, 0.15], "loss":['log_loss','deviance', 'exponential'], 'n_estimators':[50, 100, 150], 'criterion':['friedman_mse', 'squared_error', 'mse'], 'random_state':[0]}]),
            (XGBClassifier(), [{'learning_rate':[0.1, 0.3, 0.5], 'n_estimators':[50, 100, 150], 'sampling_method':['uniform', 'subsample', 'gradient_based'], 'lambda':[0, 1, 2], 'alpha':[0, 1, 2], 'random_state':[0]}])
            ]
    
        # perform hyperparameter tuning using k-fold cross-validation
        model_names = []
        scores = []
        hyperparameters = []
        model_score = {}
        model_hyperparameters = {}

        for j, k in search_space:
            
            grid = GridSearchCV(estimator=j, param_grid=k, scoring=metric, cv=k_fold)
            grid.fit(X_train, y_train)
            
            optimal_score = grid.best_score_
            optimal_hypeparameters = grid.best_params_
            first_bracket_position = re.search("\(", str(j)).start()

            model_names.append(str(j)[0:first_bracket_position])
            scores.append(round(optimal_score*100, 3))
            hyperparameters.append(str(optimal_hypeparameters))
            model_score.update({str(j)[0:first_bracket_position]: optimal_score})
            model_hyperparameters.update({str(j)[0:first_bracket_position]: optimal_hypeparameters})
        
        tuned_performance_df = pd.DataFrame({"algortithm": model_names, metric: scores, "hyperparameters": hyperparameters})
        print('\nPerformance of Tuned Classifiers:')
        print(tabulate(round(tuned_performance_df, 3), headers='keys', tablefmt='psql'))
       
       # save hyperparameter tuning results
        if save_cv_results == True:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            filename = "grid_search_results_"+str(k_fold)+"foldcv_"+str(dt_string)+".pkl"
            joblib.dump(grid, filename)
        
        # fit the best performing model       
        tuned_models_hyperparameters = {
            "LogisticRegression": LogisticRegression(**model_hyperparameters["LogisticRegression"]),
            "SVC": SVC(**model_hyperparameters["SVC"]),
            "LinearSVC": LinearSVC(**model_hyperparameters["LinearSVC"]),
            "KNeighborsClassifier": KNeighborsClassifier(**model_hyperparameters["KNeighborsClassifier"]),
            "GaussianNB": GaussianNB(**model_hyperparameters["GaussianNB"]),
            "DecisionTreeClassifier": DecisionTreeClassifier(**model_hyperparameters["DecisionTreeClassifier"]),
            "BaggingClassifier": BaggingClassifier(**model_hyperparameters["BaggingClassifier"]),
            "RandomForestClassifier": RandomForestClassifier(**model_hyperparameters["RandomForestClassifier"]),
            "AdaBoostClassifier": AdaBoostClassifier(**model_hyperparameters["AdaBoostClassifier"]),
            "GradientBoostingClassifier": GradientBoostingClassifier(**model_hyperparameters["GradientBoostingClassifier"]),
            "XGBClassifier": XGBClassifier(**model_hyperparameters["XGBClassifier"])
            }

        best_performing_model = max(model_score, key=model_score.get)
        # bpm_optimal_metric = model_score[best_performing_model]
        # bpm_hyperparameters = model_hyperparameters[best_performing_model]

        classifier = tuned_models_hyperparameters[best_performing_model]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:,1]
        
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Roc Curve
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        # Precision-Recall curve
        precision_, recall_, _ = precision_recall_curve(y_test, y_prob)     
        
        # classification metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nTest set performance of best performing model ({best_performing_model}) on the train set:")
        bpm_results = pd.DataFrame({"Model": [best_performing_model], "Accuracy": [accuracy], "Recall": [recall], "Precision": [precision], "F1 score": [f1], "ROC AUC": [roc_auc_score(y_test, y_prob)]})
        print(tabulate(round(bpm_results, 3), headers='keys', tablefmt='psql'))

        # results = {"accuracy":[accuracy],"recall":[recall], "precision":[precision], "f1":[f1] "AUC":[roc_auc_score(y_test, y_prob)]}
  
        # Visualizing Confusion Matrix
        group_names = ["True Negative","False Positive","False Negative","True Positive"]
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cm_df, annot=labels, fmt="", cmap="YlGnBu", cbar=False)
        plt.title(label = "Confusion Matrix")
        plt.show()

        # Visualizing ROC Curve
        sns.set_theme(style = 'white')
        # plt.figure(figsize = (8, 8))
        plt.plot(false_positive_rate, true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
        plt.axis('tight')
        plt.title(label = "ROC curve")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        # Visualizing Precision-Recall Curve
        plt.plot(recall_, precision_, color = '#b01717')
        plt.axis('tight')
        plt.title(label = "Precision-Recall curve")
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()

        return classifier

### MULTICLASS CLASSIFICATION FUNCTION ###
def multiclass_classification(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, k_fold: int=10, metric: str="accuracy", save_cv_results: bool=True) -> ClassifierModel:
    
    '''
    The function fits different multi-class classification models, performs hyperparameter tuning and returns the best model.

    Parameters:
        train_df (Pandas DataFrame): data structure with loaded data (train sample)
        test_df (Pandas DataFrame): data structure with loaded data (test sample)
        identifier (list): identifier features of the dataset
        target (str): target feature
        metric (str): metric to be used for the hyperparameter tuning (accuracy, recall, precision, default=accuracy)
        k_fold (int): number of k-folds for cross-validation (default=10)
        save_cv_results (bool): whether the hyperparameter tuning results should be saved (True or False, default=True)
    
    Returns:
        classifier (ClassifierModel): best performing sklearn classifier model
    '''

    if not isinstance(train_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(test_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(k_fold, int):
        error_message = "k_fold must be specified as an integer value"
        raise TypeError(error_message)
    
    elif not isinstance(metric, str):
        error_message = "metric must be specified as a string\noptions: accuracy, recall or presicion"
        raise TypeError(error_message)
    
    elif not isinstance(save_cv_results, bool):
        error_message = "save_cv_results must be specified as a boolean value (True or False)"
        raise TypeError(error_message)

    else:

        # encode the target variables
        lb = LabelEncoder()
        encoded_train_target = lb.fit_transform(train_df[target])
        encoded_test_target = lb.transform(test_df[target])

        # create train set
        X_train = train_df.drop(columns=[identifier[0], target])
        y_train = pd.DataFrame(encoded_train_target, columns=[target])

        # create test set
        X_test = test_df[X_train.columns]
        y_test = pd.DataFrame(encoded_test_target, columns=[target])

        # data structure with classifiers
        models = []
        models.append(['LogisticRegression', LogisticRegression(random_state=0)])
        models.append(['SVM', SVC(random_state=0)])
        models.append(['LinearSVM', LinearSVC(random_state=0)])
        models.append(['SGD', SGDClassifier(random_state=0)])
        models.append(['K-NearestNeigbors', KNeighborsClassifier()])
        models.append(['GaussianNB', GaussianNB()])
        models.append(['DecisionTree', DecisionTreeClassifier(random_state=0)])
        models.append(['Bagging', BaggingClassifier(random_state=0)])
        models.append(['RandomForest', RandomForestClassifier(random_state=0)])
        models.append(['AdaBoost', AdaBoostClassifier(random_state=0)])
        models.append(['GradientBoosting', GradientBoostingClassifier(random_state=0)])
        models.append(['XGBoost', XGBClassifier(random_state=0)])
        # models.append(['MLP', MLPClassifier(random_state=0)])

        # check baseline performance of classifiers
        
        # data structure to append results for each classifier
        lst_1 = []

        for m in range(len(models)):
            lst_2 = []
            model = models[m][1]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=sorted(train_df[target].unique()), columns=sorted(train_df[target].unique()))
            
            recall = recall_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            kappa = cohen_kappa_score(y_test, y_pred)

            lst_2.append(models[m][0])
            lst_2.append(accuracy)
            lst_2.append(recall)
            lst_2.append(precision)
            lst_2.append(f1)
            lst_2.append(kappa)
            lst_1.append(lst_2)
        
        baseline_performance_df = pd.DataFrame(lst_1, columns=['Algorithm', 'Accuracy', 'Recall', 'Precision', 'F1', "Cohens's kappa"])
        print('\nBaseline Performance of Classifiers:')
        print(tabulate(round(baseline_performance_df, 3), headers='keys', tablefmt='psql'))

        # define hyperparameter search space
        search_space = [
            (LogisticRegression(), [{'penalty':['l1', 'l2', 'elasticnet', 'none'], 'class_weight':['balanced', None], 'solver':["saga"], 'random_state':[0]}]),
            (SVC(), [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'class_weight':['balanced', None], 'probability': [True], 'random_state':[0]}]),
            (LinearSVC(), [{'penalty':['l1', 'l2'], 'class_weight':['balanced', None], 'random_state':[0]}]),
            (SGDClassifier(), [{'loss': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty':['l1', 'l2', 'elasticnet'], 'learning_rate':['optimal', 'constant', 'invscaling'], 'random_state':[0]}]),
            (KNeighborsClassifier(), [{'n_neighbors':[5, 10, 15], 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]),
            (GaussianNB(), [{'var_smoothing':[1e-8, 1e-9, 1e-10]}]),
            (DecisionTreeClassifier(), [{'criterion':['gini', 'entropy', 'log_loss'], 'splitter': ['best', 'random'], 'class_weight':['balanced', None], 'random_state':[0]}]),
            (BaggingClassifier(), [{'n_estimators':[5, 10, 15],  'random_state':[0]}]),
            (RandomForestClassifier(), [{'n_estimators':[50, 100, 150], 'criterion':['gini', 'entropy', 'log_loss'], 'class_weight':['balanced', None], 'random_state':[0]}]),
            (AdaBoostClassifier(), [{'n_estimators':[25, 50, 75], 'learning_rate':[0.5, 1.0, 1.5], 'algorithm':['SAMME', 'SAMME.R'], 'random_state':[0]}]),
            (GradientBoostingClassifier(), [{'learning_rate':[0.05, 0.1, 0.15], "loss":['log_loss','deviance', 'exponential'], 'n_estimators':[50, 100, 150], 'criterion':['friedman_mse', 'squared_error', 'mse'], 'random_state':[0]}]),
            (XGBClassifier(), [{'learning_rate':[0.1, 0.3, 0.5], 'n_estimators':[50, 100, 150], 'sampling_method':['uniform', 'subsample', 'gradient_based'], 'lambda':[0, 1, 2], 'alpha':[0, 1, 2], 'random_state':[0]}])
            ]
        
        # removed MLP because it takes too much time and has less predictive accuracy than some of the rest of the models
        # (MLPClassifier(), [{'hidden_layer_sizes': [(50,), (100,), (150,)], 'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver':['lbfgs', 'sgd', 'adam'], 'alpha': [0.0001, 0.0002, 0.0003], 'learning_rate':['optimal', 'constant', 'invscaling'], 'learning_rate_init': [0.0005, 0.001, 0.002], 'random_state':[0]}])
    
        # perform hyperparameter tuning using k-fold cross-validation
        model_names = []
        scores = []
        hyperparameters = []
        model_score = {}
        model_hyperparameters = {}

        for j, k in search_space:
            
            grid = GridSearchCV(estimator=j, param_grid=k, scoring=metric, cv=k_fold)
            grid.fit(X_train, y_train)
            
            optimal_score = grid.best_score_
            optimal_hypeparameters = grid.best_params_
            first_bracket_position = re.search("\(", str(j)).start()

            model_names.append(str(j)[0:first_bracket_position])
            scores.append(round(optimal_score*100, 3))
            hyperparameters.append(str(optimal_hypeparameters))
            model_score.update({str(j)[0:first_bracket_position]: optimal_score})
            model_hyperparameters.update({str(j)[0:first_bracket_position]: optimal_hypeparameters})
        
        tuned_performance_df = pd.DataFrame({"algortithm": model_names, metric: scores, "hyperparameters": hyperparameters})
        print('\nPerformance of Tuned Classifiers:')
        print(tabulate(round(tuned_performance_df, 3), headers='keys', tablefmt='psql'))
        
        # save hyperparameter tuning results
        if save_cv_results == True:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
            filename = "grid_search_results_"+str(k_fold)+"foldcv_"+str(dt_string)+".pkl"
            joblib.dump(grid, filename)
        

        # fit the best performing model       
        tuned_models_hyperparameters = {
            "LogisticRegression": LogisticRegression(**model_hyperparameters["LogisticRegression"]),
            "SVC": SVC(**model_hyperparameters["SVC"]),
            "LinearSVC": LinearSVC(**model_hyperparameters["LinearSVC"]),
            "KNeighborsClassifier": KNeighborsClassifier(**model_hyperparameters["KNeighborsClassifier"]),
            "GaussianNB": GaussianNB(**model_hyperparameters["GaussianNB"]),
            "DecisionTreeClassifier": DecisionTreeClassifier(**model_hyperparameters["DecisionTreeClassifier"]),
            "BaggingClassifier": BaggingClassifier(**model_hyperparameters["BaggingClassifier"]),
            "RandomForestClassifier": RandomForestClassifier(**model_hyperparameters["RandomForestClassifier"]),
            "AdaBoostClassifier": AdaBoostClassifier(**model_hyperparameters["AdaBoostClassifier"]),
            "GradientBoostingClassifier": GradientBoostingClassifier(**model_hyperparameters["GradientBoostingClassifier"]),
            "XGBClassifier": XGBClassifier(**model_hyperparameters["XGBClassifier"])
            }

        best_performing_model = max(model_score, key=model_score.get)
        # bpm_optimal_metric = model_score[best_performing_model]
        # bpm_hyperparameters = model_hyperparameters[best_performing_model]

        classifier = tuned_models_hyperparameters[best_performing_model]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:,1]
        
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # classification metrics
        recall = recall_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        kappa = cohen_kappa_score(y_test, y_pred)

        print(f"\nTest set performance of best performing model ({best_performing_model}) on the train set:")
        bpm_results = pd.DataFrame({"Model": [best_performing_model], "Accuracy": [accuracy], "Recall": [recall], "Precision": [precision], "F1 score": [f1], "Cohen's kappa": [kappa]})
        print(tabulate(round(bpm_results, 3), headers='keys', tablefmt='psql'))

        # results = {"accuracy":[accuracy],"recall":[recall], "precision":[precision], "f1":[f1] "AUC":[roc_auc_score(y_test, y_prob)]}

        # Visualizing Confusion Matrix
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(len(train_df[target].unique()), len(train_df[target].unique()))
        sns.heatmap(cm_df, annot=labels, fmt="", cmap="YlGnBu", cbar=False)
        plt.title(label = "Confusion Matrix")
        plt.show()

        # Plot ROC curves
        roc_visualizer = ROCAUC(
            tuned_models_hyperparameters[best_performing_model], 
            encoder={i:sorted(train_df[target].unique())[i] for i in range(len(train_df[target].unique()))})
        
        roc_visualizer.fit(X_train, y_train)
        roc_visualizer.score(X_test, y_test)
        roc_visualizer.show()

        return classifier
            
### FEATURE IMPORTANCE FUNCTION ###
def feature_importance(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, classifier: ClassifierModel, top_features_to_view: int, use_boruta_shap: bool, **kwargs) -> None:
    
    '''
    The function displays the feature importance of the given classifier.

    Parameters:
        train_df (Pandas DataFrame): data structure with loaded data (train sample)
        test_df (Pandas DataFrame): data structure with loaded data (test sample)
        identifier (list): identifier features of the dataset
        target (str): target feature
        classifier (ClassifierModel): sklearn classifier model
        top_features_to_view (int): number of most important features to view
        use_boruta_shap (bool): compute featute importance using Boruta-Shap module (True or False)

        **kwrags:
            Boruta-Shap:

    
    Returns:
        None
    '''
        
    if not isinstance(train_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(test_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    # elif not isinstance(classifier, ClassifierModel):
    #     error_message = "classifier must be a ClassifierModel"
    #     raise TypeError(error_message)

    elif not isinstance(top_features_to_view, int):
        error_message = "top_features_to_view must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(use_boruta_shap, bool):
        error_message = "use_boruta_shap must be specified as a boolean value"
        raise TypeError(error_message)
    
    else:

        # encode the target variables
        lb = LabelEncoder()
        encoded_train_target = lb.fit_transform(train_df[target])
        encoded_test_target = lb.transform(test_df[target])

        # create train set
        X_train = train_df.drop(columns=[identifier[0], target])
        y_train = pd.DataFrame(encoded_train_target, columns=[target])

        # create test set
        X_test = test_df[X_train.columns]
        y_test = pd.DataFrame(encoded_test_target, columns=[target])

        first_bracket_position = re.search("\(", str(classifier)).start()
        model_name = str(classifier)[0:first_bracket_position]

        features = X_train.columns

        if use_boruta_shap == False:

            def feature_importance_plot(title, ytitle):
                feature_importance_dict = {features[i]: importances[i] for i in range(len(features))}
                feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                sorted_features = [x[0] for x in feature_importance_sorted]
                sorted_importances = [x[1] for x in feature_importance_sorted]

                if top_features_to_view > len(sorted_features):
                    print(f"The number of specified features ({top_features_to_view}) is greater than the number of available ones ({len(sorted_features)}).")
                    print(f"Please specify a number smaller than or equal to the number of available features ({len(sorted_features)}).")
                
                elif top_features_to_view == len(sorted_features):
                    report = pd.DataFrame({"Feature": features, title: importances, "Sorted Features (descending)": sorted_features, "Sorted "+title+" (descending)": sorted_importances})
                    print(tabulate(round(report, 3), headers='keys', tablefmt='psql'))

                    # plot feature importance
                    sns.set_theme(style = 'darkgrid')
                    if len(features) > 20:
                        plt.bar([x for x in range(len(features))], importances)
                    else:
                        plt.bar([x for x in features], importances)
                        if len(sorted_features[0]) > 3:
                            plt.xticks(rotation=90)

                    plt.title(f"{model_name} {title}")
                    plt.ylabel(ytitle)
                    plt.xlabel("Features")
                    plt.show()

                else:
                    report = pd.DataFrame({"Top " + str(top_features_to_view) + " Features": sorted_features[0:top_features_to_view], title: sorted_importances[0:top_features_to_view]})
                    print(tabulate(round(report, 3), headers='keys', tablefmt='psql'))

                    # plot feature importance
                    sns.set_theme(style = 'darkgrid')
                    if len(sorted_features[0:top_features_to_view]) > 20:
                        plt.bar([x for x in range(len(sorted_features[0:top_features_to_view]))], sorted_importances[0:top_features_to_view])
                    else:
                        plt.bar([x for x in sorted_features[0:top_features_to_view]], sorted_importances[0:top_features_to_view])
                        if len(sorted_features[0]) > 3:
                            plt.xticks(rotation=90)
                    
                    plt.title(f"{model_name} {title}")
                    plt.ylabel(ytitle)
                    plt.xlabel(f"Top {top_features_to_view} Features")
                    plt.show()
            
            if model_name == "LogisticRegression" or model_name == "SGDClassifier" or model_name == "LinearSVC":
                # https://machinelearningmastery.com/calculate-feature-importance-with-python/
                # https://stackoverflow.com/questions/66574982/how-can-we-interpret-feature-importances-for-stochastic-gradient-descent-classif

                importances = classifier.coef_[0]
                feature_importance_plot(title="Feature Coefficients", ytitle="Coefficients values")

            elif model_name == "DecisionTreeClassifier" or model_name == "RandomForestClassifier" or model_name == "AdaBoostClassifier" or model_name == "GradientBoostingClassifier" or model_name == "XGBClassifier":
                # https://machinelearningmastery.com/calculate-feature-importance-with-python/

                importances = classifier.feature_importances_
                feature_importance_plot(title="Feature Importance", ytitle="Mean decrease in impurity")

            elif model_name == "BaggingClassifier":
                # https://stackoverflow.com/questions/44333573/feature-importances-bagging-scikit-learn

                importances = np.mean([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
                feature_importance_plot(title="Feature Importance", ytitle="Mean decrease in impurity")

            elif model_name == "GaussianNB" or model_name == "SVC" or model_name == "KNeighborsClassifier":
                # https://stackoverflow.com/questions/62933365/how-to-get-the-feature-importance-in-gaussian-naive-bayes
                
                feature_importance = permutation_importance(classifier, X_test, y_test, scoring="accuracy")
                importances = feature_importance.importances_mean    
                feature_importance_plot(title="Feature Permutation Importance", ytitle="Mean accuracy decrease")
        
        elif use_boruta_shap == True:
            print("BorutaShap package is not available through conda...")

            # fature_selector = BorutaShap(model=classifier, importance_measure="shap", classification=True)

            # feature_selector.fit(X=X_train, y=y_train, n_trials=100, sample=False, train_or_test = 'test', normalize=True, verbose=True)

            # # Returns Boxplot of features
            # feature_selector.plot(which_features='all')

            # # Returns a subset of the original data with the selected features
            # subset = feature_selector.Subset()

        return None

### SHAP VALUE ANALYSIS FUNCTION ###
def shap_value_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, classifier: ClassifierModel) -> None:

    '''
    The function uses SHAP (SHapley Additive exPlanations) to increase transparency and interpretability of the machine learning model used.

    Parameters:
        train_df (Pandas DataFrame): data structure with loaded data (train sample)
        test_df (Pandas DataFrame): data structure with loaded data (test sample)
        identifier (list): identifier features of the dataset
        target (str): target feature
        classifier (ClassifierModel): sklearn classifier model

    Returns:
        None
    '''
        
    if not isinstance(train_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(test_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    # elif not isinstance(classifier, ClassifierModel):
    #     error_message = "classifier must be a ClassifierModel"
    #     raise TypeError(error_message)
    
    else:
        
        # encode the target variables
        lb = LabelEncoder()
        encoded_train_target = lb.fit_transform(train_df[target])
        encoded_test_target = lb.transform(test_df[target])

        # create train set
        X_train = train_df.drop(columns=[identifier[0], target])
        y_train = pd.DataFrame(encoded_train_target, columns=[target])

        # create test set
        X_test = test_df[X_train.columns]
        y_test = pd.DataFrame(encoded_test_target, columns=[target])

        first_bracket_position = re.search("\(", str(classifier)).start()
        model_name = str(classifier)[0:first_bracket_position]

        y_pred = classifier.predict(X_test)

        features = X_train.columns

        sns.set_theme(style = 'whitegrid')

        if model_name == "LogisticRegression" or model_name == "SGDClassifier" or model_name == "LinearSVC" or model_name == "BaggingClassifier" or model_name == "GaussianNB" or model_name == "SVC" or model_name == "KNeighborsClassifier":
            print(f"No SHAP technique available for {model_name} model.")
        elif model_name == "DecisionTreeClassifier" or model_name == "RandomForestClassifier" or model_name == "AdaBoostClassifier" or model_name == "GradientBoostingClassifier" or model_name == "XGBClassifier":
            explainer = shap.Explainer(classifier.predict, X_test)
            shap_values = explainer(X_test)

            fig1 = plt.figure()
            ax0 = fig1.add_subplot(141).set_title("Bar Plot")
            shap.plots.bar(shap_values, show = False)
            ax1 = fig1.add_subplot(142).set_title("SHAP values")
            shap.plots.bar(shap_values[0], show = False)
            ax2 = fig1.add_subplot(143).set_title("SHAP base values")
            shap.plots.bar(shap_values[1], show = False)
            ax3 = fig1.add_subplot(144).set_title("SHAP data")
            shap.plots.bar(shap_values[2], show = False)
            plt.gcf().set_size_inches(40,10)
            plt.tight_layout()
            plt.show()

            fig2 = plt.figure()
            ax0 = fig2.add_subplot(131).set_title("Beeswarm")
            shap.plots.beeswarm(shap_values, show = False)
            ax1 = fig2.add_subplot(132).set_title("Summary plot")
            shap.summary_plot(shap_values, show = False)
            ax2 = fig2.add_subplot(133).set_title("Violin plot")
            shap.summary_plot(shap_values, plot_type='violin', show=False)
            plt.gcf().set_size_inches(18,6)
            plt.tight_layout()
            plt.show()

            fig3 = plt.figure()
            ax0 = fig3.add_subplot(131)
            shap.plots.waterfall(shap_values[0], show = False)
            plt.title("SHAP values")
            ax1 = fig3.add_subplot(132)
            shap.plots.waterfall(shap_values[1], show = False)
            plt.title("SHAP base values")
            ax2 = fig3.add_subplot(133)
            shap.plots.waterfall(shap_values[2], show = False)
            plt.title("SHAP data")
            plt.gcf().set_size_inches(18,6)
            plt.tight_layout()
            plt.show()

            # for feature in features:
            #     shap.plots.scatter(shap_values[:,feature], color=shap_values)
                    
        return None

### EVENT CHART FUNCTION ###
def event_chart(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, classifier: ClassifierModel) -> None:
    
    '''
    The function plots a chart event.

    Parameters:
        train_df (Pandas DataFrame): data structure with loaded data (train sample)
        test_df (Pandas DataFrame): data structure with loaded data (test sample)
        identifier (list): identifier features of the dataset
        target (str): target feature
        classifier (ClassifierModel): sklearn classifier model

    Returns:
        None
    '''
        
    if not isinstance(train_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(test_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    # elif not isinstance(classifier, ClassifierModel):
    #     error_message = "classifier must be a ClassifierModel"
    #     raise TypeError(error_message)
    
    else:
        
        # encode the target variables
        lb = LabelEncoder()
        encoded_train_target = lb.fit_transform(train_df[target])
        encoded_test_target = lb.transform(test_df[target])

        # create train set
        X_train = train_df.drop(columns=[identifier[0], target])
        y_train = pd.DataFrame(encoded_train_target, columns=[target])

        # create test set
        X_test = test_df[X_train.columns]
        y_test = pd.DataFrame(encoded_test_target, columns=[target])

        first_bracket_position = re.search("\(", str(classifier)).start()
        model_name = str(classifier)[0:first_bracket_position]

        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:,1]

        features = X_train.columns

        # print(len(y_prob))
        # print(len(y_pred))
        # print(len(y_test))
        # print(len(test_df[identifier[0]]))

        df = pd.DataFrame({"Sample": test_df[identifier[0]], "Model Score": y_prob, "Model Prediction": y_pred, "True Class": list(test_df[target])})
        df.sort_values(by="Model Score", ascending=True, inplace=True, ignore_index=True)

        # display(df)

        # # Create traces
        # fig = go.Figure()
        
        # fig.add_trace(go.Scatter(x=[i for i in range(len(df["Sample"]))], y=round(df["Model Score"], 3), mode='lines+markers', name='Model Score'))

        fig = px.scatter(df, x=[i for i in range(len(df["Sample"]))], y=round(df["Model Score"], 3), color="True Class")

        fig.update_layout(
            
            title=f'Event Chart (Model: {model_name})',
            
            xaxis = dict(
                title="Samples",
                tickmode = 'array',
                tickvals = [i for i in range(len(df["Sample"]))],
                ticktext = [str(i) for i in list(df["Sample"])]
            ),
            
            yaxis = dict(
                title="Model Score"
            )
        )
        
        fig.show()

        return None

### AUTO CLASSIFICATION FUNCTION ###
def tpot_classification(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, target_type: str,  generations: int, population_size: int, max_time_mins: int) -> None:
    '''
    The function permorms the modelling steps (fitting, hyperparameter tuning) automatically.

    Parameters:
        train_df (Pandas DataFrame): data structure with loaded data (train sample)
        test_df (Pandas DataFrame): data structure with loaded data (test sample)
        identifier (list): identifier features of the dataset
        target (str): target feature
        target_type (str): target feature type (binary or multiclass)
        generations (int): number of iterations to run the pipeline optimization process (integer or None, default=None)
        population_size (int): number of individuals to retain in the GP population every generation (integer or None, default=None)
        max_time_mins (int): many minutes TPOT has to optimize the pipeline (integer or None)

    Returns:
        None
    '''

    if not isinstance(train_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(test_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(target_type, str):
        error_message = "target_type must be specified as a string\noptions: binary or multiclass"
        raise TypeError(error_message)

    # elif not isinstance(generations, int):
    #     error_message = "generations must be specified as an integer value or None"
    #     raise TypeError(error_message)
    
    # elif not isinstance(population_size, int):
    #     error_message = "population_size must be specified as an integer value or None"
    #     raise TypeError(error_message)
    
    # elif not isinstance(max_time_mins, int):
    #     error_message = "max_time_mins must be specified as an integer value or None"
    #     raise TypeError(error_message)

    else:   
        
        # encode the target variables
        lb = LabelEncoder()
        encoded_train_target = lb.fit_transform(train_df[target])
        encoded_test_target = lb.transform(test_df[target])

        # create train set
        X_train = train_df.drop(columns=[identifier[0], target])
        y_train = pd.DataFrame(encoded_train_target, columns=["target"])

        # create test set
        X_test = test_df[X_train.columns]
        y_test = pd.DataFrame(encoded_test_target, columns=["target"])

        tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, max_time_mins=max_time_mins)

        def fit_tpot(tpot, X, y):
            tpot.fit(X, y)
            return tpot
        
        tpot = fit_tpot(tpot, X_train, y_train)

        best_pipeline = str(tpot.export())

        custom_best_pipeline = str()

        hastag_positions = [match.start() for match in re.finditer("#", best_pipeline)]
        newline_positions = [match.start() for match in re.finditer("\n", best_pipeline)]

        custom_best_pipeline += "def run_tpot_pipeline():\n\t"

        for index, character in enumerate(best_pipeline):

            if index < hastag_positions[0]:
                custom_best_pipeline += character

            elif index == hastag_positions[0]:
                custom_best_pipeline += "# NOTE: Make sure that the outcome column is labeled 'target' in the data file:\n"
                custom_best_pipeline += "training_features=X_train\n"
                custom_best_pipeline += "training_target=y_train\n"
                custom_best_pipeline += "testing_features=X_test\n"
                custom_best_pipeline += "testing_target=y_test\n"
                
            elif  hastag_positions[0] < index < hastag_positions[1]:
                pass

            elif index >= hastag_positions[1]:
                custom_best_pipeline += character
    
        # custom_best_pipeline += "probs=exported_pipeline.predict_proba(testing_features)"
        custom_best_pipeline += "\nreturn_me = [results]\n"

        loc = {}
        exec(custom_best_pipeline, locals(), loc)
        returns = loc["return_me"]

        y_pred = returns[0] 
        # y_prob = returns[1]

        # bracket_positions = [match.start() for match in re.finditer("\(", custom_best_pipeline)]
        # exported_pipeline_position = re.search("exported_pipeline = ", custom_best_pipeline).end()
        # for index in bracket_positions:
        #     if index > exported_pipeline_position:     
        #         correct_bracket_position = index
        #         break
        # model_name = custom_best_pipeline[exported_pipeline_position:correct_bracket_position]

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=sorted(train_df[target].unique()), columns=sorted(train_df[target].unique()))

        # # Roc Curve
        # false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)
        # roc_auc = auc(false_positive_rate, true_positive_rate)

        # # Precision-Recall curve
        # precision_, recall_, _ = precision_recall_curve(y_test, y_prob)     
        
        if target_type == "binary":

            # classification metrics
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"\nTest set performance of best performing model on the train set:")
            bpm_results = pd.DataFrame({"Accuracy": [accuracy], "Recall": [recall], "Precision": [precision], "F1 score": [f1]})
            print(tabulate(round(bpm_results, 3), headers='keys', tablefmt='psql'))

            # results = {"accuracy":[accuracy],"recall":[recall], "precision":[precision], "f1":[f1] "AUC":[roc_auc_score(y_test, y_prob)]}
    
            # Visualizing Confusion Matrix
            group_names = ["True Negative","False Positive","False Negative","True Positive"]
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            sns.heatmap(cm_df, annot=labels, fmt="", cmap="YlGnBu", cbar=False)
            plt.title(label = "Confusion Matrix")
            plt.show()

            # # Visualizing ROC Curve
            # sns.set_theme(style = 'white')
            # # plt.figure(figsize = (8, 8))
            # plt.plot(false_positive_rate, true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
            # plt.legend(loc = 'lower right')
            # plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
            # plt.axis('tight')
            # plt.title(label = "ROC curve")
            # plt.ylabel('True Positive Rate')
            # plt.xlabel('False Positive Rate')
            # plt.show()

            # # Visualizing Precision-Recall Curve
            # plt.plot(recall_, precision_, color = '#b01717')
            # plt.axis('tight')
            # plt.title(label = "Precision-Recall curve")
            # plt.ylabel('Precision')
            # plt.xlabel('Recall')
            # plt.show()
        
        elif target_type == "multiclass":
            
            # classification metrics
            recall = recall_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            kappa = cohen_kappa_score(y_test, y_pred)

            print(f"\nTest set performance of best performing model on the train set:")
            bpm_results = pd.DataFrame({"Accuracy": [accuracy], "Recall": [recall], "Precision": [precision], "F1 score": [f1], "Cohen's kappa": [kappa]})
            print(tabulate(round(bpm_results, 3), headers='keys', tablefmt='psql'))

            # results = {"accuracy":[accuracy],"recall":[recall], "precision":[precision], "f1":[f1] "AUC":[roc_auc_score(y_test, y_prob)]}

            # Visualizing Confusion Matrix
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
            labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(len(train_df[target].unique()), len(train_df[target].unique()))
            sns.heatmap(cm_df, annot=labels, fmt="", cmap="YlGnBu", cbar=False)
            plt.title(label = "Confusion Matrix")
            plt.show()

        return None
        



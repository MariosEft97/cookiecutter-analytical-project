# LOAD PACKAGES
import sys
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
from yellowbrick.classifier import ROCAUC
from typing import Union, Any
from sklearn.inspection import permutation_importance

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

        print(f"\nPerformance of best performing model ({best_performing_model}) on the test set:")
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

        # X_train2 = train_df.drop(columns=[identifier[0], target])
        # y_train_initial = train_df[[target]]
        # y_train2 = label_binarize(y_train_initial, classes=range(len(train_df[target].unique())))

        # # create test set
        # X_test2 = test_df[X_train.columns]
        # y_test_initial = test_df[[target]]
        # y_test2 = label_binarize(y_test_initial, classes=range(len(train_df[target].unique())))

        # n_classes = y_test2.shape[1]

        # # Learn to predict each class against the other
        # classifier2 = OneVsRestClassifier(tuned_models_hyperparameters[best_performing_model])
        # y_score = classifier2.fit(X_train2, y_train2).decision_function(X_test2)

        # # Compute ROC curve and ROC area for each class
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test2.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(n_classes):
        #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # # Finally average it and compute AUC
        # mean_tpr /= n_classes

        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # classification metrics
        recall = recall_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        kappa = cohen_kappa_score(y_test, y_pred)

        print(f"\nPerformance of best performing model ({best_performing_model}) on the test set:")
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

        # lw = 2
        
        # # Plot all ROC curves
        # plt.figure()
        # plt.plot(
        #     fpr["micro"],
        #     tpr["micro"],
        #     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        #     color="deeppink",
        #     linestyle=":",
        #     linewidth=4,
        # )

        # plt.plot(
        #     fpr["macro"],
        #     tpr["macro"],
        #     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        #     color="navy",
        #     linestyle=":",
        #     linewidth=4,
        # )

        # colors = cycle(["aqua", "darkorange", "cornflowerblue", "yellow"])
        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(
        #         fpr[i],
        #         tpr[i],
        #         color=color,
        #         lw=lw,
        #         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        #     )

        # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title("Multiclass ROC curve")
        # plt.legend(loc="lower right")
        # plt.show()

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
    
        # plt.plot(recall_, precision_, color = '#b01717')
        # plt.axis('tight')
        # plt.title(label = "Precision-Recall curve")
        # plt.ylabel('Precision')
        # plt.xlabel('Recall')
        # plt.show()

        return classifier
            
### FEATURE IMPORTANCE FUNCTION ###
def feature_importance(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, classifier: ClassifierModel) -> None:
    
    '''
    The function displays the feature importance of the given classifier.

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

        features = X_train.columns

        def feature_importance_plot(title, ytitle):
            feature_importance_dict = {features[i]: importances[i] for i in range(len(features))}
            feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_features = [x[0] for x in feature_importance_sorted]
            sorted_importances = [x[1] for x in feature_importance_sorted]
            report = pd.DataFrame({"Feature": features, title: importances, "Sorted Features (descending)": sorted_features, "Sorted "+title+" (descending)": sorted_importances})
            print(tabulate(round(report, 3), headers='keys', tablefmt='psql'))

            # plot feature importance
            sns.set_theme(style = 'darkgrid')
            if len(features) > 20:
                plt.bar([x for x in range(len(features))], importances)
            else:
                plt.bar([x for x in features], importances)
            plt.title(f"{model_name} {title}")
            plt.ylabel(ytitle)
            plt.xlabel("Features")
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




        



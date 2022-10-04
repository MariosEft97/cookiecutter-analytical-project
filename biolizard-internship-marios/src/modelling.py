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
from sklearn.feature_selection import RFE, SelectFromModel
from boruta import BorutaPy
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import plotly.figure_factory as ff

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
def binary_classification(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, target: str, k_fold: int=10, metric: str="accuracy", save_cv_results: bool=True, interactive_visuals: bool=True) -> None:
    
    '''
    The function fits different binary classification models, performs hyperparameter tuning and returns the best model.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        identifier (list): identifier features of the dataset
        target (str): target feature
        metric (str): metric to be used for the hyperparameter tuning (accuracy, recall, precision, default=accuracy)
        k_fold (int): number of k-folds for cross-validation (default=10)
        save_cv_results (bool): whether the hyperparameter tuning results should be saved (True or False, default=True)
        interavtive_visuals (bool): if True plots are create with plotly else with seaborn and matplotlib (True or False, default=True)
    
    Returns:
        None
    '''

    if not isinstance(train_df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    if not isinstance(test_df, pd.DataFrame):
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
        encoded_test_target = lb.transform(test_df['Diagnosis'])

        # create train set
        X_train = train_df.drop(columns=[identifier[0], target])
        y_train = pd.DataFrame(encoded_train_target, columns=["Diagnosis"])

        # create test set
        X_test = test_df[X_train.columns]
        y_test = pd.DataFrame(encoded_test_target, columns=["Diagnosis"])

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

        for i, m in enumerate(range(len(models))):
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

            # print(f'{i+1}) {models[m][0]}:')
            # print('-'*(len(models[m][0])+5))
            # print("Confusion Matrix:")
            # print(cm_df)
            # print(f'Accuracy: {round(accuracy, 3)}')
            # print(f'Recall: {round(recall, 3)}')
            # print(f'Presicion: {round(precision, 3)}')
            # print('-'*100)

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
        # display(round(baseline_performance_df, 3))

        # define hyperparameter search space
        search_space = [
            (LogisticRegression(), [{'penalty':['l1', 'l2', 'elasticnet', 'none'], 'class_weight':['balanced', None], 'solver':["saga"], 'random_state':[0]}]),
            (SVC(), [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'class_weight':['balanced', None], 'random_state':[0]}]),
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

        for i, (j, k) in enumerate(search_space):
            
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
        # display(round(tuned_performance_df, 3))

            # print(f'{i+1}) {str(j)[0:first_bracket_position]}')
            # print('-'*(len(str(j)[0:first_bracket_position])+5))
            # print(f'Optimal Accuracy: {round(optimal_score*100, 3)}%')
            # print(f'Optimal Hyperparameters: {optimal_hypeparameters}')
            # print('-'*100)
        
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
        false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        # Precision-Recall curve
        precision_, recall_, orc_thresholds = precision_recall_curve(y_test, y_prob)     
        
        # classification metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nPerformance of best performing model ({best_performing_model}) on the test set:")
        # print(classification_report(y_test, y_pred))
        # print('\n')
        # print(f'Accuracy: {round(accuracy, 3)}')
        # print(f'Recall: {round(recall, 3)}')
        # print(f'Presicion: {round(precision, 3)}')
        # print(f'F1 score: {round(f1, 3)}')
        # print(f'ROC AUC: {round(roc_auc_score(y_test, y_prob), 3)}')
        # print('\n')
        bpm_results = pd.DataFrame({"Model": [best_performing_model], "Accuracy": [accuracy], "Recall": [recall], "Precision": [precision], "F1 score": [f1], "ROC AUC": [roc_auc_score(y_test, y_prob)]})
        print(tabulate(round(bpm_results, 3), headers='keys', tablefmt='psql'))

        # results = {"accuracy":[accuracy],"recall":[recall], "precision":[precision], "f1":[f1] "AUC":[roc_auc_score(y_test, y_prob)]}



        if interactive_visuals == False:
            
            # Visualizing Confusion Matrix
            group_names = ["True Negative","False Positive","False Negative","True Positive"]
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            sns.heatmap(cm_df, annot=labels, fmt="", cmap="YlGnBu", cbar=False)
            plt.title(label = "Confusion Matrix")
            plt.show()

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
        
            plt.plot(recall_, precision_, color = '#b01717')
            plt.axis('tight')
            plt.title(label = "Precision-Recall curve")
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.show()

        # elif interactive_visuals == True:
            # z = [
            #     [cm_df.iloc[0][0], cm_df.iloc[0][1]],
            #     [cm_df.iloc[1][0], cm_df.iloc[1][1]]
            #     ]

            # # visualize confusion matrix
            # x = train_df[target].unique()
            # y = train_df[target].unique()

            # # change each element of z to type string for annotations
            # z_text = [[str(y) for y in x] for x in cm]

            # # set up figure 
            # cm_fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

            # # add title
            # cm_fig.update_layout(title_text='Confusion matrix',
            #                 xaxis = dict(title='Predicted Class'),
            #                 yaxis = dict(title='True Class')
            #                 )

            # # add custom xaxis title
            # cm_fig.add_annotation(dict(font=dict(color="black",size=14),
            #                         x=0.5,
            #                         y=-0.15,
            #                         showarrow=False,
            #                         text="Predicted value",
            #                         xref="paper",
            #                         yref="paper"))

            # # add custom yaxis title
            # cm_fig.add_annotation(dict(font=dict(color="black",size=14),
            #                         x=-0.35,
            #                         y=0.5,
            #                         showarrow=False,
            #                         text="Real value",
            #                         textangle=-90,
            #                         xref="paper",
            #                         yref="paper"))

            # # adjust margins to make room for yaxis title
            # cm_fig.update_layout(margin=dict(t=50, l=200))

            # # add colorbar
            # cm_fig['data'][0]['showscale'] = True
            # cm_fig.show()


            # visualize roc curve

            # roc_fig = px.area(
            #     x=false_positive_rate, y=true_positive_rate,
            #     title=f'ROC Curve (AUC={auc(false_positive_rate, true_positive_rate):.3f})',
            #     labels=dict(x='False Positive Rate', y='True Positive Rate'),
            #     width=700, height=500
            #     )
            
            # roc_fig.add_shape(
            #     type='line', line=dict(dash='dash'),
            #     x0=0, x1=1, y0=0, y1=1
            # )

            # roc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
            # roc_fig.update_xaxes(constrain='domain')
            # roc_fig.show()

            # # visualize precision-recall curve

            # prc_fig = px.area(
            #     x=recall, y=precision,
            #     title=f'Precision-Recall Curve (AUC={auc(false_positive_rate, true_positive_rate):.3f})',
            #     labels=dict(x='Recall', y='Precision'),
            #     width=700, height=500
            # )
            # prc_fig.add_shape(
            #     type='line', line=dict(dash='dash'),
            #     x0=0, x1=1, y0=1, y1=0
            # )
            # prc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
            # prc_fig.update_xaxes(constrain='domain')

            # prc_fig.show()


            




        



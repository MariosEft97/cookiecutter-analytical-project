# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import plotly.express as px

def box_plot_v2(df: pd.DataFrame, features: list, categorical: list) -> None:
    
    '''
    The function creates an interactive boxplot.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        features (list): list of predictor features
        stratify_var (str): variable to stratify the data by (if stratify=True)
        group_var (str): variable to group the data by (if group=True)
        title (str): plot title
        xtitle (str): x axis title
        ytitle (str): y axis title
        widget_description (str): intercative widget description
        stratify (bool): display feature values stratified by the specified feature (True or False, default=False)
        group (bool): group feature values by the specified feature (True or False, default=False)
    
    Returns:
        None
    '''

    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(features, list):
        error_message = "features must be specified as a list of strings"
        raise TypeError(error_message)
    
    elif not isinstance(categorical, list):
        error_message = "categorical must be specified as a list of strings"
        raise TypeError(error_message)

    else:
        
        # define widgets
        bool_options = ["False", "True"]
        feature_dropdown = widgets.Dropdown(description="Select:", value=features[0], options=features)
        stratify_bool_dropdown = widgets.Dropdown(description="Stratify:", value=bool_options[0], options=bool_options)
        group_bool_dropdown = widgets.Dropdown(description="Group:", value=bool_options[0], options=bool_options)
        stratify_feature_dropdown = widgets.Dropdown(description="Stratify by:", value=categorical[0], options=categorical)
        group_feature_dropdown = widgets.Dropdown(description="Group by:", value=categorical[0], options=categorical)

        output = widgets.Output()

        def boxplot_creator(df, feature, stratify_bool, group_bool, stratify_feature, group_feature):
            
            # display boxplots per level of specified stratify feature
            if stratify_bool == "True":
                    
                    # display boxplots per level of specified stratify feature and divide into groups based on the levels of the group feature
                    if group_bool == "True":
                        fig = px.box(df, x=stratify_feature, y=feature, color=group_feature, boxmode="group", title=f"Boxplot of {feature}")
                        fig.show()
                    
                    # no grouping is done
                    elif group_bool == "False":
                        fig = px.box(df, x=stratify_feature, y=feature, boxmode="group", title=f"Boxplot of {feature}")
                        fig.show()
                    
            # no stratification is done
            elif stratify_bool == "False":
                    
                    # display boxplots divided into groups based on the levels of the specified group feature
                    if group_bool == "True":
                        # for each level of the group feature create a trace
                        fig = px.box(df, y=feature, color=group_feature, boxmode="group", title=f"Boxplot of {feature}")
                        fig.show()                    
                    
                    # no grouping is done
                    elif group_bool == "False":
                        fig = px.box(df, y=feature, boxmode="group", title=f"Boxplot of {feature}")
                        fig.show()
            
            return None
        
        def boxplot_filter(feature, stratify_bool, group_bool, stratify_feature, group_feature):

            output.clear_output()

            # display boxplots per level of specified stratify feature
            if stratify_bool == "True":
                with output:
                    display(stratify_feature_dropdown)
                        
                    # display boxplots per level of specified stratify feature and divide into groups based on the levels of the group feature
                    if group_bool == "True":
                        display(group_feature_dropdown)
                        boxplot_creator(df, feature, stratify_bool, group_bool, stratify_feature, group_feature)
                    
                    # no grouping is done
                    elif group_bool == "False":
                        boxplot_creator(df, feature, stratify_bool, group_bool, stratify_feature, group_feature)

                    output.clear_output()
                    
            # no stratification is done
            elif stratify_bool == "False":
                with output:
                    # display boxplots divided into groups based on the levels of the specified group feature
                    if group_bool == "True":
                        # for each level of the group feature create a trace
                        display(group_feature_dropdown)
                        boxplot_creator(df, feature, stratify_bool, group_bool, stratify_feature, group_feature)            
                    
                    # no grouping is done
                    elif group_bool == "False":
                        boxplot_creator(df, feature, stratify_bool, group_bool, stratify_feature, group_feature)
                    
                    output.clear_output()
            
            return None

        def feature_response(change):
            boxplot_filter(change.new, stratify_bool_dropdown.value, group_bool_dropdown.value, stratify_feature_dropdown.value, group_feature_dropdown.value)

        def stratify_bool_response(change):
            boxplot_filter(feature_dropdown.value, change.new, group_bool_dropdown.value, stratify_feature_dropdown.value, group_feature_dropdown.value)

        def group_bool_response(change):
            boxplot_filter(feature_dropdown.value, stratify_bool_dropdown.value, change.new, stratify_feature_dropdown.value, group_feature_dropdown.value)

        def stratify_feature_response(change):
            boxplot_filter(feature_dropdown.value, stratify_bool_dropdown.value, group_bool_dropdown.value, change.new, group_feature_dropdown.value)

        def group_feature_response(change):
            boxplot_filter(feature_dropdown.value, stratify_bool_dropdown.value,  group_bool_dropdown.value, stratify_feature_dropdown.value, change.new)

        feature_dropdown.observe(feature_response, names="value")
        stratify_bool_dropdown.observe(stratify_bool_response, names="value")
        group_bool_dropdown.observe(group_bool_response, names="value")
        stratify_feature_dropdown.observe(stratify_feature_response, names="value")
        group_feature_dropdown.observe(group_feature_response, names="value")

        # display widget and figure
        container = widgets.HBox([feature_dropdown, stratify_bool_dropdown, group_bool_dropdown])
        display(container)
        display(output)

        # initial plot
        with output:
            boxplot_creator(df, feature_dropdown.value, stratify_bool_dropdown.value, group_bool_dropdown.value, stratify_feature_dropdown.value, group_feature_dropdown.value)
                
### CLUSTERING PLOT CODE ###
# stage_names = ['A', 'B', 'C', 'D']
# stage_data = {stage: cluster_df[cluster_df["True_Labels"]==stage] for stage in stage_names}

# fig = go.Figure()

# for stage_name, stage_df in stage_data.items():

#     fig.add_trace(go.Scatter(
#         x=stage_df["Component_1"].values,
#         y=stage_df["Component_2"].values,
#         name=stage_name,
#         text=stage_df["ID_REF"],
#         hovertemplate =
#         'ID: %{text}'+
#         'Predicted Label: %{stage_df["Predicted_Labels"]}'+
#         'True Label: %{stage_df["True_Labels"]}'
#     ))

# fig.update_traces(
#     mode='markers',
#     marker={'size': df[marker_size_ref],
#     'sizemode':'diameter',
#     'sizeref': df[marker_size_ref].max()/50,
#     'opacity': 1,
#     'color': cluster_df["Predicted_Labels"],
#     'colorscale': "viridis"})

# fig = go.Figure(data=go.Scatter(
#         x=cluster_df["Component_1"].values,
#         y=cluster_df["Component_2"].values,
#         text=cluster_df["ID_REF"],
#         mode='markers',
#         marker=go.Marker(
#             size=df[marker_size_ref],
#             sizemode='diameter',
#             sizeref=df[marker_size_ref].max()/50,
#             opacity=1,
#             color=cluster_df["Predicted_Labels"],
#             colorscale="viridis"
#             )
#         )
#     )

# fig = go.Figure(data=go.Scatter3d(
#         x=cluster_df["Component_1"].values,
#         y=cluster_df["Component_2"].values,
#         z=cluster_df["Component_3"].values,
#         text=cluster_df["ID_REF"],
#         mode='markers',
#         marker=go.Marker(
#             size=df[marker_size_ref],
#             sizemode='diameter',
#             sizeref=df[marker_size_ref].max()/50,
#             opacity=1,
#             color=cluster_df["Predicted_Labels"],
#             colorscale="viridis"
#             )
#         )
#     )


### HISTOGRAM PLOT ###

# histogram

# gene_list = list(df.columns[9:])
# gene_slice = gene_list[0:10]
# feature_list = []
# for feature in gene_list:
#     feature_list.append((feature, feature))

# import plotly.express as px
# import ipywidgets as widgets

# def fig_creator(selection):
#     fig = px.histogram(train_df, x=selection, color="Dukes Stage", facet_col="Dukes Stage")
#     fig.show()

# widgets.interact(fig_creator, selection=widgets.Dropdown(options=feature_list, description='Select:'));



### BINARY CLASSIFICATION CODE ###

# import os

# import pandas as pd
# import numpy as np

# from tqdm import tqdm

# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB, BernoulliNB
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier

# from sklearn.metrics import *
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# import joblib
# import re
# from sklearn.preprocessing import LabelBinarizer

# lb = LabelBinarizer()
# encoded_train_target = lb.fit_transform(rfe_df['Diagnosis'])
# encoded_test_target = lb.transform(test_df['Diagnosis'])

# X_train = rfe_df.drop(columns=['ID', 'Diagnosis'])
# y_train = pd.DataFrame(encoded_train_target, columns=["Diagnosis"])

# X_test = test_df[X_train.columns]
# y_test = pd.DataFrame(encoded_test_target, columns=["Diagnosis"])

# label_dict = {"B": 0, "M": 1}

# X_train = rfe_df.drop(columns=['ID', 'Diagnosis'])
# y_train = rfe_df[['Diagnosis']].replace(label_dict)

# X_test = test_df[X_train.columns]
# y_test = test_df[['Diagnosis']].replace(label_dict)

# models = []
# models.append(['LogisticRegression', LogisticRegression(random_state=0)])
# models.append(['SVM', SVC(random_state=0)])
# models.append(['LinearSVM', LinearSVC(random_state=0)])
# models.append(['SGD', SGDClassifier(random_state=0)])
# models.append(['K-NearestNeigbors', KNeighborsClassifier()])
# models.append(['GaussianNB', GaussianNB()])
# models.append(['DecisionTree', DecisionTreeClassifier(random_state=0)])
# models.append(['Bagging', BaggingClassifier(random_state=0)])
# models.append(['RandomForest', RandomForestClassifier(random_state=0)])
# models.append(['AdaBoost', AdaBoostClassifier(random_state=0)])
# models.append(['GradientBoosting', GradientBoostingClassifier(random_state=0)])
# models.append(['XGBoost', XGBClassifier(random_state=0)])
# models.append(['MLP', MLPClassifier(random_state=0)])

# lst_1 = []

# for i, m in enumerate(tqdm(range(len(models)))):
#     lst_2 = []
#     model = models[m][1]
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     cm = confusion_matrix(y_test, y_pred)
#     cm_df = pd.DataFrame(cm, index=sorted(train_df["Diagnosis"].unique()), columns=sorted(train_df["Diagnosis"].unique()))
#     accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

#     recall = recall_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     print(f'{i}) {models[m][0]}:')
#     print('-'*(len(models[m][0])+5))
#     print("Confusion Matrix:")
#     print(cm_df)
#     print(f'Accuracy: {round(accuracy, 3)}')
#     print(f'Recall: {round(recall, 3)}')
#     print(f'Presicion: {round(precision, 3)}')
#     print('-'*100)
   
#     lst_2.append(models[m][0])
#     lst_2.append(cm_df.iloc[0][0])
#     lst_2.append(cm_df.iloc[0][1])
#     lst_2.append(cm_df.iloc[1][0])
#     lst_2.append(cm_df.iloc[1][1])
#     lst_2.append(accuracy)
#     lst_2.append(recall)
#     lst_2.append(precision)
#     lst_1.append(lst_2)

# metrics_df = pd.DataFrame(lst_1, columns=['Algorithm','TN', 'FN', 'FP', 'TP', 'Accuracy','Recall', 'Precision'])
# round(metrics_df, 3)

# models.append(['Logistic Regression', LogisticRegression(random_state=0)])
# models.append(['SVM', SVC(random_state=0)])
# models.append(['Linear SVM', LinearSVC(random_state=0)])
# models.append(['SGD', SGDClassifier(random_state=0)])
# models.append(['K-nearest Neigbors', KNeighborsClassifier()])
# models.append(['GaussianNB', GaussianNB()])
# models.append(['Decision Tree', DecisionTreeClassifier(random_state=0)])
# models.append(['Bagging', BaggingClassifier(random_state=0)])
# models.append(['Random Forest', RandomForestClassifier(random_state=0)])
# models.append(['AdaBoost', AdaBoostClassifier(random_state=0)])
# models.append(['XGBoost', GradientBoostingClassifier(random_state=0)])
# models.append(['MLP', MLPClassifier(random_state=0)])

# search_space = [
#     (LogisticRegression(), [{'penalty':['l1', 'l2', 'elasticnet', 'none'], 'class_weight':['balanced', None], 'solver':["saga"], 'random_state':[0]}]),
#     (SVC(), [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'class_weight':['balanced', None], 'random_state':[0]}]),
#     (LinearSVC(), [{'penalty':['l1', 'l2'], 'class_weight':['balanced', None], 'random_state':[0]}]),
#     (SGDClassifier(), [{'loss': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty':['l1', 'l2', 'elasticnet'], 'learning_rate':['optimal', 'constant', 'invscaling'], 'random_state':[0]}]),
#     (KNeighborsClassifier(), [{'n_neighbors':[5, 10, 15], 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]),
#     (GaussianNB(), [{'var_smoothing':[1e-8, 1e-9, 1e-10]}]),
#     (DecisionTreeClassifier(), [{'criterion':['gini', 'entropy', 'log_loss'], 'splitter': ['best', 'random'], 'class_weight':['balanced', None], 'random_state':[0]}]),
#     (BaggingClassifier(), [{'n_estimators':[5, 10, 15],  'random_state':[0]}]),
#     (RandomForestClassifier(), [{'n_estimators':[50, 100, 150], 'criterion':['gini', 'entropy', 'log_loss'], 'class_weight':['balanced', None], 'random_state':[0]}]),
#     (AdaBoostClassifier(), [{'n_estimators':[25, 50, 75], 'learning_rate':[0.5, 1.0, 1.5], 'algorithm':['SAMME', 'SAMME.R'], 'random_state':[0]}]),
#     (GradientBoostingClassifier(), [{'learning_rate':[0.05, 0.1, 0.15], "loss":['log_loss','deviance', 'exponential'], 'n_estimators':[50, 100, 150], 'criterion':['friedman_mse', 'squared_error', 'mse'], 'random_state':[0]}]),
#     (XGBClassifier(), [{'learning_rate':[0.1, 0.3, 0.5], 'n_estimators':[50, 100, 150], 'sampling_method':['uniform', 'subsample', 'gradient_based'], 'lambda':[0, 1, 2], 'alpha':[0, 1, 2], 'random_state':[0]}])
#     ]

# removed MLP from grid search because it takes too much time and has lower optimal accuracy than other classifiers
# (MLPClassifier(), [{'hidden_layer_sizes': [(50,), (100,), (150,)], 'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver':['lbfgs', 'sgd', 'adam'], 'alpha': [0.0001, 0.0002, 0.0003], 'learning_rate':['optimal', 'constant', 'invscaling'], 'learning_rate_init': [0.0005, 0.001, 0.002], 'random_state':[0]}])


# for i, (j, k) in enumerate(tqdm(search_space)):
#     grid = GridSearchCV(estimator=j, param_grid=k, scoring='accuracy', cv = 10)
#     grid.fit(X_train, y_train)
#     optimal_score = grid.best_score_
#     optimal_hypeparameters = grid.best_params_
#     first_bracket_position = re.search("\(", str(j)).start()
#     print(f'{i+1}) {str(j)[0:first_bracket_position]}')
#     print('-'*(len(str(j)[0:first_bracket_position])+5))
#     print(f'Optimal Accuracy: {round(optimal_score*100, 3)}%')
#     print(f'Optimal Hyperparameters: {optimal_hypeparameters}')
#     print('-'*100)

# classifier = RandomForestClassifier(**{'class_weight': None, 'criterion': 'entropy', 'n_estimators': 50, 'random_state': 0})
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# y_prob = classifier.predict_proba(X_test)[:,1]
# cm = confusion_matrix(y_test, y_pred)
 
# recall = recall_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)

# print(classification_report(y_test, y_pred))
# print('\n')
# print(f'Accuracy: {round(accuracy, 3)}')
# print(f'Recall: {round(recall, 3)}')
# print(f'Presicion: {round(precision, 3)}')
# print(f'ROC AUC: {round(roc_auc_score(y_test, y_prob), 3)}')
# print('\n')

# results = {"accuracy":[accuracy],"recall":[recall], "precision":[precision], "AUC":[roc_auc_score(y_test, y_prob)]}

# # Visualizing Confusion Matrix
# group_names = ["True Negative","False Positive","False Negative","True Positive"]
# group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
# group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
# labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cm_df, annot=labels, fmt="", cmap="YlGnBu")
# plt.title(label = "Confusion Matrix on test set", weight = "bold", fontsize=14)
# plt.show()

# # Roc Curve
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(false_positive_rate, true_positive_rate)

# sns.set_theme(style = 'white')
# # plt.figure(figsize = (8, 8))
# plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
# plt.axis('tight')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


### MULTICLASS CLASSIFICATION CODE ###

# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle

# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.metrics import roc_auc_score

# X_train = rfe_df.drop(columns=[df_id[0], "Dukes Stage"])
# y_train_initial = rfe_df[["Dukes Stage"]]
# y_train = label_binarize(y_train_initial, classes=range(len(rfe_df["Dukes Stage"].unique())))

# # create test set
# X_test = test_df[X_train.columns]
# y_test_initial = test_df[["Dukes Stage"]]
# y_test = label_binarize(y_test_initial, classes=range(len(rfe_df["Dukes Stage"].unique())))

# n_classes = y_test.shape[1]

# # Learn to predict each class against the other
# classifier = OneVsRestClassifier(
#     svm.SVC(kernel="linear", probability=True, random_state=0)
# )
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plt.figure()
# lw = 2
# plt.plot(
#     fpr[2],
#     tpr[2],
#     color="darkorange",
#     lw=lw,
#     label="ROC curve (area = %0.2f)" % roc_auc[2],
# )
# plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
# plt.legend(loc="lower right")
# plt.show()

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

# plt.plot([0, 1], [0, 1], "k--", lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Some extension of Receiver operating characteristic to multiclass")
# plt.legend(loc="lower right")
# plt.show()

### PLOTLY CLASSIFICATION VISUALS ###

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
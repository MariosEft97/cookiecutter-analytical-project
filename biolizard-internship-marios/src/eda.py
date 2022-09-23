# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"
from IPython.display import display
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from ipywidgets import widgets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.manifold import MDS, TSNE 
import sklearn.cluster as cluster
import umap
import warnings
warnings.filterwarnings('ignore')

### CORRELATIONS FUNCTION ###
def correlations(df: pd.DataFrame, type: str="pearson", printout: str="matrix") -> pd.DataFrame:
    
    """
    The function creates a correlation matrix/heatmap of the continuous features in the dataset.
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        type (str): type of correlations (pearson or spearman)
        printout (str): how correlations are displayed (matrix or heatmap)
    
    Returns:
        matrix (pd.DataFrame): data structure with the computed correlation matrix

    """
    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)

    elif not isinstance(type, str):
        error_message = "type must be specified as a string\noptions:\noptions: pearson or spearman"
        raise TypeError(error_message)
    
    elif not isinstance(printout, str):
        error_message = "printout must be specified as a string\noptions: matrix or heatmap"
        raise TypeError(error_message)

    else:
        
        # calculate correlation matrix
        matrix = df.corr(method=type)
        type_dict = {"pearson": "Pearson", "spearman": "Spearman"}

        # display correlation matrix
        if printout == "matrix":
            print(f"{type_dict[type]} Correlation Matrix:")
            display(round(matrix, 3))
        
        # display correlation heatmap
        elif printout == "heatmap":
            fig = px.imshow(round(matrix, 3), text_auto=True, color_continuous_scale="Viridis", title=f"{type_dict[type]} Correlation Heatmap of Continuous Features")
            fig.show()
    
        return matrix

### BOXPLOT FUNCTION ###
def box_plot(df: pd.DataFrame, features: list, stratify_var: str, group_var: str, title: str, xtitle: str, ytitle: str, widget_description: str, stratify: bool=False, group: bool=False) -> None:
    
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
    
    elif not isinstance(stratify_var, str):
        error_message = "stratify_var must be specified as a string"
        raise TypeError(error_message)

    elif not isinstance(group_var, str):
        error_message = "group_var must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(title, str):
        error_message = "title must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(xtitle, str):
        error_message = "xtitle must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(ytitle, str):
        error_message = "ytitle must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(widget_description, str):
        error_message = "widget_description must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(stratify, bool):
        error_message = "stratify must be specified as a boolean value (True or False)"
        raise TypeError(error_message)
    
    elif not isinstance(group, bool):
        error_message = "group must be specified as a boolean value (True or False)"
        raise TypeError(error_message)

    else:
        
        # define widget
        feature = widgets.Dropdown(description=widget_description, value=features[0], options=features)

        # display boxplots per level of specified stratify feature
        if stratify == True:
            # display boxplots per level of specified stratify feature and divide into groups based on the levels of the group feature
            if group == True:
                # for each level of the group feature create a trace
                traces = []
                for i, group in enumerate(df[group_var].unique()):
                    filter = df[group_var]==df[group_var].unique()[i]
                    df_filtered = df[filter]
                    trace = go.Box(x=df_filtered[stratify_var], y=df_filtered[feature.value], name=df[group_var].unique()[i], boxmean=True)
                    traces.append(trace)

                # create initial figure with traces
                g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
            
            # no grouping is done
            elif group == False:
                trace = go.Box(x=df[stratify_var], y=df[feature.value], name=feature.value, boxmean=True)
                g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

        # no stratification is done
        elif stratify == False:
            # display boxplots divided into groups based on the levels of the specified group feature
            if group == True:
                # for each level of the group feature create a trace
                traces = []
                for i, group in enumerate(df[group_var].unique()):
                    filter = df[group_var]==df[group_var].unique()[i]
                    df_filtered = df[filter]
                    trace = go.Box(y=df_filtered[feature.value], name=df[group_var].unique()[i], boxmean=True)
                    traces.append(trace)
                
                # create initial figure with traces
                g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

            # no grouping is done
            elif group == False:
                trace = go.Box(y=df[feature.value], name=feature.value, boxmean=True)
                g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
        
        # function to validate that feature selected on the widget belong to the specified ones
        def validate():
            if feature.value in features:
                return True
            else:
                return False
        
        # function to control the changes on the figure based on feature selection on the widget
        def response(change):
            if validate():
                if feature.value:
                    temp_df = df[feature.value]
            else:
                temp_df = df[features[0]]
            
            x1 = temp_df
            
            with g.batch_update():
                g.data[0].y = x1
                g.layout.barmode = 'overlay'
                g.layout.xaxis.title = xtitle
                g.layout.yaxis.title = ytitle
                g.layout.title = f"{title} {feature.value}"
        
        feature.observe(response, names="value")

        # display widget and figure
        container = widgets.HBox([feature])
        display(widgets.VBox([container, g]))

### MULTIPLE BOXPLOTS FUNCTION ###
def box_subplots(df: pd.DataFrame, features: list, stratify_var: str, columns: int, width: int, height: int, stratify: bool=False) -> None:
    
    '''
    The function creates multiple interactive boxplots on the same figure.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        features (list): predictor features
        stratify_var (str): variable to stratify the data by (if stratify=True)
        columns (int): number of columns of the figure
        stratify (bool): display feature values stratified by the specified feature (True or False, default=False)
            
    Returns:
        None
    '''
    
    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(features, list):
        error_message = "features must be specified as a list of strings"
        raise TypeError(error_message)
    
    elif not isinstance(stratify_var, str):
        error_message = "stratify_var must be specified as a string"
        raise TypeError(error_message)

    elif not isinstance(columns, int):
        error_message = "columns must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(width, int):
        error_message = "width must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(height, int):
        error_message = "height must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(stratify, bool):
        error_message = "stratify must be specified as a boolean value (True or False)"
        raise TypeError(error_message)

    else:

        # rows of the figure are calculated according to the number of specified features and number of columns
        rows = math.ceil(len(features)/columns)
        
        # title for each subplot
        titles = [feature for feature in features]
        
        # data structures for saving the type of each subplot
        outter_list = []
        inner_list = []
        
        for i in range(columns):
            inner_list.append({"type": "histogram"})
        
        for i in range(rows):
            outter_list.append(inner_list)

        # create figure with subplots
        fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple(titles), specs=outter_list)

        # create coordinates of each subplot
        coordinates = []
        for i in range(rows):
            for j in range(columns):
                coordinates.append([i+1, j+1])
        
        for (feature), (coordinate) in zip(features, coordinates):
            
            # create stratified subplots based on the specified feature
            if stratify==True:
                # add subplots to the figure
                fig.add_trace(go.Box(x=df[stratify_var], y=df[feature], name=feature, boxmean=True), row=coordinate[0], col=coordinate[1])
            
            # no stratification 
            elif stratify==False:
                # add subplots to the figure
                fig.add_trace(go.Box(y=df[feature], name=feature, boxmean=True), row=coordinate[0], col=coordinate[1])

        # add figure dimensions and display the legend
        fig.update_layout(height=rows*height, width=columns*width, showlegend=True)

        fig.show()

### HISTOGRAM FUNCTION ###
def hist_plot(df: pd.DataFrame, features: list, group_var: str, title: str, xtitle: str, ytitle: str, widget_description: str, group: bool=False) -> None:
    
    '''
    The function creates an interactive histogram plot.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        features (list): list of predictor features
        group_var (str): variable to group the data by (if group=Yes)
        title (str): plot title
        xtitle (str): x axis title
        ytitle (str): y axis title
        widget_description (str): widget description
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

    elif not isinstance(group_var, str):
        error_message = "group_var must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(title, str):
        error_message = "title must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(xtitle, str):
        error_message = "xtitle must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(ytitle, str):
        error_message = "ytitle must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(widget_description, str):
        error_message = "widget_description must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(group, bool):
        error_message = "group must be specified as a boolean value (True or False)"
        raise TypeError(error_message)

    else:
        
        # define widget 
        feature = widgets.Dropdown(description=widget_description, value=features[0], options=features)
        
        # display boxplots divided into groups based on the levels of the specified group feature
        if group == True:
            # for each level of the group feature create a trace
            traces = []
            for i, group in enumerate(df[group_var].unique()):
                filter = df[group_var]==df[group_var].unique()[i]
                df_filtered = df[filter]
                trace = go.Histogram(x=df_filtered[feature.value], name=df[group_var].unique()[i])
                traces.append(trace)

            # create initial figure with traces
            g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), barmode="group"))

        # no grouping is done
        elif group == False:
            trace = go.Histogram(x=df[feature.value], name=feature.value)
            g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), barmode="group"))
        
        # function to validate that feature selected on the widget belong to the specified ones
        def validate():
            if feature.value in features:
                return True
            else:
                return False
        
        # function to control the changes on the figure based on feature selection on the widget
        def response(change):
            if validate():
                if feature.value:
                    temp_df = df[feature.value]
            else:
                temp_df = df[features[0]]
            
            x1 = temp_df
            
            with g.batch_update():
                g.data[0].x = x1
                g.layout.xaxis.title = xtitle
                g.layout.yaxis.title = ytitle
                g.layout.title = f"{title} {feature.value}"
        
        feature.observe(response, names="value")

        # display widget and figure
        container = widgets.VBox([feature])
        display(widgets.VBox([container, g]))

### MULTIPLE HISTOGRAMS FUNCTION ###
def hist_subplots(df: pd.DataFrame, columns: int, width: int, height: int) -> None:
    
    '''
    The function creates multiple interactive histogram plots on the same figure.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        columns (int): number of columns of the figure
        width (int): figure width multiplied by column number
        height (int): figure height multiplied by column number
            
    Returns:
        None
    '''
    
    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)

    elif not isinstance(columns, int):
        error_message = "columns must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(width, int):
        error_message = "width must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(height, int):
        error_message = "height must be specified as an integer number"
        raise TypeError(error_message)

    else:
        
        # names of specified features
        features = df.columns

        # rows of the figure are calculated according to the number of specified features and number of columns
        rows = math.ceil(len(features)/columns)
        
        # title for each subplot
        titles = [feature for feature in features]
        
        # data structures for saving the type of each subplot
        outter_list = []
        inner_list = []
        
        for i in range(columns):
            inner_list.append({"type": "histogram"})
        
        for i in range(rows):
            outter_list.append(inner_list)

        # create figure with subplots
        fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple(titles), specs=outter_list)

        # create coordinates of each subplot
        coordinates = []
        for i in range(rows):
            for j in range(columns):
                coordinates.append([i+1, j+1])
        
        # add subplots to the figure
        for (feature), (coordinate) in zip(features, coordinates):
            fig.add_trace(go.Histogram(x=df[feature], name=feature), row=coordinate[0], col=coordinate[1])

        # add figure dimensions and display the legend
        fig.update_layout(height=rows*height, width=columns*width, showlegend=True)

        fig.show()

### DISTRIBUTION PLOT FUNCTION ###
def dist_plot(df: pd.DataFrame, title: str) -> None:
    
    '''
    The function creates an interactive distribution plot.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        title (str): plot title
    
    Returns:
        None
    '''

    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(title, str):
        error_message = "title must be specified as a string"
        raise TypeError(error_message)
    
    else:

        # list containing data for each feature
        hist_data = [df[gene] for gene in df.columns]

        # list containing the name of each feature
        group_labels = [gene for gene in df.columns]

        # create figure
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
        fig.update_layout(title_text=title)
        fig.show()

        return None

### MULTIPLE DISTPLOTS FUNCTION ###
def dist_subplots(df: pd.DataFrame, columns: int, width: int, height: int) -> None:
    
    '''
    The function creates multiple interactive distribution plots on the same figure.

    Source: https://stackoverflow.com/questions/58803324/plotly-how-to-combine-make-subplots-and-ff-create-distplot

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        columns (int): number of columns of the figure
        width (int): figure width multiplied by column number
        height (int): figure height multiplied by column number
            
    Returns:
        None
    '''

    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)

    elif not isinstance(columns, int):
        error_message = "columns must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(width, int):
        error_message = "width must be specified as an integer number"
        raise TypeError(error_message)
    
    elif not isinstance(height, int):
        error_message = "height must be specified as an integer number"
        raise TypeError(error_message)

    else:

        # names of specified features
        features = df.columns

        # rows of the figure are calculated according to the number of specified features and number of columns
        rows = math.ceil(len(features)/columns)
        
        # title for each subplot
        titles = [feature for feature in features]
        
        # data structures for saving the type of each subplot
        outter_list = []
        inner_list = []
        
        for i in range(columns):
            inner_list.append({"type": "histogram"})
        
        for i in range(rows):
            outter_list.append(inner_list)

        # create figure with subplots
        fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple(titles), specs=outter_list)

        # create distribution plots for each feature
        hist_data = [df[feature] for feature in features]
        group_labels = [feature for feature in features]
        dist_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

        # create coordinates of each subplot
        coordinates = []
        for i in range(rows):
            for j in range(columns):
                coordinates.append([i+1, j+1])
        
        # add distribution plots in each subplot on the figure
        for (coordinate), (trace) in zip(coordinates, dist_fig.select_traces()):
            fig.add_trace(trace, row=coordinate[0], col=coordinate[1])

        # add figure dimensions and display the legend
        fig.update_layout(height=rows*height, width=columns*width, showlegend=True)

        fig.show()

        return None

### PCA EXPLAINED VARIANCE PLOT FUNCTION ###
def pca_variance_plot(train_df: pd.DataFrame, identifier: list, categorical: list, continuous:list, target: str,) -> None:
    
    '''
    The function creates the explained variance Vs principal components plot.
    
    Parameters:
        train_df (Pandas DataFrame): data structure with train sample
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        target (str): target variable
    
    Returns:
        None
    '''
    if not isinstance(train_df, pd.DataFrame):
        error_message = "train_df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)
    
    elif not isinstance(categorical, list):
        error_message = "categorical must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(continuous, list):
        error_message = "continuous must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)

    else:
        
        # list of categorical features without target feature
        categorical_without_target = categorical.copy()
        categorical_without_target.remove(target)

        # remove target and ID feature
        X_train = train_df.drop(columns=[target, identifier[0]])

        binary_encoded_features = []
        onehot_encoded_features = []

        # encode categorical features
        for feature in categorical_without_target:
            if len(X_train[feature].unique()) == 2:
                lb = LabelBinarizer()
                encoded_feature = lb.fit_transform(X_train[feature])
                encoded_feature_df = pd.DataFrame(encoded_feature, columns=[feature])
                binary_encoded_features.append(encoded_feature_df)
            elif len(X_train[feature].unique()) > 2:
                ohe = OneHotEncoder()
                encoded_feature = ohe.fit_transform(X_train[[feature]]).toarray()
                encoded_feature_df = pd.DataFrame(encoded_feature, columns = [feature+"_"+X_train[feature].unique()[i] for i in range(len(X_train[feature].unique()))])
                onehot_encoded_features.append(encoded_feature_df)

        X_train = X_train.drop(columns=categorical_without_target)
        binary_encoded_features_df = pd.concat(binary_encoded_features, axis=1)
        onehot_encoded_features_df = pd.concat(onehot_encoded_features, axis=1)
        X_train = pd.concat([binary_encoded_features_df.reset_index(drop=True), onehot_encoded_features_df.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)

        # dataset into a numpy array
        X_train_continuous = X_train.values
        X_train_continuous = np.array(X_train_continuous)
        
        # scale dataset
        standard_scaler = StandardScaler()
        X_train_continuous_scaled = standard_scaler.fit_transform(X_train_continuous)

        # principal component analysis
        pca = PCA()
        pca.fit_transform(X_train_continuous_scaled)
               
        # explained variance per principal component
        explained_variance = list(np.cumsum(pca.explained_variance_ratio_))
        explained_variance.insert(0,0)
        explained_variance = [i * 100 for i in explained_variance]

        # plot of explained variance per principal component
        fig = go.Figure(data=go.Scatter(y=explained_variance))

        # plot options
        fig.update_layout(go.Layout(title='Percentage of Explained Variance per Principal Component',
                                xaxis=go.layout.XAxis(title='Number of PCs', range=[0, len(pca.components_)]),
                                yaxis=go.layout.YAxis(title='Explained Variance')
                                ))

        fig.show()

### DIMENSIONALITY REDUCTION FUNCTION ###
def dimensionality_reduction(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, categorical: list, continuous:list, target: str, method: str, plot_type: str, **kwargs) -> pd.DataFrame:
    
    '''
    The function performs dimensionality reduction techniques on the given data.

    Parameters:
        train_df (Pandas DataFrame): data structure with train sample
        test_df (Pandas DataFrame): data structure with test sample
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        target (str): target variable
        method (str): dimensionality reduction method (pca: PCA, mds: MDS, umap: UMAP, tsne: t-SNE)
        plot_type (str): type of plot to display (2d: 2-dimensional plot, 3d: 3-dimensional plot, multi: multi-dimensional plot)
        
        **kwargs:
            components (int): number of components to retain in the analysis (applies in PCA, MDS, t-SNE and UMAP)
            perplexity (float): number of nearest neighbors (5-50, default=30, applies in t-SNE)
            neighbors (int): controls how UMAP balances local versus global structure in the data (low/high values favor local/global structure, default=15)
            min_distance (float): controls how tightly UMAP is allowed to pack points together (0-0.99, default=0.1)
            metric (str): controls how distance is computed in UMAP (euclidean, manhattan, chebyshev, minkowski, mahalanobis, cosine, correlation)
    
    Returns:
        train_df_encoded (Pandas DataFrame): data structure with train sample before dimensionality reduction (categorical variables encoded)
        train_df_reduced (Pandas DataFrame): data structure with train sample after dimensionality reduction
        test_df_encoded (Pandas DataFrame): data structure with train sample before dimensionality reduction (categorical variables encoded)
        test_df_reduced (Pandas DataFrame): data structure with test sample after dimensionality reduction

    '''
    if not isinstance(train_df, pd.DataFrame):
        error_message = "train_df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(test_df, pd.DataFrame):
        error_message = "test_df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(categorical, list):
        error_message = "categorical must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(continuous, list):
        error_message = "continuous must be specified as a list of strings"
        raise TypeError(error_message)

    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    elif not isinstance(method, str):
        error_message = "method must be specified as a string\noptions: pca (PCA), mds (MDS), umap (UMAP), tsne (t-SNE)"
        raise TypeError(error_message)
    
    elif not isinstance(plot_type, str):
        error_message = "plot_type must be specified as a string\noptions: 2d (2-dimensional plot) 3d (3-dimensional plot) or multi (multi-dimensional plot)"
        raise TypeError(error_message)
    
    else:

        # list of categorical features without target feature
        categorical_without_target = categorical.copy()
        categorical_without_target.remove(target)

        # train set DataFrames for each type of feature (ID, categorical, continuous, target)
        X_train = train_df.drop(columns=[target, identifier[0]])
        X_train_df_identifier = train_df[identifier].reset_index(drop=True)
        X_train_df_categorical = train_df[categorical_without_target].reset_index(drop=True)
        y_train = train_df[target].reset_index(drop=True)

        train_binary_encoded_features = []
        train_onehot_encoded_features = []

        # encode categorical features of train set
        for feature in categorical_without_target:
            if len(X_train[feature].unique()) == 2:
                lb = LabelBinarizer()
                encoded_feature = lb.fit_transform(X_train[feature])
                encoded_feature_df = pd.DataFrame(encoded_feature, columns=[feature])
                train_binary_encoded_features.append(encoded_feature_df)
            elif len(X_train[feature].unique()) > 2:
                ohe = OneHotEncoder()
                encoded_feature = ohe.fit_transform(X_train[[feature]]).toarray()
                encoded_feature_df = pd.DataFrame(encoded_feature, columns = [feature+"_"+X_train[feature].unique()[i] for i in range(len(X_train[feature].unique()))])
                train_onehot_encoded_features.append(encoded_feature_df)

        X_train = X_train.drop(columns=categorical_without_target)
        train_binary_encoded_features_df = pd.concat(train_binary_encoded_features, axis=1)
        train_onehot_encoded_features_df = pd.concat(train_onehot_encoded_features, axis=1)
        X_train = pd.concat([train_binary_encoded_features_df.reset_index(drop=True), train_onehot_encoded_features_df.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
        train_df_encoded = pd.concat([X_train_df_identifier.reset_index(drop=True), X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        
        # dataset into a numpy array
        X_train_continuous = X_train.values
        X_train_continuous = np.array(X_train_continuous)
      
        # scale train set
        standard_scaler = StandardScaler()
        X_train_continuous_scaled = standard_scaler.fit_transform(X_train_continuous)

        # test set DataFrames for each type of feature (ID, categorical, continuous, target)
        X_test = test_df.drop(columns=[target, identifier[0]])
        X_test_df_identifier = test_df[identifier].reset_index(drop=True)
        X_test_df_categorical = test_df[categorical_without_target].reset_index(drop=True)
        y_test = test_df[target].reset_index(drop=True)

        test_binary_encoded_features = []
        test_onehot_encoded_features = []

        # encode categorical features of test set
        for feature in categorical_without_target:
            if len(X_test[feature].unique()) == 2:
                lb = LabelBinarizer()
                encoded_feature = lb.fit_transform(X_test[feature])
                encoded_feature_df = pd.DataFrame(encoded_feature, columns=[feature])
                test_binary_encoded_features.append(encoded_feature_df)
            elif len(X_test[feature].unique()) > 2:
                ohe = OneHotEncoder()
                encoded_feature = ohe.fit_transform(X_test[[feature]]).toarray()
                encoded_feature_df = pd.DataFrame(encoded_feature, columns = [feature+"_"+X_test[feature].unique()[i] for i in range(len(X_test[feature].unique()))])
                test_onehot_encoded_features.append(encoded_feature_df)

        X_test = X_test.drop(columns=categorical_without_target)
        test_binary_encoded_features_df = pd.concat(test_binary_encoded_features, axis=1)
        test_onehot_encoded_features_df = pd.concat(test_onehot_encoded_features, axis=1)
        X_test = pd.concat([test_binary_encoded_features_df.reset_index(drop=True), test_onehot_encoded_features_df.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
        test_df_encoded = pd.concat([X_test_df_identifier.reset_index(drop=True), X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        # test set into a numpy array
        X_test_continuous = X_test.values
        X_test_continuous = np.array(X_test_continuous)
    
        # scale test set
        X_test_continuous_scaled = standard_scaler.transform(X_test_continuous)

        # number of retained components
        components = list(kwargs.values())[0]

        if method == "pca":
           
            # principal component analysis on train set
            pca = PCA(n_components=components)
            X_train_continuous_tranformed = pca.fit_transform(X_train_continuous_scaled)
            
            # turn into Pandas DataFrame
            train_labels = ["PC"+str(i+1) for i in range(components)]
            X_train_continuous_tranformed_df = pd.DataFrame(X_train_continuous_tranformed, columns=train_labels).reset_index(drop=True)
            
            # merge all train DataFrames
            train_df_reduced = pd.concat([X_train_df_identifier, X_train_continuous_tranformed_df, y_train], axis=1)

            # calculate explained variance per component and total explained variance
            train_explained_variance = pca.explained_variance_ratio_ * 100
            total_train_explained_variance = pca.explained_variance_ratio_.sum() * 100

            # 2-dimensional pca plot
            if plot_type == "2d":
               
                # compute loadings
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                # loadings_labels = train_df.drop(columns=[target, identifier[0]]).columns

                # number of loading to display
                # loadings_number = list(kwargs.values())[1]

                # calculate explained variance per component and total explained variance
                twopca_train_explained_variance = pca.explained_variance_ratio_[0:2].sum() * 100

                train_plot_2d = px.scatter(
                    X_train_continuous_tranformed[:, 0:2], x=0, y=1,
                    color=train_df_reduced[target],
                    title=f'2D PCA plot (2-component Explained Variance: {twopca_train_explained_variance:.2f}%, Total Components: {components}, Total Explained Variance: {total_train_explained_variance:.2f}%)')
                
                # for i, feature in enumerate(continuous):
                #     train_plot_2d.add_shape(
                #         type='line',
                #         x0=0, y0=0,
                #         x1=loadings[i, 0],
                #         y1=loadings[i, 1]
                #     )
                #     train_plot_2d.add_annotation(
                #         x=loadings[i, 0],
                #         y=loadings[i, 1],
                #         ax=0, ay=0,
                #         xanchor="center",
                #         yanchor="bottom",
                #         text=feature
                #     )
                
                train_plot_2d.show()
            
            elif plot_type == "3d":

                # 3-dimensional pca plot
                # calculate explained variance per component and total explained variance
                threepca_train_explained_variance = pca.explained_variance_ratio_[0:3].sum() * 100

                train_plot_3d = px.scatter_3d(
                    X_train_continuous_tranformed[:, 0:3], x=0, y=1, z=2,
                    color=train_df_reduced[target],
                    title=f'3D PCA plot (3-component Explained Variance: {threepca_train_explained_variance:.2f}%, Total Components: {components}, Total Explained Variance: {total_train_explained_variance:.2f}%)',
                    labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'})
                
                train_plot_3d.show()
            
            # multi-dimensional pca plot
            elif plot_type == "multi":
                
                train_plot_labels = {str(i): f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(train_explained_variance)}
                
                train_plot_fig = px.scatter_matrix(
                    X_train_continuous_tranformed,
                    labels=train_plot_labels,
                    dimensions=range(len(train_labels)),
                    color=train_df_reduced[target],
                    title=f'Multi-dimensional PCA plot (Components: {components}, Total Explained Variance: {total_train_explained_variance:.2f}%)')
                
                train_plot_fig.update_traces(diagonal_visible=False)
                
                train_plot_fig.update_layout(height=components*250, width=components*250)
                
                train_plot_fig.show()
            
            # principal component analysis on test set
            X_test_continuous_tranformed = pca.transform(X_test_continuous_scaled)
            
            # turn into Pandas DataFrame
            test_labels = ["PC"+str(i+1) for i in range(components)]
            X_test_continuous_tranformed_df = pd.DataFrame(X_test_continuous_tranformed, columns=test_labels).reset_index(drop=True)

            # merge all test DataFrames
            test_df_reduced = pd.concat([X_test_df_identifier, X_test_continuous_tranformed_df, y_test], axis=1)

            return train_df_encoded, train_df_reduced, test_df_encoded, test_df_reduced
        
        elif method == "mds":
            
            # multidimensional scaling on train set
            mds = MDS(n_components=components)
            X_train_continuous_tranformed = mds.fit_transform(X_train_continuous_scaled)

            # turn into Pandas DataFrame
            train_labels = ["MD"+str(i+1) for i in range(components)]
            X_train_continuous_tranformed_df = pd.DataFrame(X_train_continuous_tranformed, columns=train_labels).reset_index(drop=True)

            # merge all train DataFrames
            train_df_reduced = pd.concat([X_train_df_identifier, X_train_continuous_tranformed_df, y_train], axis=1)

            # 2-dimensional mds plot
            if plot_type == "2d":
                train_plot_2d = px.scatter(X_train_continuous_tranformed[:, 0:2], x=0, y=1, color=train_df_reduced[target], title=f'2D MDS plot:')
                train_plot_2d.show()
            
            elif plot_type == "3d":
                
                # 3-dimensional mds plot
                
                train_plot_3d = px.scatter_3d(
                    X_train_continuous_tranformed[:, 0:3], x=0, y=1, z=2,
                    color=train_df_reduced[target],
                    title=f'3D MDS plot:',
                    labels={'0': 'MD1', '1': 'MD2', '2': 'MD3'})
                
                train_plot_3d.show()
            
            # multi-dimensional mds plot
            elif plot_type == "multi":
                
                train_plot_labels = {str(i): f"MD {i+1}" for i in range(components)}
                
                train_plot_fig = px.scatter_matrix(
                    X_train_continuous_tranformed,
                    labels=train_plot_labels,
                    dimensions=range(len(train_labels)),
                    color=train_df_reduced[target],
                    title=f'Multi-dimensional MDS plot')
                
                train_plot_fig.update_traces(diagonal_visible=False)
                
                train_plot_fig.update_layout(height=components*250, width=components*250)
                
                train_plot_fig.show()
            
            # multi-dimensional scaling on test set
            X_test_continuous_tranformed = mds.fit_transform(X_test_continuous_scaled)
            
            # turn into Pandas DataFrame
            test_labels = ["MD"+str(i+1) for i in range(components)]
            X_test_continuous_tranformed_df = pd.DataFrame(X_test_continuous_tranformed, columns=test_labels).reset_index(drop=True)

            # merge all test DataFrames
            test_df_reduced = pd.concat([X_test_df_identifier, X_test_continuous_tranformed_df, y_test], axis=1)

            return train_df_encoded, train_df_reduced, test_df_encoded, test_df_reduced

        elif method == "tsne":

            # define perplexity value
            perplexity = float(list(kwargs.values())[1])

            # t-SNE on train set
            tsne = TSNE(n_components=components, perplexity=perplexity)
            X_train_continuous_tranformed = tsne.fit_transform(X_train_continuous_scaled)

            # turn into Pandas DataFrame
            train_labels = ["SNE"+str(i+1) for i in range(components)]
            X_train_continuous_tranformed_df = pd.DataFrame(X_train_continuous_tranformed, columns=train_labels).reset_index(drop=True)

            # merge all train DataFrames
            train_df_reduced = pd.concat([X_train_df_identifier, X_train_continuous_tranformed_df, y_train], axis=1)

            # 2-dimensional tsne plot
            if plot_type == "2d":
                train_plot_2d = px.scatter(X_train_continuous_tranformed[:, 0:2], x=0, y=1, color=train_df_reduced[target], title=f'2D t-SNE plot (perplexity {perplexity}):')
                train_plot_2d.show()
            
            elif plot_type == "3d":
                
                # 3-dimensional tsne plot
                
                train_plot_3d = px.scatter_3d(
                    X_train_continuous_tranformed[:, 0:3], x=0, y=1, z=2,
                    color=train_df_reduced[target],
                    title=f'3D t-SNE plot (perplexity {perplexity}):',
                    labels={'0': 'SNE1', '1': 'SNE2', '2': 'SNE3'})
                
                train_plot_3d.show()
            
            # multi-dimensional tsne plot
            elif plot_type == "multi":
                
                train_plot_labels = {str(i): f"SNE {i+1}" for i in range(components)}
                
                train_plot_fig = px.scatter_matrix(
                    X_train_continuous_tranformed,
                    labels=train_plot_labels,
                    dimensions=range(len(train_labels)),
                    color=train_df_reduced[target],
                    title=f'Multi-dimensional t-SNE plot (perplexity {perplexity}):')
                
                train_plot_fig.update_traces(diagonal_visible=False)
                
                train_plot_fig.update_layout(height=components*250, width=components*250)
                
                train_plot_fig.show()
            
            # tsne on test set
            X_test_continuous_tranformed = tsne.fit_transform(X_test_continuous_scaled)
            
            # turn into Pandas DataFrame
            test_labels = ["SNE"+str(i+1) for i in range(components)]
            X_test_continuous_tranformed_df = pd.DataFrame(X_test_continuous_tranformed, columns=test_labels).reset_index(drop=True)

            # merge all test DataFrames
            test_df_reduced = pd.concat([X_test_df_identifier, X_test_continuous_tranformed_df, y_test], axis=1)

            return train_df_encoded, train_df_reduced, test_df_encoded, test_df_reduced
        
        elif method == "umap":
            
            # define umap hyperpamaters
            components = int(list(kwargs.values())[0])
            neighbors = int(list(kwargs.values())[1])
            min_distance = float(list(kwargs.values())[2])
            metric = str(list(kwargs.values())[3])

            # UMAP on train set
            reducer = umap.UMAP(n_components=components, n_neighbors=neighbors, min_dist=min_distance, metric=metric)
            X_train_continuous_tranformed = reducer.fit_transform(X_train_continuous_scaled)

            # turn into Pandas DataFrame
            train_labels = ["UMAP"+str(i+1) for i in range(components)]
            X_train_continuous_tranformed_df = pd.DataFrame(X_train_continuous_tranformed, columns=train_labels).reset_index(drop=True)

            # merge all train DataFrames
            train_df_reduced = pd.concat([X_train_df_identifier, X_train_continuous_tranformed_df, y_train], axis=1)

            # 2-dimensional umap plot
            if plot_type == "2d":
                train_plot_2d = px.scatter(
                    X_train_continuous_tranformed[:, 0:2], x=0, y=1,
                    color=train_df_reduced[target],
                    title=f'2D UMAP plot (neighbors: {neighbors}, distance: {min_distance}, metric: {metric}):')
                train_plot_2d.show()
            
            elif plot_type == "3d":
                
                # 3-dimensional umap plot
                
                train_plot_3d = px.scatter_3d(
                    X_train_continuous_tranformed[:, 0:3], x=0, y=1, z=2,
                    color=train_df_reduced[target],
                    title=f'3D UMAP plot (neighbors: {neighbors}, distance: {min_distance}, metric: {metric}):',
                    labels={'0': 'UMAP1', '1': 'UMAP2', '2': 'UMAP3'})
                
                train_plot_3d.show()
            
            # multi-dimensional umap plot
            elif plot_type == "multi":
                
                train_plot_labels = {str(i): f"UMAP {i+1}" for i in range(components)}
                
                train_plot_fig = px.scatter_matrix(
                    X_train_continuous_tranformed,
                    labels=train_plot_labels,
                    dimensions=range(len(train_labels)),
                    color=train_df_reduced[target],
                    title=f'Multi-dimensional UMAP plot (neighbors: {neighbors}, distance: {min_distance}, metric: {metric}):')
                
                train_plot_fig.update_traces(diagonal_visible=False)
                
                train_plot_fig.update_layout(height=components*250, width=components*250)
                
                train_plot_fig.show()
            
            # tsne on test set
            X_test_continuous_tranformed = reducer.transform(X_test_continuous_scaled)
            
            # turn into Pandas DataFrame
            test_labels = ["UMAP"+str(i+1) for i in range(components)]
            X_test_continuous_tranformed_df = pd.DataFrame(X_test_continuous_tranformed, columns=test_labels).reset_index(drop=True)

            # merge all test DataFrames
            test_df_reduced = pd.concat([X_test_df_identifier, X_test_continuous_tranformed_df, y_test], axis=1)

            return train_df_encoded, train_df_reduced, test_df_encoded, test_df_reduced

### K-MEANS INERTIA PLOT ###
def kmeans_inertia_plot(df: pd.DataFrame, identifier: list, target: str) -> None:
    '''
    The function calculates and plots inertia per cluster.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data (either encoded or after dimensioality reduction)
        identifier (list): identifier features of the dataset
        target (str): target variable
    
    Returns:
        None

    '''

    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)
    
    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
    
    else:
        
        X = df.drop(columns=[identifier[0], target]).values
        X = np.array(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Let the number of clusters be a parameter, so we can get a feel for an appropriate value thereof.
        def cluster_algorithm(n_clusters):
            kmeans = cluster.KMeans(n_clusters=n_clusters, init="k-means++")
            kmeans.fit(X_scaled)
            Z = kmeans.predict(X_scaled)
            return kmeans, Z
            
            
        max_clusters = len(df)

        inertias = np.zeros(max_clusters)

        for i in range(1, max_clusters):
            kmeans, Z = cluster_algorithm(i)
            inertias[i] = kmeans.inertia_
    
        fig = go.Figure(data=go.Scatter(x=tuple(range(1, max_clusters)), y=inertias[1:]))

        fig.update_layout(go.Layout(title=f'K-Means Clustering (select number of clusters based on elbow method)',
                                    xaxis=go.XAxis(title='Number of clusters',
                                                    range=[0, max_clusters]),
                                    yaxis=go.YAxis(title='Inertia')
        ))

        fig.show()
            
### CLUSTERING FUNCTION ###
def clustering(df: pd.DataFrame, input_df: pd.DataFrame, identifier: list, target: str, method: str, plot_type: str, marker_size_ref: str, **kwargs) -> pd.DataFrame:
    
    '''
    The function performs clustering on the given input data.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        input_df (Pandas DataFrame): data structure with loaded data after dimensionality reduction (2d or 3d data)
        identifier (list): identifier features of the dataset
        method (str): clustering algorithm (kmeans, hierarchical, dbscan)
        input_data_type (str): type of data to fit the algorithm (normal, encoded or reduced)
        plot_type (str): type of plot to display (2d: 2-dimensional plot, 3d: 3-dimensional plot)
        marker_size_ref (str): reference feature based on which the size of the markers is defined
    
    **kwargs:
        K-Means:
            clusters (int): number of clusters to retain in the analysis
            initializations (int): number of time the k-means algorithm will be run with different centroid seeds
            random_state (int): determines random number generation for centroid initialization (default=None)
           
    Returns:
        cluster_df (pd.DataFrame): data structure with loaded data

    '''

    if not isinstance(df, pd.DataFrame):
        error_message = "df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    if not isinstance(input_df, pd.DataFrame):
        error_message = "input_df must be specified as a Pandas DataFrame"
        raise TypeError(error_message)
    
    elif not isinstance(identifier, list):
        error_message = "identifier must be specified as a list of strings"
        raise TypeError(error_message)
    
    elif not isinstance(target, str):
        error_message = "target must be specified as a string"
        raise TypeError(error_message)
        
    elif not isinstance(method, str):
        error_message = "method must be specified as a string\noptions: kmeans, hierarchical or dbscan"
        raise TypeError(error_message)
    
    # elif not isinstance(input_data_type, str):
    #     error_message = "input_data_type must be specified as a string\noptions: normal, encoded or reduced"
    #     raise TypeError(error_message)
    
    elif not isinstance(plot_type, str):
        error_message = "plot_type must be specified as a string\noptions: 2d (2-dimensional plot) or 3d (3-dimensional plot)"
        raise TypeError(error_message)
    
    elif not isinstance(marker_size_ref, str):
        error_message = "marker_size_ref must be specified as a string"
        raise TypeError(error_message)
    
    else:

        if method == "kmeans":
            
            true_label_names = np.array(input_df.loc[:,[target]].values)
    
            label_encoder = LabelEncoder()
            true_labels = label_encoder.fit_transform(true_label_names)

            if plot_type == "2d":
                X = input_df.drop(columns=[identifier[0], target]).values
                X = np.array(X)

                # define kmeans hyperpamaters
                clusters = int(list(kwargs.values())[0])
                initializations = int(list(kwargs.values())[1])
                random_state = list(kwargs.values())[2]

                kmeans = cluster.KMeans(n_clusters=clusters, n_init=initializations, random_state=random_state)
                kmeans.fit(X)
                column_labels = ["Component_"+str(i+1) for i in range(2)]
                cluster_df = pd.DataFrame(X, columns=column_labels, index=input_df.index)
                cluster_df["Predicted_Labels"] = kmeans.labels_
                cluster_df["True_Labels"] = label_encoder.inverse_transform(true_labels)
                cluster_df["ID_REF"] = input_df[identifier[0]]

                fig = go.Figure(data=go.Scatter(
                        x=cluster_df["Component_1"].values,
                        y=cluster_df["Component_2"].values,
                        text=cluster_df["ID_REF"],
                        mode='markers',
                        marker=go.Marker(
                            size=df[marker_size_ref],
                            sizemode='diameter',
                            sizeref=df[marker_size_ref].max()/50,
                            opacity=1,
                            color=cluster_df["Predicted_Labels"],
                            colorscale="viridis"
                            )
                        )
                    )

                fig.update_layout(
                    go.Layout(
                        title=f'K-Means Clustering (2D plot)',
                        xaxis=go.XAxis(title="Component_1", showgrid=True, zeroline=True, showticklabels=True),
                        yaxis=go.YAxis(title="Component_2", showgrid=True, zeroline=True, showticklabels=True),
                        hovermode='closest'
                    )
                )

                fig.show()
            
            elif plot_type == "3d":
                X = input_df.drop(columns=[identifier[0], target]).values
                X = np.array(X)

                # define kmeans hyperpamaters
                clusters = int(list(kwargs.values())[0])
                initializations = int(list(kwargs.values())[1])
                random_state = list(kwargs.values())[2]

                kmeans = cluster.KMeans(n_clusters=clusters, n_init=initializations, random_state=random_state)
                kmeans.fit(X)
                column_labels = ["Component_"+str(i+1) for i in range(3)]
                cluster_df = pd.DataFrame(X, columns=column_labels, index=input_df.index)
                cluster_df["Predicted_Labels"] = kmeans.labels_
                cluster_df["True_Labels"] = label_encoder.inverse_transform(true_labels)
                cluster_df["ID_REF"] = input_df[identifier[0]]

                fig = go.Figure(data=go.Scatter3d(
                        x=cluster_df["Component_1"].values,
                        y=cluster_df["Component_2"].values,
                        z=cluster_df["Component_3"].values,
                        text=cluster_df["ID_REF"],
                        mode='markers',
                        marker=go.Marker(
                            size=df[marker_size_ref],
                            sizemode='diameter',
                            sizeref=df[marker_size_ref].max()/50,
                            opacity=1,
                            color=cluster_df["Predicted_Labels"],
                            colorscale="viridis"
                            )
                        )
                    )

                fig.update_layout(
                    go.Layout(
                        title=f'K-Means Clustering (3D plot)',
                        hovermode='closest'
                    )
                )

                fig.show()
            
           

            return cluster_df




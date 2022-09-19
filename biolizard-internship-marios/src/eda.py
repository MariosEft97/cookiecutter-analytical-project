# LOAD PACKAGES
import sys
from turtle import width
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
from sklearn.preprocessing import StandardScaler
# from scipy.stats import gaussian_kde
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
    if not isinstance(df, pd.DataFrame) or not isinstance(type, str) or not isinstance(printout, str):
        raise TypeError
    
    matrix = df.corr(method=type)
    type_dict = {"pearson": "Pearson", "spearman": "Spearman"}

    if printout == "matrix":
        print(f"{type_dict[type]} Correlation Matrix:")
        display(round(matrix, 3))
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

    if not isinstance(df, pd.DataFrame) or not isinstance(features, list) or not isinstance(stratify_var, str) or not isinstance(group_var, str) or not isinstance(title, str) or not isinstance(xtitle, str) or not isinstance(ytitle, str) or not isinstance(widget_description, str) or not isinstance(stratify, bool) or not isinstance(group, bool):
        raise TypeError

    feature = widgets.Dropdown(description=widget_description, value=features[0], options=features)

    if stratify == True:
        if group == True:
            traces = []
            for i, group in enumerate(df[group_var].unique()):
                filter = df[group_var]==df[group_var].unique()[i]
                df_filtered = df[filter]
                trace = go.Box(x=df_filtered[stratify_var], y=df_filtered[feature.value], name=df[group_var].unique()[i], boxmean=True)
                traces.append(trace)

            g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
            
        elif group == False:
            trace = go.Box(x=df[stratify_var], y=df[feature.value], name=feature.value, boxmean=True)
            g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

    elif stratify == False:
        if group == True:
            traces = []
            for i, group in enumerate(df[group_var].unique()):
                filter = df[group_var]==df[group_var].unique()[i]
                df_filtered = df[filter]
                trace = go.Box(y=df_filtered[feature.value], name=df[group_var].unique()[i], boxmean=True)
                traces.append(trace)

            g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

        elif group == False:
            trace = go.Box(y=df[feature.value], name=feature.value, boxmean=True)
            g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
    

    def validate():
        if feature.value in features:
            return True
        else:
            return False
    
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

    container = widgets.VBox([feature])
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

    if not isinstance(df, pd.DataFrame) or not isinstance(features, list) or not isinstance(stratify_var, str) or not isinstance(columns, int) or not isinstance(stratify, bool):
        raise TypeError

    rows = math.ceil(len(features)/columns)
    
    titles = [feature for feature in features]
    
    outter_list = []
    inner_list = []
    
    for i in range(columns):
        inner_list.append({"type": "histogram"})
    
    for i in range(rows):
        outter_list.append(inner_list)


    fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple(titles), specs=outter_list)

    coordinates = []
    for i in range(rows):
        for j in range(columns):
            coordinates.append([i+1, j+1])
    
    for (feature), (coordinate) in zip(features, coordinates):
        
        if stratify==True:
            
            # if group==True:
            #     for i, group in enumerate(df[group_var].unique()):
            #         filter = df[group_var]==df[group_var].unique()[i]
            #         df_filtered = df[filter]
            #         fig.add_trace(go.Box(x=df_filtered[stratify_var], y=df_filtered[feature], name=df[group_var].unique()[i], boxmean=True), row=coordinate[0], col=coordinate[1])
            
            # elif group==False:

            fig.add_trace(go.Box(x=df[stratify_var], y=df[feature], name=feature, boxmean=True), row=coordinate[0], col=coordinate[1])
        
        elif stratify==False:
            
            # if group==True:
            #     for i, group in enumerate(df[group_var].unique()):
            #         filter = df[group_var]==df[group_var].unique()[i]
            #         df_filtered = df[filter]
            #         fig.add_trace(go.Box(y=df_filtered[feature], name=df[group_var].unique()[i], boxmean=True), row=coordinate[0], col=coordinate[1])
            
            # elif group==False:

            fig.add_trace(go.Box(y=df[feature], name=feature, boxmean=True), row=coordinate[0], col=coordinate[1])

    fig.update_layout(height=rows*height, width=columns*width, showlegend=True)

    fig.show()

### HISTOGRAM FUNCTION ###
def hist_plot(df: pd.DataFrame, features: list, group_var: str, title: str, xtitle: str, ytitle: str, widget_description: str, group: bool=False) -> None:
    
    '''
    The function creates an interactive histogram plot.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        features (list): list of predictor features
        group_var (str): binary variable to group the data by (if group=Yes)
        title (str): plot title
        xtitle (str): x axis title
        ytitle (str): y axis title
        widget_description (str): widget description
        group (bool): group feature values by the specified feature (True or False, default=False)
    
    Returns:
        None
    '''

    if not isinstance(df, pd.DataFrame) or not isinstance(features, list) or not isinstance(group_var, str) or not isinstance(title, str) or not isinstance(xtitle, str) or not isinstance(ytitle, str) or not isinstance(widget_description, str) or not isinstance(group, bool):
        raise TypeError

      
    feature = widgets.Dropdown(description=widget_description, value=features[0], options=features)
    
    if group == True:
        traces = []
        for i, group in enumerate(df[group_var].unique()):
            filter = df[group_var]==df[group_var].unique()[i]
            df_filtered = df[filter]
            trace = go.Histogram(x=df_filtered[feature.value], name=df[group_var].unique()[i])
            traces.append(trace)
     
        g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), barmode="group"))

    elif group == False:
        trace = go.Histogram(x=df[feature.value], name=feature.value)
        g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), barmode="group"))
    

    def validate():
        if feature.value in features:
            return True
        else:
            return False
    
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

    if not isinstance(df, pd.DataFrame) or not isinstance(columns, int) or not isinstance(width, int) or not isinstance(height, int):
        raise TypeError

    features = df.columns

    rows = math.ceil(len(features)/columns)
    
    titles = [feature for feature in features]
    
    outter_list = []
    inner_list = []
    
    for i in range(columns):
        inner_list.append({"type": "histogram"})
    
    for i in range(rows):
        outter_list.append(inner_list)


    fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple(titles), specs=outter_list)

    coordinates = []
    for i in range(rows):
        for j in range(columns):
            coordinates.append([i+1, j+1])
    
    for (feature), (coordinate) in zip(features, coordinates):
        fig.add_trace(go.Histogram(x=df[feature], name=feature), row=coordinate[0], col=coordinate[1])

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

    if not isinstance(df, pd.DataFrame) or not isinstance(title, str):
        raise TypeError

    hist_data = [df[gene] for gene in df.columns]
    group_labels = [gene for gene in df.columns]

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

    if not isinstance(df, pd.DataFrame) or not isinstance(columns, int) or not isinstance(width, int) or not isinstance(height, int):
        raise TypeError

    features = df.columns

    rows = math.ceil(len(features)/columns)
    
    titles = [feature for feature in features]
    
    outter_list = []
    inner_list = []
    
    for i in range(columns):
        inner_list.append({"type": "histogram"})
    
    for i in range(rows):
        outter_list.append(inner_list)


    fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple(titles), specs=outter_list)

    hist_data = [df[feature] for feature in features]
    group_labels = [feature for feature in features]
    dist_fig = ff.create_distplot(hist_data, group_labels, show_hist=False)

    coordinates = []
    for i in range(rows):
        for j in range(columns):
            coordinates.append([i+1, j+1])
    
    for (coordinate), (trace) in zip(coordinates, dist_fig.select_traces()):
        fig.add_trace(trace, row=coordinate[0], col=coordinate[1])

    fig.update_layout(height=rows*height, width=columns*width, showlegend=True)

    fig.show()

    return None

### PCA PLOT FUNCTION ###
def pca_plot(train_df: pd.DataFrame, identifier: list, categorical: list, continuous:list, target: str,) -> None:
    
    '''
    The function creates the explained cariance Vs principal components plot.
    
    Parameters:
        train_df (Pandas DataFrame): data structure with train sample
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        target (str): target variable
    
    Returns:
        None
    '''

    if not isinstance(train_df, pd.DataFrame) or not isinstance(identifier, list) or not isinstance(categorical, list) or not isinstance(continuous, list) or not isinstance(target, str):
        raise TypeError
    
    X_train = train_df.drop(columns=[target])

    X_train_continuous = X_train.loc[:,continuous].values
    X_train_continuous = np.array(X_train_continuous)
    
    standard_scaler = StandardScaler()
    X_train_continuous_scaled = standard_scaler.fit_transform(X_train_continuous)

    pca = PCA()
    pca.fit_transform(X_train_continuous_scaled)
        
    explained_variance = list(np.cumsum(pca.explained_variance_ratio_))
    explained_variance.insert(0,0)
    explained_variance = [i * 100 for i in explained_variance]

    fig = go.Figure(data=go.Scatter(y=explained_variance))

    fig.update_layout(go.Layout(title='Percentage of Explained Variance per Principal Component',
                            xaxis=go.layout.XAxis(title='Number of PCs', range=[0, len(pca.components_)]),
                            yaxis=go.layout.YAxis(title='Explained Variance')
                            ))

    fig.show()

### DIMENSIONALITY REDUCTION FUNCTION ###
def dimensionality_reduction(train_df: pd.DataFrame, test_df: pd.DataFrame, identifier: list, categorical: list, continuous:list, target: str, method: str) -> pd.DataFrame:
    
    '''
    The function performs dimensionality reduction techniques on the given data.

    Parameters:
        train_df (Pandas DataFrame): data structure with train sample
        test_df (Pandas DataFrame): data structure with test sample
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        target (str): target variable
        method (str): dimensionality reduction method (pca: PCA, umap: UMAP, tsne: t-SNE)
    Returns:

    '''
    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame) or not isinstance(identifier, list) or not isinstance(categorical, list) or not isinstance(continuous, list) or not isinstance(target, str) or not isinstance(method, str):
        raise TypeError
    
    return None


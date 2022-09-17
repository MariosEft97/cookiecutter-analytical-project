# LOAD PACKAGES
import sys
sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"
from IPython.display import display
import plotly.graph_objects as go
import plotly.figure_factory as ff
from ipywidgets import widgets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
def box_plot(df: pd.DataFrame, x_var: str, y_var: list, group_var: str, title: str, xtitle: str, ytitle: str, widget_description: str, stratify: bool=False, group: bool=False) -> None:
    
    '''
    The function creates an interactive boxplot.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        y_var (list): list of predictor features
        x_var (str): target variable
        group_var (str): binary variable to group the data by (if group=Yes)
        title (str): plot title
        xtitle (str): x axis title
        ytitle (str): y axis title
        widget_description (str): widget description
        stratify (bool): display feature values stratified by the specified feature (True or False, default=False)
        group (bool): group feature values by the specified feature (True or False, default=False)
    
    Returns:
        None
    '''

    if not isinstance(df, pd.DataFrame) or not isinstance(x_var, str) or not isinstance(y_var, list) or not isinstance(group_var, str) or not isinstance(title, str) or not isinstance(xtitle, str) or not isinstance(ytitle, str) or not isinstance(widget_description, str) or not isinstance(stratify, bool) or not isinstance(group, bool):
        raise TypeError


    feature_list = []
    for feature in y_var:
        feature_list.append(feature)
    
    feature = widgets.Dropdown(description=widget_description, value=feature_list[0], options=feature_list)

    if stratify == True:
        if group == True:
            traces = []
            for i, group in enumerate(df[group_var].unique()):
                filter = df[group_var]==df[group_var].unique()[i]
                df_filtered = df[filter]
                trace = go.Box(x=df_filtered[x_var], y=df_filtered[feature.value], name=df[group_var].unique()[i], boxmean=True)
                traces.append(trace)

            g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{feature_list[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
            
        elif group == False:
            trace = go.Box(x=df[x_var], y=df[feature.value], name=feature.value, boxmean=True)
            g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{feature_list[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

    elif stratify == False:
        if group == True:
            traces = []
            for i, group in enumerate(df[group_var].unique()):
                filter = df[group_var]==df[group_var].unique()[i]
                df_filtered = df[filter]
                trace = go.Box(y=df_filtered[feature.value], name=df[group_var].unique()[i], boxmean=True)
                traces.append(trace)

            g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{feature_list[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

        elif group == False:
            trace = go.Box(y=df[feature.value], name=feature.value, boxmean=True)
            g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{feature_list[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
    

    def validate():
        if feature.value in feature_list:
            return True
        else:
            return False
    
    def response(change):
        if validate():
            if feature.value:
                temp_df = df[feature.value]
        else:
            temp_df = df[feature_list[0]]
        
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

### HISTOGRAM FUNCTION ###
def hist_plot(df: pd.DataFrame, y_var: list, group_var: str, title: str, xtitle: str, ytitle: str, widget_description: str, group: bool=False) -> None:
    
    '''
    The function creates an interactive histogram plot.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        y_var (list): list of predictor features
        group_var (str): binary variable to group the data by (if group=Yes)
        title (str): plot title
        xtitle (str): x axis title
        ytitle (str): y axis title
        widget_description (str): widget description
        group (bool): group feature values by the specified feature (True or False, default=False)
    
    Returns:
        None
    '''

    if not isinstance(df, pd.DataFrame) or not isinstance(group_var, str) or not isinstance(title, str) or not isinstance(xtitle, str) or not isinstance(ytitle, str) or not isinstance(widget_description, str) or not isinstance(group, bool):
        raise TypeError

    feature_list = []
    for feature in y_var:
        feature_list.append(feature)
    
    feature = widgets.Dropdown(description=widget_description, value=feature_list[0], options=feature_list)
    
    if group == True:
        traces = []
        for i, group in enumerate(df[group_var].unique()):
            filter = df[group_var]==df[group_var].unique()[i]
            df_filtered = df[filter]
            trace = go.Histogram(x=df_filtered[feature.value], name=df[group_var].unique()[i])
            traces.append(trace)
     
        g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{feature_list[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle)))

    elif group == False:
        trace = go.Histogram(x=df[feature.value], name=feature.value)
        g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{feature_list[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle)))
    

    def validate():
        if feature.value in feature_list:
            return True
        else:
            return False
    
    def response(change):
        if validate():
            if feature.value:
                temp_df = df[feature.value]
        else:
            temp_df = df[feature_list[0]]
        
        x1 = temp_df
        
        with g.batch_update():
            g.data[0].x = x1
            g.layout.xaxis.title = xtitle
            g.layout.yaxis.title = ytitle
            g.layout.title = f"{title} {feature.value}"
    
    feature.observe(response, names="value")

    container = widgets.VBox([feature])
    display(widgets.VBox([container, g]))

### DISTRIBUTION PLOT FUNCTION ###
def dist_plot(df: pd.DataFrame, selection: list, title: str, xtitle: str, ytitle: str) -> None:
    
    '''
    The function creates an interactive distribution plot.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        selection (list): list of predictor features
        title (str): plot title
        xtitle (str): x axis title
        ytitle (str): y axis title
    
    Returns:
        None
    '''

    if not isinstance(df, pd.DataFrame) or not isinstance(selection, list) or not isinstance(title, str) or not isinstance(xtitle, str) or not isinstance(ytitle, str):
        raise TypeError

    feature_list = []
    for feature in selection:
        feature_list.append(feature)

    hist_data = [df[gene] for gene in feature_list]
    group_labels = [gene for gene in feature_list]

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
    fig.update_layout(title_text='Gene Expression Distribution Plot:')
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


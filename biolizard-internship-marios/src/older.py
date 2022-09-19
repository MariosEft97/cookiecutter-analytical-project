# ### LOAD PACKAGES ###
# import sys
# sys.path.append(r"C:\Users\35799\Desktop\cookiecutter-analytical-project\biolizard-internship-marios\src")
# import pandas as pd

# ### TREAT NA FUNCTION ### (NOT USED)
# def treat_na(df: pd.DataFrame, identifier: list, categorical: list, continuous:list, drop_na_rows: bool=False, impute_cutoff: float=0.5, categorical_imputer: str="mode", continuous_imputer: str="mean") -> pd.DataFrame:

#     """
#     The function treats missing values.

#     Parameters:
#         df (Pandas DataFrame): data structure with loaded data
#         identifier (list): identifier features of the dataset
#         categorical (list): categorical features of the dataset
#         continuous (list): continuous features of the dataset
#         drop_na (bool): drop rows containing missing values (True or False, default=False)
#         impute_cutoff (float): if NA fraction is less or equal to the specified value, missing values are imputed otherwise the feature is removed (defaul=0.5)
#         categorical_imputer (str): how categorcial missing values are imputed (mode, default=mode)
#         continuous_imputer (str): how missing values are imputed (mean, median, default=mean)
    
#     Returns:
#         df_treat_na (Pandas DataFrame): data structure with no missing values
#     """

#     if not isinstance(df, pd.DataFrame) or not isinstance(identifier, list) or not isinstance(categorical, list) or not isinstance(continuous, list) or not isinstance(drop_na_rows, bool) or not isinstance(impute_cutoff, float) or not isinstance(categorical_imputer, str) or not isinstance(continuous_imputer, str):
#         raise TypeError
   
#     df_treat_na = df.copy(deep=True)

#     for column in df_treat_na.columns:
        
#         missing_fraction = df_treat_na[column].isnull().sum()/df_treat_na.shape[0]
        
#         if column in identifier:
#             if drop_na_rows == True:
#                 df_treat_na.drop(df_treat_na.loc[df_treat_na[column].isnull()].index, inplace=True)
        
#         if column in continuous:
#             if drop_na_rows == False:
#                 if missing_fraction < impute_cutoff:
#                     if continuous_imputer == "mean":
#                         df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].mean())
#                     elif continuous_imputer == "median":
#                         df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].median())
#                 elif missing_fraction >= impute_cutoff:
#                     df_treat_na.dropna(axis=1, subset=[column], inplace=True)
#             elif drop_na_rows == True:
#                 if missing_fraction < impute_cutoff:
#                     df_treat_na.dropna(axis=0, subset=[column], inplace=True)
#                 elif missing_fraction >= impute_cutoff:
#                     df_treat_na.dropna(axis=1, subset=[column], inplace=True)
        
#         if column in categorical:
#             if drop_na_rows == False:
#                 if missing_fraction < impute_cutoff:
#                     if categorical_imputer == "mode":
#                         df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].mode()[0])
#                 elif missing_fraction >= impute_cutoff:
#                     df_treat_na.dropna(axis=1, subset=[column], inplace=True)
#             elif drop_na_rows == True:
#                 if missing_fraction < impute_cutoff:
#                     df_treat_na.dropna(axis=0, subset=[column], inplace=True)
#                 elif missing_fraction >= impute_cutoff:
#                     df_treat_na.dropna(axis=1, subset=[column], inplace=True)
    
#     return df_treat_na

# ### DISTRIBUTION PLOT FUNCTION (version 2 experimental)###
# def calc_curve(df: pd.DataFrame, feature: str):
    
#     """
#     The function calculates the probability density of the given data.
    
#     Parameters:
#         df (Pandas DataFrame): data structure with loaded data
#         feature (str): feature whose probability density we want to calculate 

#     Returns:
#         X (list):
#         Y (list):
#     """
#     # min_, max_ = df[feature].min(), df[feature].max()
#     # X = [min_ + i * ((max_ - min_) / 500) for i in range(501)]
#     # Y = gaussian_kde(df[feature]).evaluate(X)
#     # return(X, Y)
#     count, bins_count = np.histogram(df[feature], bins=10)
#     pdf = count/sum(count)
#     return pdf

# def dist_plot_v2(df: pd.DataFrame, y_var: list, group_var: str, title: str, xtitle: str, ytitle: str, widget_description: str, group: bool=False) -> None:
    
#     '''
#     The function creates an interactive histogram plot.

#     Parameters:
#         df (Pandas DataFrame): data structure with loaded data
#         y_var (list): list of predictor features
#         group_var (str): binary variable to group the data by (if group=Yes)
#         title (str): plot title
#         xtitle (str): x axis title
#         ytitle (str): y axis title
#         widget_description (str): widget description
#         group (bool): group feature values by the specified feature (True or False, default=False)
    
#     Returns:
#         None
#     '''

#     if not isinstance(df, pd.DataFrame) or not isinstance(group_var, str) or not isinstance(title, str) or not isinstance(xtitle, str) or not isinstance(ytitle, str) or not isinstance(widget_description, str) or not isinstance(group, bool):
#         raise TypeError

#     feature_list = []
#     for feature in y_var:
#         feature_list.append(feature)
    
#     feature = widgets.Dropdown(description=widget_description, value=feature_list[0], options=feature_list)
    
#     if group == True:
#         traces = []
#         for i, group in enumerate(df[group_var].unique()):
#             filter = df[group_var]==df[group_var].unique()[i]
#             df_filtered = df[filter]
#             # x, y = calc_curve(df_filtered, feature.value)
#             # traces.append({"x":x, "y":y, "name":df[group_var].unique()[i]})
#             pdf = calc_curve(df_filtered, feature.value)
#             traces.append({"y":pdf, "name":df[group_var].unique()[i]})
     
#         g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle)))

#     elif group == False:
#         traces = []
#         # x, y = calc_curve(df, feature.value)
#         # traces.append({"x":x, "y":y, "name":feature.value})
#         pdf = calc_curve(df, feature.value)
#         traces.append({"y":pdf, "name":feature.value})
#         g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle)))
    

#     def validate():
#         if feature.value in feature_list:
#             return True
#         else:
#             return False
    
#     def response(change):
#         if validate():
#             if feature.value:
#                 temp_df = df[feature.value]
#         else:
#             temp_df = df[feature_list[0]]
        
#         x1 = temp_df
        
#         with g.batch_update():
#             g.data[0].x = x1
#             g.layout.xaxis.title = xtitle
#             g.layout.yaxis.title = ytitle
#             g.layout.title = f"{title} {feature.value}"
    
#     feature.observe(response, names="value")

#     container = widgets.VBox([feature])
#     display(widgets.VBox([container, g]))
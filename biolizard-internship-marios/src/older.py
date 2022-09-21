# ### BOXPLOT FUNCTION ###
# def box_plot_v2(df: pd.DataFrame, features: list, categorical: list, stratify_var: str, group_var: str, title: str, xtitle: str, ytitle: str, widget_description: str, stratify: bool=False, group: bool=False) -> None:
    
#     '''
#     The function creates an interactive boxplot.

#     Parameters:
#         df (Pandas DataFrame): data structure with loaded data
#         features (list): list of predictor features
#         stratify_var (str): variable to stratify the data by (if stratify=True)
#         group_var (str): variable to group the data by (if group=True)
#         title (str): plot title
#         xtitle (str): x axis title
#         ytitle (str): y axis title
#         widget_description (str): intercative widget description
#         stratify (bool): display feature values stratified by the specified feature (True or False, default=False)
#         group (bool): group feature values by the specified feature (True or False, default=False)
    
#     Returns:
#         None
#     '''

#     if not isinstance(df, pd.DataFrame):
#         error_message = "df must be specified as a Pandas DataFrame"
#         raise TypeError(error_message)
    
#     elif not isinstance(features, list):
#         error_message = "features must be specified as a list of strings"
#         raise TypeError(error_message)
    
#     elif not isinstance(stratify_var, str):
#         error_message = "stratify_var must be specified as a string"
#         raise TypeError(error_message)

#     elif not isinstance(group_var, str):
#         error_message = "group_var must be specified as a string"
#         raise TypeError(error_message)
    
#     elif not isinstance(title, str):
#         error_message = "title must be specified as a string"
#         raise TypeError(error_message)
    
#     elif not isinstance(xtitle, str):
#         error_message = "xtitle must be specified as a string"
#         raise TypeError(error_message)
    
#     elif not isinstance(ytitle, str):
#         error_message = "ytitle must be specified as a string"
#         raise TypeError(error_message)
    
#     elif not isinstance(widget_description, str):
#         error_message = "widget_description must be specified as a string"
#         raise TypeError(error_message)
    
#     elif not isinstance(stratify, bool):
#         error_message = "stratify must be specified as a boolean value (True or False)"
#         raise TypeError(error_message)
    
#     elif not isinstance(group, bool):
#         error_message = "group must be specified as a boolean value (True or False)"
#         raise TypeError(error_message)

#     else:
        
#         # define widget
#         feature = widgets.Dropdown(description="Select:", value=features[0], options=features)
#         stratify_bool = widgets.Dropdown(description="Stratify:", value=False, options=[False, True])
#         group_bool = widgets.Dropdown(description="Group:", value=False, options=[False, True])
#         stratify_feature = widgets.Dropdown(description="Stratify by:", value=categorical[0], options=categorical)
#         group_feature = widgets.Dropdown(description="Group by:", value=categorical[0], options=categorical)

#         input_widgets = widgets.HBox([feature, stratify_bool, group_bool])

#         output = widgets.Output()

#         def filter(feature, stratify_bool, group_bool, stratify_feature, group_feature):

#             output.clear_output()
            
#             # display boxplots per level of specified stratify feature
#             if stratify_bool == True:
#                 with output:
#                     display(stratify_feature)
#                     # display boxplots per level of specified stratify feature and divide into groups based on the levels of the group feature
#                     if group_bool == True:
#                         display(group_feature)
#                         # for each level of the group feature create a trace
#                         traces = []
#                         for i, group in enumerate(df[group_feature].unique()):
#                             filter = df[group_feature]==df[group_feature].unique()[i]
#                             df_filtered = df[filter]
#                             trace = go.Box(x=df_filtered[stratify_feature], y=df_filtered[feature], name=df[group_feature].unique()[i], boxmean=True)
#                             traces.append(trace)

#                         # create initial figure with traces
#                         g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
                
#                     # no grouping is done
#                     elif group == False:
#                         trace = go.Box(x=df[stratify_feature], y=df[feature], name=feature, boxmean=True)
#                         g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

#             # no stratification is done
#             elif stratify_bool == False:
#                 with output:
#                     # display boxplots divided into groups based on the levels of the specified group feature
#                     if group_bool == True:
#                         display(group_feature)
#                         # for each level of the group feature create a trace
#                         traces = []
#                         for i, group in enumerate(df[group_feature].unique()):
#                             filter = df[group_feature]==df[group_feature].unique()[i]
#                             df_filtered = df[filter]
#                             trace = go.Box(y=df_filtered[feature], name=df[group_feature].unique()[i], boxmean=True)
#                             traces.append(trace)
                        
#                         # create initial figure with traces
#                         g = go.FigureWidget(data=traces, layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))

#                     # no grouping is done
#                     elif group_bool == False:
#                         trace = go.Box(y=df[feature], name=feature, boxmean=True)
#                         g = go.FigureWidget(data=[trace], layout=go.Layout(title=dict(text=f'{title}{features[0]}'), xaxis=dict(title=xtitle), yaxis=dict(title=ytitle), boxmode='group'))
            
#             if feature.value:
#                 with output:
#                     temp_df = df[feature.value]
#             else:
#                 with output:
#                     temp_df = df[features[0]]
            
#             x1 = temp_df
            
#             with g.batch_update():
#                 g.data[0].y = x1
#                 g.layout.barmode = 'overlay'
#                 g.layout.xaxis.title = xtitle
#                 g.layout.yaxis.title = ytitle
#                 g.layout.title = f"{title} {feature.value}"
            
#             return g
            
#         def feature_response(change):
#             filter(change.new, stratify_bool.value, group_bool.value, stratify_feature.value, group_feature.value)
        
#         def stratify_bool_response(change):
#             filter(feature.value, change.new, group_bool.value, stratify_feature.value, group_feature.value)
        
#         def group_bool_response(change):
#             filter(feature.value, stratify_bool.value, change.new, stratify_feature.value, group_feature.value)
        
#         def stratify_feature_response(change):
#             filter(feature.value, stratify_bool.value, group_bool.value, change.new, group_feature.value)
        
#         def group_feature_response(change):
#             filter(feature.value, stratify_bool.value,  group_bool.value, stratify_feature.value, change.new)
        
#         feature.observe(feature_response, names="value")
#         stratify_bool.observe(stratify_bool_response, names="value")
#         group_bool.observe(group_bool_response, names="value")
#         stratify_feature.observe(stratify_feature_response, names="value")
#         group_feature.observe(group_feature_response, names="value")

#         # display widget and figure
#         container = widgets.HBox([feature, stratify_bool, group_bool])
#         display(widgets.VBox([container, g]))

        
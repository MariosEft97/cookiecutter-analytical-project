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
                
        







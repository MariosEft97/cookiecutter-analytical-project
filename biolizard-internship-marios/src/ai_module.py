def data_load(folder, filename):
    
    '''
    The function loads the data and returns an appropriate data structure (Pandas DataFrame).

    Parameters:
        folder: path of the folder where the data are found
        filename: name of the data file 
    
    Returns:
        Pandas DataFrame with loaded data
    '''
    
    import os
    import pandas as pd
  
    filepath = os.path.join(folder, filename)
    
    df = ""
    
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.txt'):
        df = pd.read_table(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        print(f"Error in {filename}")
        print("File extension is not correct.")
        print("Please specify a correct one (xlsx, csv, txt, json).")
    
    df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

    return df
    
def data_info(df, filename, threshold):
    
    """
    The function creates and displays a report containing the following info:
        1) Name of the data file
        2) DataFrame dimensions (row, columns)
        3) Categorical features (if present) and their unique values
        4) Continuous features (if present) and their descriptive statistics
        5) Missing data (if present)
    
    Parameters:
        df: Pandas DataFrame
        filename: name of the data file
        threshold (int): a feature having more unique values than the threshold is considered as continuous
    
    Returns:
        categorical (list): list of categorical features in the DataFrame
        continuous (list): list of continuous features in the DataFrame    
    """

    import numpy as np
    from tabulate import tabulate

    # Number of entries and features
    
    entry_num = len(df)
    feature_num = len(df.columns)
    
    print("\n")
    print(f"DATA FILE:")
    print(100*"-")
    print(f"{filename}")
    print(100*"-")
    print("\n")
    print("DIMENSIONS:")
    print(100*"-")
    print(f"Entries: {entry_num}")
    print(f"Features: {feature_num}")
    print(100*"-")
    print("\n")
    
    # print("FEATURES:")
    # print(100*"-")
    # print(df.columns)
    # print(100*"-")
    # print("\n")
    
    # Categorical and Continuous features
    
    categorical = []
    continuous = []
    identifier = []

    unique_values = []
    continuous_statistics = []

    columns = list(df.columns)

    for column in columns:
        if (len(df.loc[:,column].unique()) == entry_num) and (df[column].dtype == "object"):
            identifier.append(column)
        elif len(df.loc[:,column].unique()) >= threshold:
            continuous.append(column)
        else:
            categorical.append(column)
    
    for column in categorical:
        unique_values.append([column, df[column].unique()])
    
    for column in continuous:
        continuous_statistics.append(
            [
                column,
                round(len(df[column]), 3),
                round(np.mean(df[column]), 3),
                round(np.std(df[column]), 3),
                round(np.min(df[column]), 3),
                round(np.percentile(df[column], q=25), 3),
                round(np.median(df[column]), 3),
                round(np.percentile(df[column], q=75), 3),
                round(np.max(df[column]), 3)
                ]
                )
    
    print(f"CATEGORICAL FEATURES:")
    print(100*"-")
    
    if len(categorical)>=1:
        print(tabulate(unique_values, headers=["Features", "Categories"]))
    else:
        print("There are no categorical features in the dataset.")
    
    print(100*"-")
    print("\n")
    
    print(f"CONTINUOUS FEATURES:")
    print(100*"-")

    print(tabulate(continuous_statistics, headers=["Features", "Count", "Mean", "Std", "Min", "25th", "Median", "75th", "Max"]))
    print(100*"-")
    print("\n")

    # Missing data
    
    missing_data = df.isnull().sum()
    missing_data_dict = dict(missing_data)
    with_na = []
    without_na = []
    
    print("MISSING DATA:")
    print(100*"-")
   
    for key, val in missing_data_dict.items():
        if val>0:
            with_na.append([key, val])
        elif val==0:
            without_na.append([key, val])
    
    if len(with_na)>=1:
        print(f"The following features have missing values:")
        print(tabulate(with_na, headers=["Features", "Missing Data"]))
        print("\n")
    else:
        pass
        
    if len(without_na)==feature_num:
        print("There are no missing values in the dataset.")
    elif len(without_na)>=1:
        print(f"The following features do not have missing values:")
        print(tabulate(without_na, headers=["Features", "Missing Data"]))
    else:
        pass
    print(100*"-")

    return categorical, continuous


def correlations(df, type, printout):
    
    """
    The function creates a correlation matrix/heatmap of the continuous features in the dataset.
    
    Parameters:
        df: Pandas DataFrame
        type (str): type of correlations (pearson or spearman)
        printout (str): how correlations are displayed (matrix or heatmap)
    
    Returns:
        None

    """

    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default = "vscode"
    from IPython.display import display

    matrix = df.corr(method=type)

    if printout == "matrix":
        print("Correlation Matrix:")
        display(round(matrix, 3))
    elif printout == "heatmap":
        print("Heatmap:")
        fig = px.imshow(round(matrix, 3), text_auto=True, color_continuous_scale="Viridis")
        fig.show()


# def data_processing(df):




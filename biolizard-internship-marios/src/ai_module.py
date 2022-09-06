def data_load(folder, filename):
    
    '''
    The function loads the data and returns an appropriate data structure (Pandas DataFrame).

    Parameters:
        folder (str): path of the folder where the data are found
        filename (str): name of the data file 
    
    Returns:
        df (Pandas DataFrame): data structure with loaded data
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
    
def data_info(df, filename, threshold=20):
    
    """
    The function creates and displays a report containing the following info:
        1) Name of the data file
        2) DataFrame dimensions (row, columns)
        3) Categorical features (if present) and their unique values
        4) Continuous features (if present) and their descriptive statistics
        5) Missing data (if present)
        6) Duplicate values (if present)
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        filename (str): name of the data file
        threshold (int): a feature having more unique values than the threshold is considered as continuous
    
    Returns:
        identifier (list): list with identifier columns
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
        unique_values.append([column, df[column].value_counts().sort_index(ascending=True).to_dict()])
    
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
        print(tabulate(unique_values, headers=["Features", "Categories & Counts"]))
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

    print("DUPLICATE VALUES:")
    print(100*"-")

    if df.duplicated().sum() == 0:
        print("There are now duplicate entries in the dataset.")
    elif df.duplicated().sum() > 0:
        print("There following entries are duplicated.")
        print(print(df[df.duplicated()==True]))
    
    print(100*"-")
    
    return identifier, categorical, continuous


def correlations(df, type="matrix", printout="pearson"):
    
    """
    The function creates a correlation matrix/heatmap of the continuous features in the dataset.
    
    Parameters:
        df (Pandas DataFrame): data structure with loaded data
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

# def data_split(df, target, type, train_proportion=0.8, test_proportion=0.2, validation_proportion=0.2, stratify=True, shuffle=False):
    
#     """
#     The function splits the data into train-test sets or train-validation-test sets.

#     Parameters:
#         df (Pandas DataFrame): data structure with loaded data
#         target (str): target variable
#         type (str): split type (tt, tvt)
#             tt: train-test
#             tvt: train-validation-test
#         train_proportion (float): fraction of data to be used for training (0-1, default=0.8)
#         test_proportion (float): fraction of data to be used for testing (0-1, default=0.2)
#         validation_proportion (float): fraction of data to be used for validation (0-1, default=0.2)
#         stratify (bool): stratified split (True or False, default=True)
#         shuffle (bool): shuffle data (True or False, default=False)

#     Returns:
#         train_df (Pandas DataFrame): data structure with training data
#         test_df (Pandas DataFrame): data structure with testing data
#         validation_df (Pandas DataFrame): data structure with validation data
#     """

#     import numpy as np
#     from sklearn.model_selection import train_test_split

#     X = df.drop(columns=[target])
#     y = df[[target]]

#     if shuffle == False:
#         if stratify == False:
#             if type == "tt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, random_state=0)
#             elif type == "tvt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, random_state=0)
#                 X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_proportion, test_size=validation_proportion, random_state=0)
#         elif stratify == True:
#             if type == "tt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, stratify=target, random_state=0)
#             elif type == "tvt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, stratify=target, random_state=0)
#                 X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_proportion, test_size=validation_proportion, stratify=y_train, random_state=0)
#     elif shuffle == True:
#         if stratify == False:
#             if type == "tt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, shuffle=True)
#             elif type == "tvt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, shuffle=True)
#                 X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_proportion, test_size=validation_proportion, shuffle=True)
#         elif stratify == True:
#             if type == "tt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, stratify=target, shuffle=True)
#             elif type == "tvt":
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, test_size=test_proportion, stratify=target, shuffle=True)
#                 X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_proportion, test_size=validation_proportion, stratify=y_train, shuffle=True)

#     return X, y, X_train, y_train, X_val, y_val, X_test, y_test


def treat_na(df, identifier, categorical, continuous, drop_na_rows=False, impute_value=0.5, categorical_imputer="mode", continuous_imputer="mean"):

    """
    The function treats missing values.

    Parameters:
        df (Pandas DataFrame): data structure with loaded data
        identifier (list): identifier features of the dataset
        categorical (list): categorical features of the dataset
        continuous (list): continuous features of the dataset
        drop_na (bool): drop rows containing missing values (True or False, default=False)
        impute_value (float): if NA fraction is less or equal to the specified value, missing values are imputed otherwise the feature is removed (defaul=0.5)
        categorical_imputer (str): how categorcial missing values are imputed (mode, default=mode)
        continuous_imputer (str): how missing values are imputed (mean, median, default=mean)
    
    Returns:
        df_treat_na (Pandas DataFrame): data structure with no missing values
    """
   
    df_treat_na = df.copy(deep=True)

    for column in df_treat_na.columns:
        
        missing_fraction = df_treat_na[column].isnull().sum()/df_treat_na.shape[0]
        
        if column in identifier:
            if drop_na_rows == False:
                pass
            elif drop_na_rows == True:
                df_treat_na.drop(df_treat_na.loc[df_treat_na[column].isnull()].index, inplace=True)
        
        if column in continuous:
            if drop_na_rows == False:
                if missing_fraction < impute_value:
                    if continuous_imputer == "mean":
                        df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].mean())
                    elif continuous_imputer == "median":
                        df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].median())
                elif missing_fraction >= impute_value:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
            elif drop_na_rows == True:
                if missing_fraction < impute_value:
                    df_treat_na.dropna(axis=0, subset=[column], inplace=True)
                elif missing_fraction >= impute_value:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
        
        if column in categorical:
            if drop_na_rows == False:
                if missing_fraction < impute_value:
                    if categorical_imputer == "mode":
                        df_treat_na[column] = df_treat_na[column].fillna(df_treat_na[column].mode()[0])
                elif missing_fraction >= impute_value:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
            elif drop_na_rows == True:
                if missing_fraction < impute_value:
                    df_treat_na.dropna(axis=0, subset=[column], inplace=True)
                elif missing_fraction >= impute_value:
                    df_treat_na.dropna(axis=1, subset=[column], inplace=True)
    
    return df_treat_na

def treat_duplicate(df_treat_na, subset=None, keep="first"):
    
    '''
    The function identifies and removes duplicate entries from the dataset (if present).

    Parameters:
        df_treat_na (Pandas DataFrame): data structure with no missing values
        subset (list): subset of features to be considered, by deault all features are considered (default=None)
        keep (str): which occurance of duplicate value to keep (first, last, False)
    
    Returns:
        df_treat_duplicate (Pandas DataFrame): data structure with no missing values or duplicated entries
    '''

    df_treat_duplicate = df_treat_na.drop_duplicates(keep=keep, subset=subset)

    return df_treat_duplicate

# def treat_outliers(df_treat_duplicate, method):
    
#     """
#     The function identifies and removes outlying observations from the dataset (if present).

#     Parameters:
#         df_treat_duplicate (Pandas DataFrame): data structure with no missing values or duplicated entries
#         method (str): automatic outlier detection method (if, mcd, lof, svm)
#             if: isolation forest
#             mcd: minimum covariance distance
#             lof: local outlier factor
#             svm: one-class support vector machine
    
#     Returns:
#         df_treat_outlier (Pandas DataFrame): data structure with no missing values, duplicated entries or outlying obrervations
#     """

#     from sklearn.ensemble import IsolationForest

#     return None




# def data_processing(df, missing=True, impute=0.5, imputer, outliers=True, normalize=False, standardize=True, encoding=False):
    
#     """
#     The function performs the preprocessing of the data:
#     1) Identifies and removes/imputes missing values.
#     2) Identifies and removes outlying values.
#     3) Normalizes/standardizes data if needed.
#     4) Changes the format/encoding of values if needed.

#     Parameters:
#         df: Pandas DataFrame
#         missing (bool): treat missing values (True or False) (default=True)
#         impute (float): if NA fraction is less or equal to the specified value missing values are imputed otherwise they are removed (defaul=0.5)
#         imputer (str): how missing values are imputed (mean, median, mode)
#         outliers (bool): treat outliers
#     """



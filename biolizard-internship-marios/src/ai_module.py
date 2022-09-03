def data_import(folder, filename):
    
    '''The function imports the data and returns an appropriate data structure (Pandas DataFrame)'''
    
    import os
    import pandas as pd
  
    filepath = os.path.join(folder, filename)
    
    data = ""
    
    if filepath.endswith('.xlsx'):
        data = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.txt'):
        data = pd.read_table(filepath)
    elif filepath.endswith('.json'):
        data = pd.read_json(filepath)
    else:
        print(f"Error in {filename}")
        print("File extension is not correct.")
        print("Please specify a correct one (xlsx, csv, txt, json).")
    
    return data
    
def data_info(df, name, threshold):
    
    """The function displays important information about the DataFrames"""
    
    import pandas as pd
    from tabulate import tabulate

    # Number of rows and columns
    
    row_num = len(df)
    column_num = len(df.columns)
    
    print("\n")
    print(f"{name}")
    print("\n")
    print(100*"-")
    print(f"Rows: {row_num}")
    print(f"Columns: {column_num}")
    print(100*"-")
    print("\n")
    
    # Categorical and Continuous features
    
    categorical = []
    continuous = []

    categories = []

    columns = list(df.columns)

    for column in columns:
        if (len(df.loc[:,column].unique()) >= threshold):
            continuous.append(column)
        else:
            categorical.append(column)
    
    for column in categorical:
        categories.append([column, df[column].unique()])
    
    print(f"CATEGORICAL FEATURES:")
    print(100*"-")
    
    if len(categorical)>=1:
        # print("Feature: \t\t\t Categories:")
        # for i, column in enumerate(categorical):
        #     print(f"{i+1}: {column} \t\t\t {df[column].unique()}")
        print(tabulate(categories, headers=["Features", "Categories"]))
    else:
        print("There are no categorical features in the dataset.")
    print(100*"-")
    print("\n")
    print(f"CONTINUOUS FEATURES:")
    print(100*"-")
    print(round(df[continuous].describe(), 3))
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
            # print(f"Feature {key} has {val} missing values.")
        elif val==0:
            without_na.append([key, val])
            # print(f"No missing values in feature {key}.")
    
    if len(with_na)>=1:
        print(f"The following features have missing values:")
        print(tabulate(with_na, headers=["Features", "Missing Data"]))
        # for key, val in with_na.items():
        #     print(f"{key}: \t {val} missing values.")
    else:
        pass
    
    print("\n")
    
    if len(without_na)==column_num:
        print("There are no missing values in the dataset.")
    elif len(without_na)>=1:
        print(f"The following features do not have missing values:")
        print(tabulate(without_na, headers=["Features", "Missing Data"]))
        # for key, val in without_na.items():
        #     print(f"{key}")
    else:
        pass
    print(100*"-")


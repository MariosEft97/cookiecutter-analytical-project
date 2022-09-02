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
    
    return data
    
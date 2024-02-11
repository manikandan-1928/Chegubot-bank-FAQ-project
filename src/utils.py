import numpy as np

def get_max5(arr):
    '''
    Get the indices of the top 5 maximum values in an array.

    Parameters:
    - arr (list or numpy array): Input array.

    Returns:
    - list: Indices of the top 5 maximum values.
    '''
    ixarr = [(el, ix) for ix, el in enumerate(arr)]
    ixarr.sort(reverse=True)
    ixs = [i[1] for i in ixarr[:5]]
    return ixs

def filter_data_by_class(selected_class_idx, data):
    '''
    Filter data by a selected class index.

    Parameters:
    - selected_class_idx (int): Index of the selected class.
    - data (pandas DataFrame): Input data.

    Returns:
    - pandas DataFrame: Filtered data for the selected class.
    - str: Name of the selected class.
    '''
    class_names = ['security', 'loans', 'accounts', 'insurance', 'investments', 'fundstransfer', 'cards']
    selected_class = class_names[selected_class_idx - 1]
    filtered_data = data[data['Class'] == selected_class]
    return filtered_data, selected_class

def calculate_class_indices(data):
    '''
    Calculate the starting indices for each class in the data.

    Parameters:
    - data (pandas DataFrame): Input data.

    Returns:
    - dict: Dictionary mapping class names to their starting indices.
    '''
    class_indices = {}
    start_index = 0
    for class_name in data['Class'].unique():
        class_indices[class_name] = start_index
        start_index += len(data[data['Class'] == class_name])
    return class_indices

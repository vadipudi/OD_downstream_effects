import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

def one_hot_encode_and_save(input_file, output_file):
    # Read the dataset from CSV
    dataset = pd.read_csv(input_file)

    # Remove the last column from the dataset
    last_col = dataset.iloc[:, -1]
    data = dataset.iloc[:, :-1]
    
    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(exclude=['object']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    

    # Perform one-hot encoding for categorical columns
    encoded_dataset = pd.get_dummies(data, columns=categorical_cols)

    sub_encoded_dataset = pd.get_dummies(data[categorical_cols], columns=categorical_cols)


    # Reorder columns to place hot-encoded columns in the same position as the original categorical columns
    added_count=0
    for col in categorical_cols:
        hot_cols = [i for i, s in enumerate(list(sub_encoded_dataset.columns)) if s.startswith(col)]
        col_ind = data.columns.get_loc(col)+added_count
        reordered_columns = list(data.columns)[:col_ind]+list(np.array(sub_encoded_dataset.columns)[hot_cols])+ list(data.columns)[col_ind+1:]
        added_count=added_count+len(hot_cols)


    encoded_dataset = encoded_dataset.reindex(columns=reordered_columns)

    # Append last_col to encoded_dataset
    encoded_dataset[last_col.name] = last_col



    # Save the encoded dataset to a new CSV file
    encoded_dataset.to_csv(output_file, index=False)



def cyclical_encode_and_save(input_file, output_file):
    # Read the dataset from CSV
    dataset = pd.read_csv(input_file)

    # Remove the last column from the dataset
    last_col = dataset.iloc[:, -1]
    data = dataset.iloc[:, :-1]

    # Define cyclical encoding function
    def encode_cyclical(data, column, period):
        data[column + '_sin'] = (np.sin(2 * np.pi * data[column] / period)+1)
        data[column + '_cos'] = (np.cos(2 * np.pi * data[column] / period)+1)
        return data.drop(column, axis=1)

    # Cyclical encode applicable features
    for column in data.columns:
        if column == 'Date':
            data = encode_cyclical(data, column, 31)
        elif column == 'Month':
            data = encode_cyclical(data, column, 12)
        elif column == 'weekday':
            data = encode_cyclical(data, column, 7)
        elif column == 'Hour':
            data = encode_cyclical(data, column, 24)
        elif column == 'Minute':
            data = encode_cyclical(data, column, 60)
        elif column == 'Second':
            data = encode_cyclical(data, column, 60)
        elif column == 'season':
            data = encode_cyclical(data, column, 4)

    # Save the encoded dataset to a new CSV file
    encoded_dataset = pd.concat([data, last_col], axis=1)
    encoded_dataset.to_csv(output_file, index=False)

def nominal_encode(input_file, output_file):
    # Read the dataset from CSV
    dataset = pd.read_csv(input_file)

    # Get the unique values in the target column
    unique_values = dataset.iloc[:, -1].unique()

    # Create a mapping dictionary to map each unique value to a number
    mapping_dict = {value: index for index, value in enumerate(unique_values)}

    # Replace the target column values with the corresponding numbers
    dataset.iloc[:, -1] = dataset.iloc[:, -1].map(mapping_dict)

    # Save the updated dataset to a new CSV file
    dataset.to_csv(output_file, index=False)

def normalize(input_file, output_file, target_column=None):
    # Read the dataset as a Pandas DataFrame
    dataset = pd.read_csv(input_file)

    # If target_column is not provided, use the last column as the target column
    if target_column is None:
        target_column = dataset.columns[-1]

    # Extract the column names before normalization
    column_names = dataset.columns

    # Separate the target column from the rest of the data
    target_data = dataset[target_column]
    features_data = dataset.drop(columns=[target_column])

    # Use MinMaxScaler to normalize the features data
    scaler = MinMaxScaler()
    normalized_features_data = scaler.fit_transform(features_data)

    # Concatenate the normalized features data with the target data
    normalized_df = pd.concat([pd.DataFrame(normalized_features_data, columns=features_data.columns), target_data], axis=1)

    # Save the normalized DataFrame to a CSV file
    normalized_df.to_csv(output_file, index=False)

def add_impulse_noise(input_file, output_file,target_column=None,  noise_level=0.1):
    """
    Add impulse noise to the input data.

    Parameters:
    data (DataFrame): Input data as pandas DataFrame.
    noise_level (float): Level of impulse noise to add.

    Returns:
    DataFrame: Data with added impulse noise.
    """
    
    # Read the dataset as a Pandas DataFrame
    data = pd.read_csv(input_file)
    
    if target_column is None:
        target_column = data.columns[-1]
    # Separate the target column from the rest of the data
    target_data = data[target_column]
    features_data = data.drop(columns=[target_column])

    noisy_data = features_data.copy()
    num_rows, num_cols = noisy_data.shape

    # Determine the number of cells to corrupt based on noise level
    num_corrupt_cells = int(noise_level * num_rows * num_cols)

    # Randomly select cells to corrupt
    indices_to_corrupt = np.random.choice(range(num_rows * num_cols), num_corrupt_cells, replace=False)
    rows_to_corrupt = indices_to_corrupt // num_cols
    cols_to_corrupt = indices_to_corrupt % num_cols

    # Add impulse noise by flipping the values of selected cells
    for row, col in zip(rows_to_corrupt, cols_to_corrupt):
        noisy_data.iloc[row, col] = np.random.choice([0, 1])
    # Concatenate the normalized features data with the target data
    normalized_df = pd.concat([pd.DataFrame(noisy_data, columns=features_data.columns), target_data], axis=1)

    # Save the normalized DataFrame to a CSV file
    normalized_df.to_csv(output_file, index=False)



# Example usage:
# input_filename = "data/Regresssion/sulfur/sulfur.csv"
# output_filename = "data/Regresssion/sulfur/sulfur_normMM.csv"

# nominal_encode(input_filename, output_filename)

folder_path = "/Users/vikramadipudi/Desktop/Thesis_research/Workspace/data/Clustering/test_sets"
output_folder_path = "/Users/vikramadipudi/Desktop/Thesis_research/Workspace/data/Clustering/test_sets"
# Iterate through files in the folder
noise_levels = [0.2]
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        name, _ = os.path.splitext(filename)

        input_file = os.path.join(folder_path, filename)
        nominal_encode(input_file, input_file)

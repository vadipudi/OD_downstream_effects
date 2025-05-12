import sys
sys.path.append('/Users/vikramadipudi/Desktop/Thesis_research/Workspace')
import os
import inspect
import types
import logging

import time


import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt


from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.outlier_rem import iforest, LOF, kNN, pca, control, kde, mahalanobis,random
import regression_models

# Define the dataset path
dataset_folder = '/Users/vikramadipudi/Desktop/Thesis_research/Workspace/data/Regresssion/minmax_data_noise'
results_folder = '/Users/vikramadipudi/Desktop/Thesis_research/Workspace/results/regression'
outlier_folder = '/Users/vikramadipudi/Desktop/Thesis_research/Workspace/outlier_data/regression'

reg_functions = [o[1] for o in inspect.getmembers(regression_models) if inspect.isfunction(o[1])]





# Define the list of functions
outlier_functions = [random, iforest, LOF, kNN, pca, control, kde, mahalanobis]


# Define the lists of parameters
train_test_splits = np.arange(400,1400,200)
threshold_vals = np.arange(0.05, 0.4, 0.05)
parA_vals = np.arange(2,7,1)
parB_vals = ['Min', 'Small', 'Med', 'Large', 'Max']


# train_test_splits = np.arange(800,900,100)
# threshold_vals = np.arange(0.05, 0.1, 0.05)
# parA_vals = np.arange(4,5,1)
# parB_vals = ['Med']


predictions_arr = np.empty((len(os.listdir(dataset_folder)),len(train_test_splits), len(outlier_functions), len(reg_functions), len(threshold_vals)), dtype=object)
results_df = pd.DataFrame(
    columns=[
        'Dataset', 
        'Split_Val', 
        'Outlier_Function', 
        'Reg Function',
        'Threshold_Val', 
        'ParA_Val',
        'ParB_Val',
        'Outliers Removed Pct',
        'rMSE', 
        'R_squared',
        'Adjusted_R_squared', 
        'MAE',
        'Max Error'])
actual_values = np.empty((len(os.listdir(dataset_folder)),len(train_test_splits)), dtype=object)

dataset_folders = os.listdir(dataset_folder)

# Iterate through the files in the dataset folder
for file_count, file in enumerate(dataset_folders):
    if not file.endswith('.csv'):
        continue
    print(file)
    filename_without_extension = os.path.splitext(file)[0]
    # Get the full file path
    file_path = os.path.join(dataset_folder, file)

    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    
    outliers_df = np.empty(len(train_test_splits), dtype=object)
    
                



    for split_val_count, split_val in enumerate(train_test_splits):
        print(split_val)
        if split_val>=1:
            train_ratio = 1- split_val/features.shape[0]
        else:
            train_ratio = 1- split_val
        if split_val >= features.shape[0]:
            continue
        # Split the data into training and testing sets
        #TODO maybe change test_target to a numpy array
        train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=train_ratio, random_state=1)
        outliers_df[split_val_count] = pd.DataFrame(index=train_features.index, columns=[func.__name__ for func in outlier_functions])

        train_data = train_features
        
        actual_values[file_count, split_val_count] = test_target
        
        # Normalize the train_target variable
        # scaler = MinMaxScaler()
        
        # train_target_encoded = pd.get_dummies(train_target)
        # train_target_norm = scaler.fit_transform(train_target_encoded.values.reshape(-1, 1)).flatten()
        
        # train_data = pd.concat([train_features, train_target_encoded],axis =1)


        # Iterate through the list and call the respective function
        for outlier_function_count,outlier_function in enumerate(outlier_functions):
            
            

            print(f"{outlier_function.__name__} outlier detection, with {split_val} samples")
            

                
            for threshold_val_count,threshold in enumerate(threshold_vals) :
                print(f"Threshold: {threshold}")
                for a,parA_val in enumerate(parA_vals):
                    outlier_parameters = list(inspect.signature(outlier_function).parameters.keys())
                    
                    # skips the rest of the loop if OR algo doesnt require a A Parameter
                    if outlier_parameters[2] == 'trashA' and a>=1:
                        continue
                    
                    for b,parB_val in enumerate(parB_vals):
                        # skips the rest of the loop if OR algo doesnt require a B Parameter
                        if outlier_parameters[3] == 'trashB' and b>=1:
                            continue

                        # class-specific outlier detection
                        outlier_indices=np.array([]) 
                        error_count=0
                        
                        file_path = os.path.join(outlier_folder, f"outliers_{filename_without_extension}_{outlier_function.__name__}_{split_val}.csv")
                        time_file_path = os.path.join(outlier_folder, f"time_{filename_without_extension}_{outlier_function.__name__}_{split_val}.csv")
                        if os.path.exists(file_path):
                            # print(f"{file_path}  exists")
                            outliers = pd.read_csv(file_path)
                            time_csv = pd.read_csv(time_file_path)
                            if f'{threshold}_{parA_val}_{parB_val}' in outliers.columns:
                                is_outlier = outliers[f'{threshold}_{parA_val}_{parB_val}'].values.astype(bool)
                                
                                elapsed_time = time_csv[f'{threshold}_{parA_val}_{parB_val}'].values[0]
                            else:
                                # print(f"{threshold}_{parA_val}_{parB_val} does not exist")
                                    
                                try:
                                    start_time = time.time()
                                    is_outlier = outlier_function(train_data,threshold,parA_val, parB_val)
                                    elapsed_time = time.time() - start_time
                                except Exception as e:
                                    logging.error(f"An error occurred: {e}")
                                    print("Error in ",[file_count, split_val_count, outlier_function_count, a,b] )
                                    is_outlier =  np.full(len(train_data), False)
                                    continue
                                
                                outliers[f'{threshold}_{parA_val}_{parB_val}'] = is_outlier.astype(int)
                                outliers.to_csv(file_path, index=False)
                                
                                time_csv[f'{threshold}_{parA_val}_{parB_val}'] = elapsed_time
                                time_csv.to_csv(time_file_path, index=False)
                        else:
                            # print(f"{file_path} does not exist")
                            try:
                                start_time = time.time()
                                is_outlier = outlier_function(train_data,threshold,parA_val, parB_val)
                                elapsed_time = time.time() - start_time
                            except Exception as e:
                                logging.error(f"An error occurred: {e}")
                                print("Error in ",[file_count, split_val_count, outlier_function_count, a,b] )
                                is_outlier =  np.full(len(train_data), False)
                                continue
                            
                            outliers = pd.DataFrame(index=train_features.index)
                            outliers[f'{threshold}_{parA_val}_{parB_val}'] = is_outlier.astype(int)
                            outliers.to_csv(file_path, index=False)
                            
                            time_csv = pd.DataFrame(index=train_features.index)
                            time_csv[f'{threshold}_{parA_val}_{parB_val}'] = elapsed_time
                            time_csv.to_csv(time_file_path, index=False)

                        # #non-class-specific OD
                        # is_outlier = outlier_function(train_data,threshold=threshold)
                        
                        inlier_features = train_features[~is_outlier]
                        inlier_target = train_target[~is_outlier]
                        inliers = inlier_features.assign(target=inlier_target)
                        outliers = train_features.assign(target= train_target[is_outlier])

                        outliers_df[split_val_count][outlier_function.__name__] = is_outlier
                        
                        predicted_contamination=np.sum(is_outlier)/len(train_features)

                        for reg_count,regression in enumerate(reg_functions):


                            #TODO build datastructure for results
                            try:
                                predictions = regression(inlier_features, test_features, inlier_target)
                                predictions_arr[file_count, split_val_count, outlier_function_count,  reg_count, threshold_val_count] = predictions
                                results_df.loc[len(results_df)] = {
                                    'Dataset': filename_without_extension, 
                                    'Split_Val': split_val, 
                                    'Outlier_Function': outlier_function.__name__, 
                                    'Threshold_Val': threshold,
                                    'Reg Function': regression.__name__, 
                                    'ParA_Val': parA_val,
                                    'ParB_Val': parB_val,
                                    'Outliers Removed Pct': predicted_contamination,
                                    'rMSE': root_mean_squared_error(test_target, predictions),
                                    'R_squared':r2_score(test_target, predictions),
                                    'Adjusted_R_squared':1-(1-r2_score(test_target, predictions))*(len(test_target)-1)/(len(test_target)-test_features.shape[1]-1), 
                                    'MAE': mean_absolute_error(test_target, predictions),
                                    'Max Error': max_error(test_target, predictions),
                                    'Elapsed Time': elapsed_time,
                                }

                            except Exception as e:
                                logging.error(f"An error occurred: {e}")
                                print("Error in ",[file_count, split_val_count, outlier_function_count, reg_count] )
                                
    # results_df.to_csv(os.path.join(results_folder,'results_test22.csv'), index=False)
    
    
    # # Create separate Excel files for each data set
    # dataset_file_path = os.path.join(results_folder,f'outliers/{filename_without_extension}_outliers.xlsx')
    # with pd.ExcelWriter(dataset_file_path) as writer:
    #     # Create separate sheets for each split_val
    #     for i,df in enumerate(outliers_df):
    #         sheet_name = f"Split_{train_test_splits[i]}"
    #         df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    
results_df.to_csv(os.path.join(results_folder,'results_test_mahala.csv'), index=False)


print(2)





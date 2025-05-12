import sys
sys.path.append('/Users/vikramadipudi/Desktop/Thesis_research/Workspace')
import os
import inspect
import types
import logging
import traceback
import time


import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, v_measure_score, adjusted_rand_score, rand_score, completeness_score, homogeneity_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.outlier_rem import iforest, LOF, kNN, pca, control, kde, mahalanobis, random
from clust_models import k_means_clustering,gaussian_mixture_clustering, agglomerative_clustering, dbscan_clustering, hdbscan_clustering

# Define the dataset path
dataset_folder = 'data/Clustering/kde_fix'
results_folder = '/Users/vikramadipudi/Desktop/Thesis_research/Workspace/results/clustering'
outlier_folder = '/Users/vikramadipudi/Desktop/Thesis_research/Workspace/outlier_data/clustering'

# clust_functions = [o[1] for o in inspect.getmembers(clust_models) if inspect.isfunction(o[1])]

clust_functions = [k_means_clustering,gaussian_mixture_clustering, agglomerative_clustering, dbscan_clustering, hdbscan_clustering]



# Define the list of functions
outlier_functions = [kde, iforest, LOF, kNN, pca, control, mahalanobis, random]


# Define the lists of parameters
threshold_vals = np.arange(0.05, 0.4, 0.05)
parA_vals = np.arange(2,7,1)
parB_vals = ['Min', 'Small', 'Med', 'Large', 'Max']
parB_vals = ['Med']
# train_test_splits = np.arange(800,900,100)
# threshold_vals = np.arange(0.05, 0.1, 0.05)
# parA_vals = np.arange(4,5,1)
# parB_vals = ['Med']


predictions_arr = np.empty((len(os.listdir(dataset_folder)), len(outlier_functions), len(clust_functions), len(threshold_vals)), dtype=object)
results_df = pd.DataFrame(
    columns=[
        'Dataset', 
        'Split_Val', 
        'Outlier_Function', 
        'Class_Function',
        'Threshold_Val', 
        'ParA_Val',
        'ParB_Val',
        'Outliers Removed Pct',
        'Silhouette_score',
        'Calinski_Harabasz_score',
        'Davies_Bouldin_score',
        'V_measure_score',
        'Adjusted_Rand_score',
        'Rand_score',
        'Completeness_score',
        'Homogeneity_score',
        'Elapsed Time'])
actual_values = np.empty(len(os.listdir(dataset_folder)), dtype=object)


# Iterate through the files in the dataset folder
for file_count, file in enumerate(os.listdir(dataset_folder)):
    print(file)
    filename_without_extension = os.path.splitext(file)[0]
    # Get the full file path
    file_path = os.path.join(dataset_folder, file)

    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    
    
    # get all possible labels classes
    labels_arr = np.unique(labels)


    actual_values[file_count] = labels
    



    # Iterate through the list and call the respective function
    for outlier_function_count,outlier_function in enumerate(outlier_functions):
        
        

        print(f"{outlier_function.__name__} outlier detection, with {file} file")
        
        #TODO change clusterer outside of threshold in other files tooo

            
        for threshold_val_count,threshold in enumerate(threshold_vals) :
            print(f"Threshold: {threshold}")
            if outlier_function.__name__ == 'control' and threshold_val_count >=1:
                continue
            
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
                    
                    file_path = os.path.join(outlier_folder, f"outliers_{filename_without_extension}_{outlier_function.__name__}.csv")
                    time_file_path = os.path.join(outlier_folder, f"time_{filename_without_extension}_{outlier_function.__name__}.csv")
                    if os.path.exists(file_path):
                        # print(f"{file_path}  exists")
                        outliers = pd.read_csv(file_path)
                        time_csv = pd.read_csv(time_file_path)
                        if f'{threshold}_{parA_val}_{parB_val}' in outliers.columns:
                            is_outlier = outliers[f'{threshold}_{parA_val}_{parB_val}'].values.astype(bool)
                            
                            elapsed_time = time_csv[f'{threshold}_{parA_val}_{parB_val}'].values[0]
                        else:
                            # print(f"{threshold}_{parA_val}_{parB_val} does not exist")
                            
                            start_time = time.time()
                            is_outlier = outlier_function(features,threshold,parA_val, parB_val)
                            elapsed_time = time.time() - start_time
                            
                            
                            outliers[f'{threshold}_{parA_val}_{parB_val}'] = is_outlier.astype(int)
                            outliers.to_csv(file_path, index=False)
                            
                            time_csv[f'{threshold}_{parA_val}_{parB_val}'] = elapsed_time
                            time_csv.to_csv(time_file_path, index=False)
                    else:
                        
                        start_time = time.time()
                        is_outlier = outlier_function(features,threshold,parA_val, parB_val)
                        elapsed_time = time.time() - start_time
                        
                        
                        outliers = pd.DataFrame(index=features.index)
                        outliers[f'{threshold}_{parA_val}_{parB_val}'] = is_outlier.astype(int)
                        outliers.to_csv(file_path, index=False)
                        
                        time_csv = pd.DataFrame(index=[1])
                        time_csv[f'{threshold}_{parA_val}_{parB_val}'] = elapsed_time
                        time_csv.to_csv(time_file_path, index=False)

                    # #non-class-specific OD
                    # is_outlier = outlier_function(train_data,threshold=threshold)
                    
                    inlier_features = features[~is_outlier]
                    inlier_labels = labels[~is_outlier]
                    
                    predicted_contamination=np.sum(is_outlier)/len(features)

                    for clusterer_count,clusterer in enumerate(clust_functions):


                        #TODO build datastructure for results
                        try:
                            predictions = clusterer(inlier_features,num_clusters=len(labels_arr))
                            predictions_arr[file_count, outlier_function_count,  clusterer_count, threshold_val_count] = predictions
                            results_df.loc[len(results_df)] = {
                                'Dataset': filename_without_extension,
                                'Outlier_Function': outlier_function.__name__, 
                                'Threshold_Val': threshold,
                                'Class_Function': clusterer.__name__, 
                                'ParA_Val': parA_val,
                                'ParB_Val': parB_val,
                                'Outliers Removed Pct': predicted_contamination,
                                'Silhouette_score': silhouette_score(inlier_features,predictions),
                                'Calinski_Harabasz_score': calinski_harabasz_score(inlier_features,predictions),
                                'Davies_Bouldin_score': davies_bouldin_score(inlier_features,predictions),
                                'V_measure_score': v_measure_score(inlier_labels,predictions),
                                'Adjusted_Rand_score': adjusted_rand_score(inlier_labels,predictions),
                                'Rand_score': rand_score(inlier_labels,predictions),
                                'Completeness_score': completeness_score(inlier_labels,predictions),
                                'Homogeneity_score': homogeneity_score(inlier_labels,predictions),
                                'Elapsed Time': elapsed_time,
                            }

                        except Exception as e:
                            logging.error(f"An error occurred: {e}")
                            print("Error in ",[file_count, outlier_function_count, clusterer_count] )
                            traceback.print_exc()
                                
    # results_df.to_csv(os.path.join(results_folder,'results_test3.csv'), index=False)

results_df.to_csv(os.path.join(results_folder,'results_test_kde_covnoise.csv'), index=False)


print(2)
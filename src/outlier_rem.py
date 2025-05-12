import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors,  LocalOutlierFactor, KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.models import Model



def control(data,threshold,trashA, trashB):
    """
    Control function. returns no outliers

    Parameters:
    - data: The input dataset (numpy array or pandas DataFrame).

    Returns:
    - Arrray of all false values.
    """
    is_outlier = np.full(len(data), False)
    return is_outlier

def random(data,threshold, trashA, trashB):
    """
    Remove outliers from a dataset using random sampling.

    Parameters:
    - data: The input dataset (numpy array or pandas DataFrame).
    - threshold: Threshold multiplier for outlier detection.
    - trashA: Placeholder parameter.
    - trashB: Placeholder parameter.

    Returns:
    - Indicies of outliers.
    """
    if isinstance(data, np.ndarray):
        X = data
    elif 'pandas' in str(type(data)):
        X = data.values
    else:
        raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

    # Identify and remove outliers
    is_outlier = np.full(len(data), False)
    
    is_outlier[np.random.choice(len(data), int(threshold * len(data)), replace=False)] = True

    return is_outlier

def mahalanobis(data, threshold=1, trashA=None, trashB=None):
    """
    Remove outliers from a dataset using Mahalanobis distance.

    Parameters:
    - data: The input dataset (numpy array or pandas DataFrame).
    - threshold: Threshold multiplier for outlier detection.
    - trashA: Placeholder parameter.
    - trashB: Placeholder parameter.

    Returns:
    - Indicies of outliers.
    """
    if isinstance(data, np.ndarray):
        X = data
    elif 'pandas' in str(type(data)):
        X = data.values
    else:
        raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

    # Calculate the mean and covariance matrix of the data
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)

    # Calculate the Mahalanobis distance for each data point
    mahalanobis_dist = np.sqrt(np.sum(np.square(np.dot(X - mean, np.linalg.pinv(cov))), axis=1))

    # Set the threshold for outlier removal

    outlier_threshold = np.percentile(mahalanobis_dist, 100 * (1- threshold))
    # Identify and remove outliers
    is_outlier = mahalanobis_dist > outlier_threshold
    
    # Identify and remove outliers
    is_outlier = mahalanobis_dist > outlier_threshold

    return is_outlier
def kNN(data, threshold=1.5, n_neighbors=5, trashB=None):
    """
    Remove outliers from a dataset using K-Nearest Neighbors.

    Parameters:
    - data: The input dataset (numpy array or pandas DataFrame).
    - n_neighbors: Number of neighbors to consider for outlier detection.
    - threshold: Threshold multiplier for outlier detection.
    TODO make threshold default to getting the best threshold for the dataset
    Returns:
    - Indicies of outliers.
    """
    if isinstance(data, np.ndarray):
        X = data
    elif 'pandas' in str(type(data)):
        X = data.values
    else:
        raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

    # Fit K-NN model
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(X)

    # Compute distances to k-neighbors for each data point
    distances, _ = knn_model.kneighbors(X)

    # Calculate median distance to k-neighbors for each data point
    median_distances = np.median(distances, axis=1)

    # Set the threshold for outlier removal
    outlier_threshold = threshold * np.median(median_distances)
    outlier_threshold = np.percentile(median_distances, 100 * (1- threshold))
    # Identify and remove outliers
    is_outlier = median_distances > outlier_threshold
    # cleaned_data = data[~is_outlier]
    # outliers = data[is_outlier]
    return is_outlier




def LOF(data, threshold = 0, n_neighbors=5, trashB=None):
    
    # TODO maybe choose contamination

    if isinstance(data, np.ndarray):
        X = data
    elif 'pandas' in str(type(data)):
        X = data.values
    else:
        raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

    lof_model=LocalOutlierFactor(n_neighbors=n_neighbors, contamination = 'auto')
    lof_model.fit_predict(X)
    outlier_scores = lof_model.negative_outlier_factor_

    # The negative LOF scores indicate outliers
    # cleaned_data = data[outlier_scores > 0]
    # outliers = data[outlier_scores <= 0]
    outlier_threshold = np.percentile(outlier_scores, 100 * threshold)
    return outlier_scores <= outlier_threshold

def iforest(data,threshold = 0, n_estimators=1, max_features="Min"):
    """
    Perform isolation forest outlier detection.

    Parameters:
    - data: The input dataset (numpy array or pandas DataFrame).
    - n_estimators: Number of isolation forest estimators.
    - max_samples: Maximum number of samples to consider when fitting each tree.
    - max_features: Maximum number of features to consider when fitting each tree.
    - bootstrap: Whether to bootstrap samples when fitting each tree.
    - random_state: Random state for bootstrapping and sampling.

    Returns:
    - Indicies of outliers.
    - Indices of outlier samples.
    """
    if isinstance(data, np.ndarray):
        X = data
    elif 'pandas' in str(type(data)):
        X = data.values
    else:
        raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

    switcher = {
        'Min': 1,
        'Small': .25,
        'Med':.5,
        'Large': .75,
        'Max':  X.shape[1]
    }
    
    n_estimators = n_estimators *25
    
    # Fit isolation forest model
    iforest_model = IsolationForest(n_estimators=n_estimators, max_samples=0.5, max_features=switcher.get(max_features), bootstrap=True, random_state=None)
    iforest_model.fit(X)

    # Predict outlier scores
    outlier_scores = iforest_model.decision_function(X)

    # The negative outlier scores indicate inliers

    # cleaned_data = data[outlier_scores > 0]
    # outliers = data[outlier_scores <= 0]
    outlier_threshold = np.percentile(outlier_scores, 100 *threshold)
    # Identify and remove outliers
    is_outlier = outlier_scores <= outlier_threshold
    return is_outlier

def kde(data, threshold = 1, trashA=None, trashB=None):
    
    if isinstance(data, np.ndarray):
        X = data
    elif 'pandas' in str(type(data)):
        X = data.values
    else:
        raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

    # Fit KDE model
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1)  # You can adjust bandwidth based on your dataset
    kde_model.fit(X)

    # Estimate density for each data point
    log_density = kde_model.score_samples(X)  # Log density values

    # Set threshold for outlier detection
    # threshold = np.percentile(log_density, 5)  # Example: 5th percentile as threshold
    outlier_threshold = np.percentile(log_density, 100 *threshold)
    # Identify and remove outliers
    is_outlier = log_density <= outlier_threshold
    return is_outlier

def pca(data,threshold = 1, n_neighbors = 3, n_components='Med'):
    """
    Perform principal component analysis (PCA) on the input dataset.

    Parameters:
    - data: The input dataset (numpy array or pandas DataFrame).
    - n_components: Number of principal components to return. if -1 then default value
    Returns:
    - The principal components of the input dataset.
    """
    if isinstance(data, np.ndarray):
        X = data
    elif 'pandas' in str(type(data)):
        X = data.values
    else:
        raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

    switcher = {
        'Min': 1,
        'Small': round(data.shape[1] * 0.25),
        'Med': round(data.shape[1] * 0.5),
        'Large': round(data.shape[1] * 0.75),
        'Max':  data.shape[1]-1
    }
    
    # Fit PCA
    pca = PCA(n_components=switcher.get(n_components))
    X_pca = pca.fit_transform(X)

    # # Finding outliers using EllipticEnvelope
    # envelope = EllipticEnvelope(contamination=0.2)
    # envelope.fit(X_pca)
    # outliers = envelope.predict(X_pca)
    # is_outlier=np.where(outliers==-1,True,False)

    # Finding outliers using kNN
    is_outlier = kNN(X_pca,n_neighbors=n_neighbors,threshold=threshold)

    
    return is_outlier

# def ae(data, threshold = 1):
#     """
#     Perform autoencoder outlier detection.

#     Parameters:
#     - data: The input dataset (numpy array or pandas DataFrame).
#     - n_components: Number of principal components to return. if -1 then default value
#     Returns:
#     - The principal components of the input dataset.
#     """
#     if isinstance(data, np.ndarray):
#         X = data
#     elif 'pandas' in str(type(data)):
#         X = data.values
#     else:
#         raise ValueError("Unsupported data type. Use numpy array or pandas DataFrame.")

#     switcher = {
#         'Min': 1,
#         'Small': round(data.shape[1] * 0.25),
#         'Med': round(data.shape[1] * 0.5),
#         'Large': round(data.shape[1] * 0.75),
#         'Max':  data.shape[1]-1
#     }
    
    
#     # Define an autoencoder architecture
#     input_dim = 784  # Example: MNIST data with 28x28 images
#     encoding_dim = 32
#     input_layer = Input(shape=(input_dim,))
#     encoded = Dense(encoding_dim, activation='relu')(input_layer)
#     decoded = Dense(input_dim, activation='sigmoid')(encoded)
#     autoencoder = Model(input_layer, decoded)

#     # Compile the autoencoder
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#     # Load and preprocess the dataset (e.g., MNIST)
    
#     X_train = X

#     # Train the autoencoder on normal data
#     autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True)

#     # Use the trained autoencoder for outlier detection
#     decoded = autoencoder.predict(X_train)
#     mse = np.mean(np.square(X_train - decoded), axis=1)
#     threshold = np.percentile(mse, 95)  # Set threshold at the 95th percentile of reconstruction errors

#     # Identify outliers based on reconstruction error
#     is_outlier = mse > threshold


    
#     return is_outlier


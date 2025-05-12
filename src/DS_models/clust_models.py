from sklearn.cluster import KMeans, DBSCAN,HDBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture


def k_means_clustering(dataset, num_clusters):
    # Create a KMeans object with the specified number of clusters
    model = KMeans(n_clusters=num_clusters)
    
    # Fit the dataset to the KMeans model and obtain the cluster labels
    labels = model.fit_predict(dataset)
    
    return labels

# TODO fix with all the parameters
def gaussian_mixture_clustering(dataset, num_clusters):
    model = GaussianMixture(n_components=num_clusters)
    labels = model.fit_predict(dataset)
    return labels


def agglomerative_clustering(dataset, num_clusters):
    model = AgglomerativeClustering(n_clusters=num_clusters)
    labels = model.fit_predict(dataset)
    return labels


def meanshift_clustering(dataset, num_clusters= None,  bandwidth = None):
    model = MeanShift(bandwidth=bandwidth)
    labels = model.fit_predict(dataset)
    return labels


def dbscan_clustering(dataset, num_clusters= None, eps=0.25, min_samples = 5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(dataset)
    return labels

def hdbscan_clustering(dataset, num_clusters= None,min_cluster_size=5):
    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(dataset)
    return labels
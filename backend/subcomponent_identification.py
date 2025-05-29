from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering


def compute_cosine_distance(X):
    return pairwise_distances(X, metric='cosine')

def cluster_reqs(distance_matrix, threshold=0.60):
    model = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - threshold
    )
    labels = model.fit_predict(distance_matrix)
    return labels






   

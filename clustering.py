import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from collections import Counter


def do_sc(data, n_clusters=2, seed=None):
    # gamma = 0.01, 0.1, 1
    sc = SpectralClustering(n_clusters=n_clusters, gamma=0.1, assign_labels="discretize")
    sc.fit(data)
    labels = sc.labels_
    return labels


def do_mb_kmeans(data, n_clusters, batch=64, seed=None):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch, random_state=seed).fit(data)
    cluster_info = kmeans.labels_
    cluster_size_map = Counter(cluster_info)
    return cluster_info, cluster_size_map

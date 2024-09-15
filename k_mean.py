from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def k_means(df): 
    df = df.iloc[1:, ]
    df = df.iloc[:, :-1]
    
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
    df_pca = pca.fit_transform(df)

    # Applying k-means clustering
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(df_pca)

    # Getting the cluster centers and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    print("Inertia:", kmeans.inertia_)

    return kmeans, df_pca, pca
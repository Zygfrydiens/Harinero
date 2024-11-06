import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from ..features.encoders.feature_encoding import encode_feature_names


def perform_kmeans_clustering(df, features, weights=None, n_clusters=3):
    """
    Perform K-means clustering on the given DataFrame using the specified features.

    :param df: pandas DataFrame containing the data
    :param features: List of feature names to use for clustering
    :param weights: List of weights to apply to the features
    :param n_clusters: Number of clusters to form
    :return: A new DataFrame with 'song_id' and 'cluster' columns, scaled data, and clusters
    """
    # Extract and scale the relevant features
    data = df[features].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply weights if provided
    if weights:
        data_scaled = data_scaled * weights

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Create a new DataFrame with song_id and cluster columns
    result_df = df.loc[data.index, ['song_id']].copy()
    result_df['cluster'] = clusters

    return result_df, data_scaled, clusters


def prepare_plot_parameters(features, weights, n_clusters, output_folder, output_file=None):
    """
    Prepare a dictionary with the necessary parameters for plotting PCA.

    :param features: List of feature names to use for PCA
    :param weights: List of weights to apply to the features (used for filename)
    :param n_clusters: Number of clusters
    :param output_folder: Folder to save the output plot image
    :param output_file: Name of the output plot image file (if None, a name will be generated)
    :return: Dictionary with plotting parameters
    """
    if output_file is None:
        encoded_features = encode_feature_names(features)
        encoded_weights = [str(w).replace('.', '') for w in weights] if weights else ['1' for _ in features]
        feature_part = '_'.join(encoded_features)
        weight_part = '_'.join(encoded_weights)
        output_file = f"PCA_{n_clusters}_clusters_{feature_part}_weights_{weight_part}.png"

    return {
        'features': features,
        'weights': weights,
        'n_clusters': n_clusters,
        'output_folder': output_folder,
        'output_file': output_file
    }


def plot_pca(df, cluster_df, params, cluster_column='cluster'):
    """
    Perform PCA and plot the first two principal components, colored by cluster assignments.

    :param df: pandas DataFrame containing the data
    :param cluster_df: DataFrame containing the cluster assignments
    :param params: Dictionary containing plotting parameters
    :param cluster_column: Name of the column containing cluster assignments
    """
    features = params['features']
    weights = params['weights']
    n_clusters = params['n_clusters']
    output_folder = params['output_folder']
    output_file = params['output_file']

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract the relevant features and cluster assignments
    data = df[features].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    clusters = cluster_df.loc[data.index, cluster_column]

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)

    # Create a scatter plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    plt.title('PCA of Selected Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Add legend
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles, labels, title="Clusters")

    # Save the plot
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path)
    plt.close()

    print(f"PCA plot saved as {output_path}")


def evaluate_clustering_performance(data, clusters):
    """
    Evaluate the clustering performance using the silhouette score.

    :param data: Scaled feature data
    :param clusters: Cluster assignments
    :return: Silhouette score
    """
    if len(set(clusters)) > 1:
        score = silhouette_score(data, clusters)
    else:
        score = -1
    return score

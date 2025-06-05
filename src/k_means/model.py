# src/k_means/model.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

_base_output_dir = os.path.dirname(os.path.abspath(__file__))
_results_dir = os.path.join(_base_output_dir, 'results')
_artifacts_dir = os.path.join(_base_output_dir, 'artifacts')


def _ensure_output_dirs_exist():
    os.makedirs(_results_dir, exist_ok=True)
    os.makedirs(_artifacts_dir, exist_ok=True)

def find_optimal_k(X_scaled_df, k_range=range(2, 11)):
    sse = []
    print(f"[INFO] Finding optimal k using Elbow Method in range {k_range.start} to {k_range.stop - 1}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled_df)
        sse.append(kmeans.inertia_)

    # Find elbow point using maximum distance to line method
    x = np.array(list(k_range))
    y = np.array(sse)
    a = np.array([x[0], y[0]])
    b = np.array([x[-1], y[-1]])

    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        d = np.linalg.norm(np.cross(b - a, a - p)) / np.linalg.norm(b - a)
        distances.append(d)

    elbow_k = x[np.argmax(distances)]
    print(f"[INFO] Elbow point detected at k = {elbow_k}")

    # Plot elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, sse, marker='o')
    plt.axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow at k={elbow_k}')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Inertia)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(_results_dir, 'elbow_detected.png'))
    plt.show()

    return elbow_k, sse

def train_kmeans_model(X_scaled_df, optimal_k):
    print(f"[INFO] Training KMeans with k = {optimal_k}...")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled_df)
    print("[INFO] KMeans training complete.")
    return kmeans_model

def characterize_clusters(df_clustered, features_to_cluster):
    _ensure_output_dirs_exist()

    print("\n[INFO] Characterizing cluster profiles:")
    cluster_profiles = df_clustered.groupby('cluster_label')[features_to_cluster].mean()
    print(cluster_profiles.round(2))

    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_profiles, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title(f'Cluster Feature Means ({df_clustered["cluster_label"].nunique()} clusters)')
    plt.xlabel('Features')
    plt.ylabel('Cluster Label')
    plt.savefig(os.path.join(_results_dir, 'cluster_profiles_heatmap.png'))
    plt.show()

def save_model_artifacts(scaler, kmeans_model):
    _ensure_output_dirs_exist()
    print(f"[INFO] Saving model artifacts to {_artifacts_dir}...")
    joblib.dump(scaler, os.path.join(_artifacts_dir, 'scaler.pkl'))
    joblib.dump(kmeans_model, os.path.join(_artifacts_dir, 'kmeans_model.pkl'))
    print("[INFO] Artifacts saved.")

def plot_clusters_2d(X_scaled_df, labels, filename='cluster_plot.png'):
    _ensure_output_dirs_exist()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled_df)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=30)
    plt.title("K-Means Clusters (PCA-reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster Label")
    plt.tight_layout()
    plt.savefig(os.path.join(_results_dir, filename))
    plt.show()

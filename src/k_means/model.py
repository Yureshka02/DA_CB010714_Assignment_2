
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Define the base directory for outputs relative to this file's location
# This resolves to CB010714-DA-2/src/k_means/
_base_output_dir = os.path.dirname(os.path.abspath(__file__))
_results_dir = os.path.join(_base_output_dir, 'results')
_artifacts_dir = os.path.join(_base_output_dir, 'artifacts')

def _ensure_output_dirs_exist():
    """Ensures the results and artifacts directories exist within k_means."""
    os.makedirs(_results_dir, exist_ok=True)
    os.makedirs(_artifacts_dir, exist_ok=True)

def find_optimal_k(X_scaled_df, k_range=range(2, 11)):
    """
    Determines the optimal number of clusters (k) using Elbow Method and Silhouette Score.
    Plots are saved to CB010714-DA-2/src/k_means/results/.
    """
    _ensure_output_dirs_exist() # Ensure directories exist before saving

    sse = []
    silhouette_scores = []

    print(f"Testing k from {k_range.start} to {k_range.stop - 1}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled_df)
        sse.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_scaled_df, kmeans.labels_))
        else:
            silhouette_scores.append(0)

    # Plot Elbow Method
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (Inertia)')
    plt.grid(True)
    plt.savefig(os.path.join(_results_dir, 'elbow_method.png'))


    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(_results_dir, 'silhouette_score.png'))
    plt.tight_layout()
    plt.show()

    return sse, silhouette_scores

def train_kmeans_model(X_scaled_df, optimal_k):
    """
    Trains the KMeans model with the specified optimal k.
    """
    print(f"Training KMeans model with k = {optimal_k}...")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled_df)
    print("KMeans model training complete.")
    return kmeans_model

def characterize_clusters(df_clustered, features_to_cluster):
    """
    Calculates and visualizes the mean profile of each cluster.
    Heatmap is saved to CB010714-DA-2/src/k_means/results/.
    Clustered data CSV is saved to CB010714-DA-2/src/k_means/results/.
    """
    _ensure_output_dirs_exist() # Ensure directories exist before saving

    print("\nCharacterizing Player Archetypes (Cluster Means):")
    cluster_profiles = df_clustered.groupby('cluster_label')[features_to_cluster].mean()
    print(cluster_profiles.round(2))

    # Visualize Cluster Profiles (Heatmap)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_profiles, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title(f'Average Feature Values for {df_clustered["cluster_label"].nunique()} Player Archetypes (Clusters)')
    plt.xlabel('Player Statistics')
    plt.ylabel('Cluster Label')
    plt.savefig(os.path.join(_results_dir, 'cluster_profiles_heatmap.png'))
    plt.show()

    # Display example players from each cluster
    print("\n--- Example Players from Each Archetype ---")
    for i in range(df_clustered['cluster_label'].nunique()):
        print(f"\nArchetype {i} (Cluster {i}):")
        print(df_clustered[df_clustered['cluster_label'] == i]
              [['player_name', 'season', 'pts', 'reb', 'ast', 'net_rating', 'ts_pct']].head(5).to_string())

    # Save the clustered DataFrame
    clustered_csv_path = os.path.join(_results_dir, 'clustered_player_data.csv')
    df_clustered.to_csv(clustered_csv_path, index=False)
    print(f"Clustered data saved to {clustered_csv_path}")

def save_model_artifacts(scaler, kmeans_model):
    """
    Saves the trained StandardScaler and KMeans model to CB010714-DA-2/src/k_means/artifacts/.
    """
    _ensure_output_dirs_exist() # Ensure directories exist before saving

    import joblib
    print(f"\nSaving model artifacts to {_artifacts_dir}...")
    joblib.dump(scaler, os.path.join(_artifacts_dir, 'scaler.pkl'))
    joblib.dump(kmeans_model, os.path.join(_artifacts_dir, 'kmeans_model.pkl'))
    print("Models saved.")
# Path: CB010714-DA-2/run_clustering.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.k_means.data import load_data, preprocess_data, get_features_for_clustering
from src.k_means.model import find_optimal_k, train_kmeans_model, characterize_clusters, save_model_artifacts

def main():
    # Directories for results and models are now managed internally by src/k_means/models.py
    # No need to define or create them here.

    print("--- Starting K-Means Clustering Pipeline ---")

    # 1. Load and Preprocess Data
    print("\nStep 1: Loading and Preprocessing Data...")
    raw_df = load_data() # Will load from data/raw/all_seasons.csv as configured in src/k_means/data.py
    processed_df = preprocess_data(raw_df.copy(), is_training=True)

    # 2. Data Preparation and Feature Selection for Clustering
    features_to_cluster = get_features_for_clustering()

    min_games_played = 41
    df_for_clustering = processed_df[processed_df['gp'] >= min_games_played].copy()

    X_cluster = df_for_clustering[features_to_cluster]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features_to_cluster, index=df_for_clustering.index)
    print(f"Data prepared for clustering. Shape: {X_scaled.shape}")

    # 3. Determining the Optimal Number of Clusters (k)
    print("\nStep 3: Determining Optimal Number of Clusters (k)...")
    sse, silhouette_scores = find_optimal_k(X_scaled_df)

    # --- Manual Decision Point ---
    # REVIEW THE GENERATED PLOTS (CB010714-DA-2/src/k_means/results/elbow_method.png and ...)
    # to choose the optimal_k. Then, update this variable.
    optimal_k = 5 # This is an EXAMPLE. Adjust based on your plot analysis.
    print(f"\nManual Decision: Chosen Optimal k = {optimal_k}")
    # --- End Manual Decision Point ---

    # 4. Applying K-Means Clustering with Optimal k
    print(f"\nStep 4: Applying K-Means Clustering with k = {optimal_k}...")
    kmeans_model = train_kmeans_model(X_scaled_df, optimal_k)
    
    df_for_clustering['cluster_label'] = kmeans_model.labels_
    print("Clustering complete. Cluster labels added to DataFrame.")

    # 5. Characterizing and Interpreting Clusters (Player Archetypes)
    print("\nStep 5: Characterizing Player Archetypes...")
    characterize_clusters(df_for_clustering, features_to_cluster)

    # 6. Save Model Artifacts (Scaler and KMeans Model)
    print("\nStep 6: Saving Model Artifacts...")
    save_model_artifacts(scaler, kmeans_model)

    print("\n--- K-Means Clustering Pipeline Complete ---")

if __name__ == "__main__":
    # Ensure all_seasons.csv is in the project root: CB010714-DA-2/data/raw/
    # The internal directories (results/ and artifacts/) within src/k_means/
    # will be created by the functions in src/k_means/models.py
    main()
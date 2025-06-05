# run_clustering.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.k_means.data import load_data, preprocess_data, get_features_for_clustering
from src.k_means.model import (
    find_optimal_k, train_kmeans_model,
    characterize_clusters, save_model_artifacts,
    plot_clusters_2d
)

def main():
    print("\n=== K-Means Clustering Pipeline Start ===")

    # Step 1: Load and preprocess data
    print("\n[STEP 1] Loading and preprocessing data...")
    raw_df = load_data()
    processed_df = preprocess_data(raw_df.copy(), is_training=True)

    # Step 2: Prepare data for clustering
    print("[STEP 2] Preparing data for clustering...")
    features_to_cluster = get_features_for_clustering()
    df_for_clustering = processed_df[processed_df['gp'] >= 41].copy()

    X_cluster = df_for_clustering[features_to_cluster]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features_to_cluster, index=df_for_clustering.index)
    print(f"[INFO] Clustering data shape: {X_scaled.shape}")

    # Step 3: Find optimal k and plot elbow
    print("\n[STEP 3] Determining optimal number of clusters (k)...")
    optimal_k, sse = find_optimal_k(X_scaled_df)

    # Step 4: Train KMeans
    print("\n[STEP 4] Training KMeans model...")
    kmeans_model = train_kmeans_model(X_scaled_df, optimal_k)
    df_for_clustering['cluster_label'] = kmeans_model.labels_

    # Step 5: Characterize clusters
    print("\n[STEP 5] Characterizing clusters...")
    characterize_clusters(df_for_clustering, features_to_cluster)

    # Step 6: Save artifacts
    print("\n[STEP 6] Saving model artifacts...")
    save_model_artifacts(scaler, kmeans_model)

    # Step 7: Optional - Plot clusters in 2D
    print("\n[STEP 7] Visualizing clusters in 2D...")
    plot_clusters_2d(X_scaled_df, kmeans_model.labels_)

    print("\n=== K-Means Clustering Pipeline Complete ===")

if __name__ == "__main__":
    main()

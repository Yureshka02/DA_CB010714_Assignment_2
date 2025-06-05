
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

    
    print("\n[STEP 1] Loading and preprocessing data...")
    raw_df = load_data()
    processed_df = preprocess_data(raw_df.copy(), is_training=True)

    
    print("[STEP 2] Preparing data for clustering...")
    features_to_cluster = get_features_for_clustering()
    df_for_clustering = processed_df[processed_df['gp'] >= 41].copy()

    X_cluster = df_for_clustering[features_to_cluster]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features_to_cluster, index=df_for_clustering.index)
    print(f"[INFO] Clustering data shape: {X_scaled.shape}")

    
    print("\n[STEP 3] Determining optimal number of clusters (k)...")
    optimal_k, sse = find_optimal_k(X_scaled_df)

   
    print("\n[STEP 4] Training KMeans model...")
    kmeans_model = train_kmeans_model(X_scaled_df, optimal_k)
    df_for_clustering['cluster_label'] = kmeans_model.labels_

    
    print("\n[STEP 5] Characterizing clusters...")
    characterize_clusters(df_for_clustering, features_to_cluster)

    
    print("\n[STEP 6] Saving model artifacts...")
    save_model_artifacts(scaler, kmeans_model)

    
    print("\n[STEP 7] Visualizing clusters in 2D...")
    plot_clusters_2d(X_scaled_df, kmeans_model.labels_)

    print("\n=== K-Means Clustering Pipeline Complete ===")

if __name__ == "__main__":
    main()

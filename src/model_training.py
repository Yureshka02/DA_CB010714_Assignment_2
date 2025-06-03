import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import argparse
import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your data processing functions from the local src package
from src.data_processing import load_data, preprocess_data, get_features_target

# Ensure MLflow connects to your local UI/tracking server
mlflow.set_tracking_uri("http://localhost:5000")

def train_model(training_seasons_str, test_season, n_estimators, random_state):
    """
    Trains a RandomForestRegressor model, evaluates it, and logs results to MLflow.
    """
    logging.info(f"Starting MLflow run for training seasons {training_seasons_str} and test season {test_season}...")

    # Convert training seasons string to list of integers
    training_seasons = [int(s) for s in training_seasons_str.split('-')]
    min_training_season = training_seasons[0]
    max_training_season = training_seasons[1]

    # Generate a unique model name
    unique_model_name = f"PlayerPerformanceModel_{min_training_season}_{max_training_season}_test_{test_season}"

    with mlflow.start_run(run_name=f"Training_{min_training_season}-{max_training_season}_Test_{test_season}"):
        mlflow.log_param("training_seasons", training_seasons_str)
        mlflow.log_param("test_season", test_season)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Load and preprocess data
        logging.info(f"Loading and preprocessing data for training seasons {min_training_season}-{max_training_season}...")
        all_data = load_data()

        all_data_processed = preprocess_data(all_data.copy(), is_training=False)

        train_df = all_data_processed[
            (all_data_processed['season'] >= min_training_season) & 
            (all_data_processed['season'] <= max_training_season)
        ].copy()
        test_df = all_data_processed[
            all_data_processed['season'] == test_season
        ].copy()

        train_df_final = preprocess_data(train_df, is_training=True)
        test_df_final = preprocess_data(test_df, is_training=False)

        X_train, y_train = get_features_target(train_df_final)
        X_test, y_test = get_features_target(test_df_final)

        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Test data shape: {X_test.shape}")

        # Train the model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate on known targets
        known_test_indices = y_test[y_test != -1].index
        y_test_known = y_test.loc[known_test_indices]
        predictions_known = pd.Series(predictions, index=y_test.index).loc[known_test_indices]

        mae = mean_absolute_error(y_test_known, predictions_known)
        r2 = r2_score(y_test_known, predictions_known)

        logging.info(f"Model MAE on test season {test_season}: {mae:.2f}")
        logging.info(f"Model R2 Score on test season {test_season}: {r2:.2f}")

        # Log metrics to MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log the model with a unique name
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=unique_model_name
        )

        logging.info(f"MLflow Run completed and model logged as: {unique_model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a player performance prediction model.")
    parser.add_argument("--training_seasons", type=str, default="1996-1999",
                        help="Range of seasons for training (e.g., '1996-1999')")
    parser.add_argument("--test_season", type=int, default=2000,
                        help="The season to test predictions against")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of estimators for RandomForestRegressor")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")

    args = parser.parse_args()

    train_model(args.training_seasons, args.test_season, args.n_estimators, args.random_state)

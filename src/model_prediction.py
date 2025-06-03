import pandas as pd
import mlflow
import mlflow.pyfunc
import argparse
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from src.data_processing import load_data, preprocess_data

# Ensure MLflow connects to your local UI/tracking server
mlflow.set_tracking_uri("http://localhost:5000")

def predict_performance(model_name, model_stage, input_season, output_file_path):
    """
    Loads a model from MLflow Model Registry and makes predictions for a specified season.
    """
    logging.info(f"Starting MLflow prediction run for model '{model_name}' (stage: {model_stage})...")

    with mlflow.start_run(run_name=f"Prediction_Season_{input_season}_Model_{model_stage}") as run:
        try:
            # Load the model
            model_uri = f"models:/{model_name}/{model_stage}"
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logging.info(f"Successfully loaded model from: {model_uri}")

            # --- DEBUGGING LOGS START ---
            logging.info(f"Current working directory: {os.getcwd()}")
            data_file_path = "data/raw/all_seasons.csv" # This is the relative path load_data uses
            logging.info(f"Attempting to load data from relative path: {data_file_path}")
            logging.info(f"Absolute path attempted: {os.path.abspath(data_file_path)}")
            # --- DEBUGGING LOGS END ---

            # Load and preprocess data for prediction
            logging.info(f"Loading data for prediction season: {input_season}...")
            
            # --- DEBUGGING LOGS START ---
            df_raw = load_data()
            logging.info(f"Raw DataFrame head after load_data():\n{df_raw.head()}")
            logging.info(f"Unique seasons in raw data after load_data(): {df_raw['season'].unique()}")
            # --- DEBUGGING LOGS END ---

            df_processed = preprocess_data(df_raw.copy(), is_training=False) # Use a copy to prevent modifying df_raw
            
            # --- DEBUGGING LOGS START ---
            logging.info(f"DataFrame head after preprocess_data():\n{df_processed.head()}")
            logging.info(f"Unique seasons in processed data after preprocess_data(): {df_processed['season'].unique()}")
            logging.info(f"Data type of 'season' column after preprocess_data(): {df_processed['season'].dtype}")
            # --- DEBUGGING LOGS END ---

            # Filter data for the specified prediction season
            season_data = df_processed[df_processed['season'] == input_season]
            
            # --- DEBUGGING LOGS START ---
            logging.info(f"Number of rows found for season {input_season} after filtering: {len(season_data)}")
            # --- DEBUGGING LOGS END ---

            if season_data.empty:
                logging.warning(f"No data found for season {input_season}. Cannot make predictions.")
                return # Exit if no data for the season

            
            features = [
                'age', 'player_height', 'player_weight', 'gp', 'ast', 'reb', 
                'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
            ]
            
            # Filter season_data to only include columns that are features and exist
            X_predict = season_data[features]
            
            # Make predictions
            predictions = loaded_model.predict(X_predict)
            
            # Add predictions to the DataFrame
            season_data['predicted_pts'] = predictions
            
            # Save results to CSV
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            season_data.to_csv(output_file_path, index=False)
            logging.info(f"Predictions saved to: {output_file_path}")

            # Log the output file as an MLflow artifact
            mlflow.log_artifact(output_file_path, "predictions")
            logging.info("Prediction results logged as MLflow artifact.")

        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}", exc_info=True)
            mlflow.log_param("error", str(e))
            sys.exit(1) # Exit with an error code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained MLflow model.")
    parser.add_argument("--model_name", type=str, default="PlayerPerformanceModel",
                        help="Name of the model in MLflow Model Registry (e.g., PlayerPerformanceModel)")
    parser.add_argument("--model_stage", type=str, default="None",
                        help="Stage of the model to use (e.g., 'Production', 'Staging', 'None' for latest)")
    parser.add_argument("--input_season", type=int, default=2023,
                        help="The season for which to make predictions (e.g., 2023)")
    parser.add_argument("--output_file_path", type=str, default="data/predictions/season_2023_predictions.csv",
                        help="Path to save the prediction results CSV file")
    
    args = parser.parse_args()

    predict_performance(
        args.model_name,
        args.model_stage,
        args.input_season,
        args.output_file_path
    )
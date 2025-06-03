import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from scipy.stats import linregress
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.data_processing import load_data, preprocess_data, get_features_target

mlflow.set_tracking_uri("http://localhost:5000")

def analyze_data(season_for_analysis):
    logging.info(f"Starting MLflow run for data analysis for season {season_for_analysis}...")

    with mlflow.start_run(run_name=f"Data_Analysis_Season_{season_for_analysis}"):
        mlflow.log_param("analysis_season", season_for_analysis)

        # --- Load and Preprocess ---
        all_data = load_data()
        processed_data = preprocess_data(all_data.copy(), is_training=True)
        analysis_season_data = processed_data[processed_data['season'] == season_for_analysis].copy()

        if analysis_season_data.empty:
            logging.warning(f"No data available for season {season_for_analysis}. Skipping analysis.")
            return

        X, y = get_features_target(analysis_season_data)
        X = X.fillna(X.median())

        # --- Decision Tree ---
        logging.info("Generating Decision Tree plot...")
        dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
        dt_model.fit(X, y)

        plt.figure(figsize=(25, 15))
        plot_tree(dt_model, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10)
        plt.title(f"Decision Tree for Player Performance (Season {season_for_analysis}, Max Depth 4)", fontsize=16)
        dt_plot_path = f"decision_tree_season_{season_for_analysis}.png"
        plt.savefig(dt_plot_path)
        plt.close()
        mlflow.log_artifact(dt_plot_path, "decision_trees")

        feature_importances = pd.Series(dt_model.feature_importances_, index=X.columns)
        mlflow.log_dict(feature_importances.sort_values(ascending=False).to_dict(), "feature_importances_dt.json")

        # --- Export Decision Tree Rules ---
        tree_rules = export_text(dt_model, feature_names=list(X.columns))
        logging.info("Top decision rules:\n" + tree_rules)
        with open("tree_rules.txt", "w") as f:
            f.write(tree_rules)
        mlflow.log_artifact("tree_rules.txt", artifact_path="decision_trees")

        # --- Storyline Analysis ---
        logging.info("Generating Player Performance Storyline plot...")
        processed_data_for_story = preprocess_data(all_data.copy(), is_training=False)
        processed_data_for_story['pts_display'] = processed_data_for_story['pts'].replace(-1, np.nan)
        processed_data_for_story['pts_display'] = processed_data_for_story['pts_display'].fillna(
            processed_data_for_story['pts_display'].mean()
        )

        avg_ppg_per_season = processed_data_for_story.groupby('season')['pts_display'].mean().reset_index()
        prominent_players = analysis_season_data[analysis_season_data['gp'] > 30].sort_values(by='pts', ascending=False)['player_name'].unique()
        player_to_spotlight = prominent_players[0] if len(prominent_players) > 0 else "Stephen Curry"

        player_data = processed_data_for_story[processed_data_for_story['player_name'] == player_to_spotlight].copy()
        player_ppg_per_season = player_data.groupby('season')['pts_display'].mean().reset_index()

        # --- Slope Trend Analysis ---
        if len(player_ppg_per_season) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(player_ppg_per_season['season'], player_ppg_per_season['pts_display'])
            mlflow.log_metric("ppg_trend_slope", slope)
            logging.info(f"PPG Trend Slope for {player_to_spotlight}: {slope:.2f}")
        else:
            slope = np.nan
            logging.info("Not enough data points for trend analysis.")

        # --- Standout Seasons vs League ---
        player_vs_avg = player_ppg_per_season.merge(avg_ppg_per_season, on="season", suffixes=("_player", "_league"))
        player_vs_avg["ppg_diff"] = player_vs_avg["pts_display_player"] - player_vs_avg["pts_display_league"]
        standout_seasons = player_vs_avg[player_vs_avg["ppg_diff"] > 5]
        if not standout_seasons.empty:
            standout_text = standout_seasons.to_string(index=False)
            with open("standout_seasons.txt", "w") as f:
                f.write(f"Standout Seasons for {player_to_spotlight}:\n{standout_text}")
            mlflow.log_artifact("standout_seasons.txt", "storyline_plots")
            logging.info(f"Logged standout seasons for {player_to_spotlight}")
        else:
            logging.info("No standout seasons found.")

        # --- Plot Storyline ---
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=avg_ppg_per_season, x='season', y='pts_display', marker='o', label='League Average PPG')
        sns.lineplot(data=player_ppg_per_season, x='season', y='pts_display', marker='s', label=f'{player_to_spotlight} PPG')
        plt.title(f'Evolution of Player Scoring: League Average vs. {player_to_spotlight}', fontsize=14)
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Points Per Game (PPG)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()

        storyline_plot_path = f"player_storyline_season_{season_for_analysis}.png"
        plt.savefig(storyline_plot_path)
        plt.close()
        mlflow.log_artifact(storyline_plot_path, "storyline_plots")

        logging.info(f"Data analysis run completed for season {season_for_analysis}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform data analysis and generate visualizations.")
    parser.add_argument("--season_for_analysis", type=int, default=2000,
                        help="The specific season to analyze data for.")
    args = parser.parse_args()
    analyze_data(args.season_for_analysis)

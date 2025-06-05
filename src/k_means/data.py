
import pandas as pd
import numpy as np
import os

def load_data():
    """
    Loads raw player data from a CSV file.
    Assumes 'all_seasons.csv' is in 'data/raw/' relative to the project root.
    """
    # **CHANGE**: Path adjusted to go three levels up to reach the project root
    # os.path.abspath(__file__) -> CB010714-DA-2/src/k_means/data.py
    # os.path.dirname(os.path.abspath(__file__)) -> CB010714-DA-2/src/k_means/
    # os.path.dirname(os.path.dirname(...)) -> CB010714-DA-2/src/
    # os.path.dirname(os.path.dirname(os.path.dirname(...))) -> CB010714-DA-2/ (project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root, 'data', 'raw', 'all_seasons.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found. Expected at: {file_path}")

    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, is_training=True):
    """
    Cleans and preprocesses the player data.
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'college' in df.columns:
        df = df.drop('college', axis=1)

    df['draft_round'] = df['draft_round'].replace('Undrafted', 0)
    df['draft_number'] = df['draft_number'].replace('Undrafted', 0)

    df['draft_round'] = pd.to_numeric(df['draft_round'], errors='coerce').fillna(0).astype(int)
    df['draft_number'] = pd.to_numeric(df['draft_number'], errors='coerce').fillna(int)

    def get_season_year(season_str):
        try:
            return int(str(season_str)[:4])
        except ValueError:
            return np.nan
    df['season'] = df['season'].apply(get_season_year)
    df['season'] = df['season'].fillna(df['season'].median()).astype(int)

    performance_metrics = [
        'ast', 'reb', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
    ]
    for col in performance_metrics:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if is_training:
        df = df[(df['pts'] != -1) & (~df['pts'].isna())].copy()

    return df

def get_features_for_clustering():
    """
    Returns the list of feature names to be used for K-Means clustering.
    """
    features = [
        'age',
        'player_height',
        'player_weight',
        'gp',
        'ast',
        'reb',
        'net_rating',
        'oreb_pct',
        'dreb_pct',
        'usg_pct',
        'ts_pct',
        'ast_pct'
    ]
    return features
import pandas as pd
import numpy as np

def load_data(file_path="data/raw/all_seasons.csv"):
    """Loads raw player data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, is_training=True): # <-- ENSURE 'is_training=True' IS HERE
    
    # Drop 'Unnamed: 0' and 'college' columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'college' in df.columns:
        df = df.drop('college', axis=1)

    # Handle 'Undrafted' in 'draft_round' and 'draft_number'
    # Replace 'Undrafted' string with 0
    df['draft_round'] = df['draft_round'].replace('Undrafted', 0)
    df['draft_number'] = df['draft_number'].replace('Undrafted', 0)

    # Convert 'draft_round' and 'draft_number' to integer type
    # Using errors='coerce' will turn non-convertible values into NaN, then fillna(0)
    df['draft_round'] = pd.to_numeric(df['draft_round'], errors='coerce').fillna(0).astype(int)
    df['draft_number'] = pd.to_numeric(df['draft_number'], errors='coerce').fillna(0).astype(int)

    #Extract season year from 'season' column (e.g., '1996-97' -> 1996)
    def get_season_year(season_str):
        try:
            return int(str(season_str)[:4])
        except ValueError:
            
            return np.nan
    df['season'] = df['season'].apply(get_season_year)
    
    
    df['season'] = df['season'].fillna(df['season'].median()).astype(int) 

    # Impute missing performance metrics (ast, reb, etc.) with 0 for new/anonymous players
    performance_metrics = [
        'ast', 'reb', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
    ]
    for col in performance_metrics:
        if col in df.columns: 
            df[col] = df[col].fillna(0) 

    
    if is_training: 
       
        df = df[(df['pts'] != -1) & (~df['pts'].isna())].copy()
    
    return df

def get_features_target(df):
    """Separates features and target variable based on all_seasons.csv columns."""
    # Features chosen based on your problem description and available columns in all_seasons.csv
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
    target = 'pts' # Using 'pts' (points) as the target variable for prediction

    X = df[features]
    y = df[target]
    return X, y
"""
Data Processor Module
Handles loading, cleaning, and preparing football match data for prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


class DataProcessor:
    """Process and prepare football match data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
        
    def load_csv_data(self, filepath):
        """Load match data from CSV file"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {len(self.data)} rows")
            return self.data
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return None
    
    def create_sample_data(self):
        """Create sample historical data for testing"""
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'home_team': np.random.choice(['MCI', 'LIV', 'MUN', 'CHE', 'ARS', 'BHA', 'FOR', 'WOL'], n_samples),
            'away_team': np.random.choice(['MCI', 'LIV', 'MUN', 'CHE', 'ARS', 'BHA', 'FOR', 'WOL'], n_samples),
            'home_goals': np.random.poisson(1.5, n_samples),
            'away_goals': np.random.poisson(1.2, n_samples),
            'home_shots': np.random.randint(8, 20, n_samples),
            'away_shots': np.random.randint(6, 18, n_samples),
            'home_possession': np.random.randint(35, 70, n_samples),
            'away_possession': np.random.randint(30, 65, n_samples),
            'home_passes': np.random.randint(300, 600, n_samples),
            'away_passes': np.random.randint(250, 550, n_samples),
        }
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def add_result_column(self, data=None):
        """Add result column: 1 (home win), 0 (draw), -1 (away win)"""
        if data is None:
            data = self.data
            
        data['result'] = data.apply(
            lambda row: 1 if row['home_goals'] > row['away_goals'] 
            else (-1 if row['home_goals'] < row['away_goals'] else 0),
            axis=1
        )
        return data
    
    def calculate_team_stats(self, data=None):
        """Calculate team statistics for features"""
        if data is None:
            data = self.data.copy()
        
        stats = {}
        teams = set(data['home_team'].unique()) | set(data['away_team'].unique())
        
        for team in teams:
            home_matches = data[data['home_team'] == team]
            away_matches = data[data['away_team'] == team]
            
            # Goals
            goals_for = home_matches['home_goals'].sum() + away_matches['away_goals'].sum()
            goals_against = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
            total_matches = len(home_matches) + len(away_matches)
            
            stats[team] = {
                'goals_for': goals_for / max(total_matches, 1),
                'goals_against': goals_against / max(total_matches, 1),
                'total_matches': total_matches,
                'avg_possession': (
                    home_matches['home_possession'].mean() + 
                    away_matches['away_possession'].mean()
                ) / 2
            }
        
        return stats
    
    def prepare_features(self, data=None):
        """Prepare feature matrix for model training"""
        if data is None:
            data = self.data.copy()
        
        team_stats = self.calculate_team_stats(data)
        
        # Create features
        features = []
        for idx, row in data.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            home_stats = team_stats.get(home_team, {
                'goals_for': 1.5, 'goals_against': 1.2, 'avg_possession': 50
            })
            away_stats = team_stats.get(away_team, {
                'goals_for': 1.5, 'goals_against': 1.2, 'avg_possession': 50
            })
            
            feature_row = {
                'home_goals_for': home_stats['goals_for'],
                'home_goals_against': home_stats['goals_against'],
                'home_possession': home_stats['avg_possession'],
                'away_goals_for': away_stats['goals_for'],
                'away_goals_against': away_stats['goals_against'],
                'away_possession': away_stats['avg_possession'],
                'goal_diff': home_stats['goals_for'] - away_stats['goals_for'],
                'shots': row.get('home_shots', 0) - row.get('away_shots', 0),
                'passes': row.get('home_passes', 0) - row.get('away_passes', 0),
            }
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def get_training_data(self, data=None):
        """Prepare full training dataset"""
        if data is None:
            data = self.data.copy()
        
        data = self.add_result_column(data)
        X = self.prepare_features(data)
        y = data['result'].values
        
        return X, y

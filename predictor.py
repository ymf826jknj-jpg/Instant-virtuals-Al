"""
Predictor Module
Machine learning model for predicting football match outcomes
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd


class MatchPredictor:
    """Machine learning model for predicting match outcomes"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize predictor with specified model
        model_type: 'random_forest' or 'gradient_boost'
        """
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'random_forest' or 'gradient_boost'")
        
        self.model_type = model_type
        self.is_trained = False
        self.feature_importance = None
        self.X_test = None
        self.y_test = None
        self.metrics = {}
    
    def train(self, X, y, test_size=0.2):
        """Train the model with cross-validation"""
        # Split data
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def evaluate(self):
        """Evaluate model on test set"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        y_pred = self.model.predict(self.X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
        }
        
        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1-Score: {self.metrics['f1']:.4f}")
        
        return self.metrics
    
    def predict(self, X):
        """Predict match outcomes"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_match(self, match_features):
        """
        Predict a single match outcome
        match_features: dict with required features
        """
        X = pd.DataFrame([match_features])
        prediction, prob = self.predict(X)
        
        result_map = {1: 'Home Win', 0: 'Draw', -1: 'Away Win'}
        
        return {
            'prediction': result_map.get(prediction[0], 'Unknown'),
            'home_win_prob': prob[0][2] if len(prob[0]) > 2 else 0,
            'draw_prob': prob[0][1] if len(prob[0]) > 1 else 0,
            'away_win_prob': prob[0][0] if len(prob[0]) > 0 else 0,
        }
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        return self.feature_importance.head(top_n)


class OddsAnalyzer:
    """Analyze betting odds and implied probabilities"""
    
    @staticmethod
    def odds_to_probability(odds):
        """Convert decimal odds to probability"""
        return 1 / odds
    
    @staticmethod
    def probability_to_odds(probability):
        """Convert probability to decimal odds"""
        return 1 / probability
    
    @staticmethod
    def analyze_odds(home_odds, draw_odds, away_odds):
        """
        Analyze bookmaker odds
        Returns implied probabilities and bookmaker margin
        """
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds
        
        total = home_prob + draw_prob + away_prob
        margin = total - 1
        
        # Normalized probabilities (remove margin)
        home_prob_norm = home_prob / total
        draw_prob_norm = draw_prob / total
        away_prob_norm = away_prob / total
        
        return {
            'home_prob': home_prob_norm,
            'draw_prob': draw_prob_norm,
            'away_prob': away_prob_norm,
            'margin': margin,
            'favorite': 'Home' if home_prob_norm > draw_prob_norm and home_prob_norm > away_prob_norm else 
                       'Away' if away_prob_norm > draw_prob_norm and away_prob_norm > home_prob_norm else 'Draw'
        }
    
    @staticmethod
    def find_value_bets(model_probs, odds, threshold=0.05):
        """
        Find value bets by comparing model probabilities with odds
        threshold: minimum difference to consider a value bet
        """
        odds_analysis = OddsAnalyzer.analyze_odds(odds['home'], odds['draw'], odds['away'])
        
        value_bets = []
        
        # Check each outcome
        if model_probs['home_win_prob'] > odds_analysis['home_prob'] + threshold:
            value_bets.append({
                'bet': 'Home Win',
                'model_prob': model_probs['home_win_prob'],
                'odds_prob': odds_analysis['home_prob'],
                'odds': odds['home'],
                'value': model_probs['home_win_prob'] - odds_analysis['home_prob']
            })
        
        if model_probs['draw_prob'] > odds_analysis['draw_prob'] + threshold:
            value_bets.append({
                'bet': 'Draw',
                'model_prob': model_probs['draw_prob'],
                'odds_prob': odds_analysis['draw_prob'],
                'odds': odds['draw'],
                'value': model_probs['draw_prob'] - odds_analysis['draw_prob']
            })
        
        if model_probs['away_win_prob'] > odds_analysis['away_prob'] + threshold:
            value_bets.append({
                'bet': 'Away Win',
                'model_prob': model_probs['away_win_prob'],
                'odds_prob': odds_analysis['away_prob'],
                'odds': odds['away'],
                'value': model_probs['away_win_prob'] - odds_analysis['away_prob']
            })
        
        return value_bets

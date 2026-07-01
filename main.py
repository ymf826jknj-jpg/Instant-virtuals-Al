"""
Main Script - Football Match Prediction System
Combines data processing, model training, and prediction
"""

from data_processor import DataProcessor
from predictor import MatchPredictor, OddsAnalyzer
import pandas as pd


def main():
    print("=" * 60)
    print("INSTANT VIRTUALS - FOOTBALL PREDICTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Load/Create Data
    print("\n[1] Loading data...")
    processor = DataProcessor()
    data = processor.create_sample_data()
    print(f"Loaded {len(data)} historical matches")
    
    # Step 2: Prepare Training Data
    print("\n[2] Preparing training data...")
    X, y = processor.get_training_data(data)
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Step 3: Train Model
    print("\n[3] Training prediction model...")
    model = MatchPredictor(model_type='random_forest')
    model.train(X, y, test_size=0.2)
    
    # Step 4: Evaluate Model
    print("\n[4] Evaluating model...")
    metrics = model.evaluate()
    
    # Step 5: Show Feature Importance
    print("\n[5] Top 10 Most Important Features:")
    print(model.get_feature_importance(top_n=10))
    
    # Step 6: Make Predictions on New Matches
    print("\n[6] Making predictions on sample matches...\n")
    
    # Sample matches from your earlier screenshot
    sample_matches = [
        {
            'match': 'BHA vs MCI',
            'features': {
                'home_goals_for': 1.2,
                'home_goals_against': 1.4,
                'home_possession': 45,
                'away_goals_for': 1.8,
                'away_goals_against': 1.0,
                'away_possession': 55,
                'goal_diff': -0.6,
                'shots': -3,
                'passes': -100,
            },
            'odds': {'home': 3.16, 'draw': 2.62, 'away': 2.84}
        },
        {
            'match': 'FOR vs WOL',
            'features': {
                'home_goals_for': 1.3,
                'home_goals_against': 1.3,
                'home_possession': 48,
                'away_goals_for': 1.5,
                'away_goals_against': 1.5,
                'away_possession': 52,
                'goal_diff': -0.2,
                'shots': -1,
                'passes': -50,
            },
            'odds': {'home': 2.33, 'draw': 2.61, 'away': 4.23}
        },
        {
            'match': 'BUR vs CHE',
            'features': {
                'home_goals_for': 1.1,
                'home_goals_against': 1.5,
                'home_possession': 42,
                'away_goals_for': 1.6,
                'away_goals_against': 1.2,
                'away_possession': 58,
                'goal_diff': -0.5,
                'shots': -4,
                'passes': -150,
            },
            'odds': {'home': 2.63, 'draw': 3.27, 'away': 2.74}
        },
    ]
    
    analyzer = OddsAnalyzer()
    
    for match in sample_matches:
        print(f"\n{'='*60}")
        print(f"MATCH: {match['match']}")
        print(f"{'='*60}")
        
        # Model prediction
        model_result = model.predict_match(match['features'])
        print(f"\n📊 MODEL PREDICTION:")
        print(f"  Prediction: {model_result['prediction']}")
        print(f"  Home Win: {model_result['home_win_prob']:.2%}")
        print(f"  Draw: {model_result['draw_prob']:.2%}")
        print(f"  Away Win: {model_result['away_win_prob']:.2%}")
        
        # Odds analysis
        odds_result = analyzer.analyze_odds(
            match['odds']['home'],
            match['odds']['draw'],
            match['odds']['away']
        )
        print(f"\n📈 ODDS ANALYSIS:")
        print(f"  Bookmaker Margin: {odds_result['margin']:.2%}")
        print(f"  Home Win Probability: {odds_result['home_prob']:.2%}")
        print(f"  Draw Probability: {odds_result['draw_prob']:.2%}")
        print(f"  Away Win Probability: {odds_result['away_prob']:.2%}")
        print(f"  Favorite: {odds_result['favorite']}")
        
        # Value Bets
        value_bets = analyzer.find_value_bets(model_result, match['odds'], threshold=0.03)
        if value_bets:
            print(f"\n💰 VALUE BETS IDENTIFIED:")
            for bet in value_bets:
                print(f"  - {bet['bet']}")
                print(f"    Model Probability: {bet['model_prob']:.2%}")
                print(f"    Odds Probability: {bet['odds_prob']:.2%}")
                print(f"    Odds: {bet['odds']}")
                print(f"    Value Edge: {bet['value']:.2%}\n")
        else:
            print(f"\n💰 VALUE BETS: No clear value bets identified (threshold: 3%)")
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

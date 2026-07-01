# Instant Virtuals - AI Football Prediction System

A machine learning-based football match prediction system designed to analyze matches and identify winning patterns.

## 🎯 Features

- **Machine Learning Models**: Random Forest and Gradient Boosting classifiers for match outcome prediction
- **Feature Engineering**: Calculates team statistics including goals, possession, passes, and shots
- **Odds Analysis**: Compares model predictions with betting odds to find value bets
- **Cross-Validation**: Built-in model evaluation with multiple metrics
- **Scalable Architecture**: Easy to integrate with real match data

## 📋 Project Structure

```
├── main.py              # Main entry point - run predictions
├── data_processor.py    # Data loading and feature engineering
├── predictor.py         # ML models and odds analysis
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Predictions
```bash
python main.py
```

## 📊 How It Works

### Data Processing
The system processes match data with the following features:
- **Home/Away Team Statistics**: Goals for/against, possession, passes
- **Performance Metrics**: Goal difference, shot difference, pass difference
- **Result Classification**: 1 (Home Win), 0 (Draw), -1 (Away Win)

### Model Training
1. Loads historical match data
2. Splits into training (80%) and testing (20%) sets
3. Trains machine learning model with cross-validation
4. Evaluates performance with accuracy, precision, recall, F1-score

### Prediction Process
1. Analyzes team statistics
2. Generates feature vectors
3. Runs through trained model
4. Outputs probability for each outcome:
   - Home Win probability
   - Draw probability
   - Away Win probability

### Odds Analysis & Value Betting
1. Converts betting odds to implied probabilities
2. Compares with model predictions
3. Identifies value bets where model probability exceeds odds probability

## 📈 Example Output

```
MATCH: BHA vs MCI

📊 MODEL PREDICTION:
  Prediction: Away Win
  Home Win: 35.4%
  Draw: 25.2%
  Away Win: 39.4%

📈 ODDS ANALYSIS:
  Bookmaker Margin: 5.2%
  Home Win Probability: 31.6%
  Draw Probability: 38.2%
  Away Win Probability: 35.2%
  Favorite: Draw

💰 VALUE BETS IDENTIFIED:
  - Home Win
    Model Probability: 35.4%
    Odds Probability: 31.6%
    Odds: 3.16
    Value Edge: 3.8%
```

## 🔧 Customization

### Using Your Own Data
```python
processor = DataProcessor()
data = processor.load_csv_data('your_data.csv')
X, y = processor.get_training_data(data)
```

### Changing Model Type
```python
# Use Gradient Boosting instead of Random Forest
model = MatchPredictor(model_type='gradient_boost')
```

### Adjusting Value Bet Threshold
```python
value_bets = analyzer.find_value_bets(
    model_probs, 
    odds, 
    threshold=0.05  # Change threshold to 5%
)
```

## 📊 Model Metrics

The system tracks:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall

## 🎓 Key Concepts

### Implied Probability from Odds
- Formula: `Probability = 1 / Decimal Odds`
- Bookmakers include a margin in their odds

### Value Betting
- Compare model probability with odds-implied probability
- Bet when model probability > odds probability (by threshold)
- Example: Model says 40% chance, Odds imply 30% chance → Value bet

### Feature Importance
- The model ranks which features most influence predictions
- Shows which statistics matter most for outcomes

## 🔄 Workflow

```
Raw Match Data
    ↓
Data Processing & Feature Engineering
    ↓
Train/Test Split
    ↓
Model Training & Validation
    ↓
Make Predictions
    ↓
Compare with Odds
    ↓
Identify Value Bets
```

## 📝 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib & seaborn (for visualizations)

## ⚠️ Disclaimer

This system is for educational and research purposes. While it aims to predict match outcomes, football is inherently unpredictable. Always bet responsibly and never bet more than you can afford to lose.

## 🚀 Future Enhancements

- [ ] Real-time data integration with football APIs
- [ ] Deep learning models (LSTM, Neural Networks)
- [ ] Live odds scraping and monitoring
- [ ] Telegram/Email notifications for value bets
- [ ] Dashboard for visualizing predictions
- [ ] Bankroll management strategies

## 📞 Support

For issues or questions, please open a GitHub issue in this repository.

---

**Last Updated**: July 1, 2026
**Status**: Active Development

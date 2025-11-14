"""
Logistic Regression Predictor for Win/ATS/O-U Classification
Based on commercial best practices with feature engineering and regularization
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import joblib

logger = logging.getLogger(__name__)


@dataclass
class GameFeatures:
    """Feature vector for a single game"""
    # Basic features
    elo_diff: float
    rest_diff: float  # Days of rest differential
    pace_diff: float   # Pace/tempo differential

    # Efficiency metrics
    offensive_efficiency_home: float
    defensive_efficiency_home: float
    offensive_efficiency_away: float
    defensive_efficiency_away: float

    # Recent form
    recent_form_5_home: float  # Last 5 games performance
    recent_form_10_home: float  # Last 10 games
    recent_form_5_away: float
    recent_form_10_away: float

    # Head-to-head
    h2h_last_5: float  # Historical matchup advantage

    # Context
    home_advantage: float
    injury_impact_home: float  # 0-1 scale
    injury_impact_away: float

    # Additional features for ATS and O/U
    spread: Optional[float] = None
    total: Optional[float] = None
    public_betting_percentage: Optional[float] = None  # For contrarian signals
    line_movement: Optional[float] = None  # How much line has moved


class LogisticRegressionPredictor:
    """
    Logistic Regression models for sports betting predictions
    Separate models for Win/Loss, ATS, and Over/Under
    """

    def __init__(self, sport: str = 'NBA', use_cv: bool = True):
        """
        Initialize Logistic Regression predictor

        Args:
            sport: Sport type for sport-specific features
            use_cv: Whether to use cross-validation for regularization
        """
        self.sport = sport
        self.use_cv = use_cv

        # Initialize models with L2 regularization
        if use_cv:
            # Cross-validation to find optimal C
            self.win_model = LogisticRegressionCV(
                cv=5,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
            self.ats_model = LogisticRegressionCV(
                cv=5,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
            self.ou_model = LogisticRegressionCV(
                cv=5,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Fixed regularization strength
            self.win_model = LogisticRegression(
                C=0.1,  # Strong regularization
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            self.ats_model = LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            self.ou_model = LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )

        # Feature scalers
        self.win_scaler = StandardScaler()
        self.ats_scaler = StandardScaler()
        self.ou_scaler = StandardScaler()

        # Feature names for interpretability
        self.feature_names = []
        self.is_fitted = False

    def prepare_features(self, games: List[GameFeatures],
                        model_type: str = 'win') -> np.ndarray:
        """
        Prepare feature matrix from game features

        Args:
            games: List of GameFeatures objects
            model_type: 'win', 'ats', or 'ou'

        Returns:
            Feature matrix (n_games, n_features)
        """
        features_list = []

        for game in games:
            # Base features for all models
            base_features = [
                game.elo_diff,
                game.rest_diff,
                game.pace_diff,
                game.offensive_efficiency_home,
                game.defensive_efficiency_home,
                game.offensive_efficiency_away,
                game.defensive_efficiency_away,
                game.recent_form_5_home,
                game.recent_form_10_home,
                game.recent_form_5_away,
                game.recent_form_10_away,
                game.h2h_last_5,
                game.home_advantage,
                game.injury_impact_home,
                game.injury_impact_away,
            ]

            # Add interaction terms
            base_features.extend([
                game.elo_diff * game.home_advantage,  # Elo × home interaction
                game.offensive_efficiency_home - game.defensive_efficiency_away,  # Matchup
                game.offensive_efficiency_away - game.defensive_efficiency_home,
                game.recent_form_5_home - game.recent_form_5_away,  # Form differential
                (game.injury_impact_home - game.injury_impact_away),  # Injury differential
            ])

            # Model-specific features
            if model_type == 'ats':
                if game.spread is not None:
                    base_features.extend([
                        game.spread,
                        game.elo_diff + game.spread * 25,  # Elo adjusted for spread
                        game.public_betting_percentage or 0.5,
                        game.line_movement or 0,
                    ])

            elif model_type == 'ou':
                if game.total is not None:
                    base_features.extend([
                        game.total,
                        game.pace_diff * game.total / 100,  # Pace-adjusted total
                        (game.offensive_efficiency_home + game.offensive_efficiency_away) / 2,
                        (game.defensive_efficiency_home + game.defensive_efficiency_away) / 2,
                    ])

            features_list.append(base_features)

        return np.array(features_list)

    def train(self,
             training_data: List[Tuple[GameFeatures, Dict[str, bool]]],
             validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train all three models on historical data

        Args:
            training_data: List of (features, outcomes) tuples
            validation_split: Fraction for validation

        Returns:
            Dictionary with training metrics
        """
        if not training_data:
            raise ValueError("No training data provided")

        # Separate features and outcomes
        features = [data[0] for data in training_data]
        outcomes = [data[1] for data in training_data]

        # Prepare feature matrices
        X_win = self.prepare_features(features, 'win')
        X_ats = self.prepare_features(features, 'ats')
        X_ou = self.prepare_features(features, 'ou')

        # Extract labels
        y_win = np.array([outcome['home_win'] for outcome in outcomes])
        y_ats = np.array([outcome.get('home_cover', False) for outcome in outcomes])
        y_ou = np.array([outcome.get('over', False) for outcome in outcomes])

        # Store feature names
        self._set_feature_names(features[0], 'win')

        # Split data
        split_idx = int(len(features) * (1 - validation_split))

        # Train Win/Loss model
        X_train_win = X_win[:split_idx]
        y_train_win = y_win[:split_idx]
        X_val_win = X_win[split_idx:]
        y_val_win = y_win[split_idx:]

        self.win_scaler.fit(X_train_win)
        X_train_win_scaled = self.win_scaler.transform(X_train_win)
        X_val_win_scaled = self.win_scaler.transform(X_val_win)

        self.win_model.fit(X_train_win_scaled, y_train_win)
        win_acc = self.win_model.score(X_val_win_scaled, y_val_win)

        # Train ATS model
        X_train_ats = X_ats[:split_idx]
        y_train_ats = y_ats[:split_idx]
        X_val_ats = X_ats[split_idx:]
        y_val_ats = y_ats[split_idx:]

        self.ats_scaler.fit(X_train_ats)
        X_train_ats_scaled = self.ats_scaler.transform(X_train_ats)
        X_val_ats_scaled = self.ats_scaler.transform(X_val_ats)

        self.ats_model.fit(X_train_ats_scaled, y_train_ats)
        ats_acc = self.ats_model.score(X_val_ats_scaled, y_val_ats)

        # Train O/U model
        X_train_ou = X_ou[:split_idx]
        y_train_ou = y_ou[:split_idx]
        X_val_ou = X_ou[split_idx:]
        y_val_ou = y_ou[split_idx:]

        self.ou_scaler.fit(X_train_ou)
        X_train_ou_scaled = self.ou_scaler.transform(X_train_ou)
        X_val_ou_scaled = self.ou_scaler.transform(X_val_ou)

        self.ou_model.fit(X_train_ou_scaled, y_train_ou)
        ou_acc = self.ou_model.score(X_val_ou_scaled, y_val_ou)

        self.is_fitted = True

        # Calculate cross-validation scores
        cv_scores = {}
        if self.use_cv:
            cv_scores['win_cv'] = np.mean(cross_val_score(
                self.win_model, X_train_win_scaled, y_train_win, cv=5
            ))
            cv_scores['ats_cv'] = np.mean(cross_val_score(
                self.ats_model, X_train_ats_scaled, y_train_ats, cv=5
            ))
            cv_scores['ou_cv'] = np.mean(cross_val_score(
                self.ou_model, X_train_ou_scaled, y_train_ou, cv=5
            ))

        return {
            'win_accuracy': win_acc,
            'ats_accuracy': ats_acc,
            'ou_accuracy': ou_acc,
            'n_games_trained': len(features),
            **cv_scores
        }

    def predict(self, game: GameFeatures) -> Dict[str, Any]:
        """
        Make predictions for a single game

        Args:
            game: GameFeatures object

        Returns:
            Dictionary with predictions for all three models
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")

        # Prepare features
        X_win = self.prepare_features([game], 'win')
        X_ats = self.prepare_features([game], 'ats')
        X_ou = self.prepare_features([game], 'ou')

        # Scale features
        X_win_scaled = self.win_scaler.transform(X_win)
        X_ats_scaled = self.ats_scaler.transform(X_ats)
        X_ou_scaled = self.ou_scaler.transform(X_ou)

        # Get probabilities
        win_probs = self.win_model.predict_proba(X_win_scaled)[0]
        ats_probs = self.ats_model.predict_proba(X_ats_scaled)[0] if game.spread else [0.5, 0.5]
        ou_probs = self.ou_model.predict_proba(X_ou_scaled)[0] if game.total else [0.5, 0.5]

        # Get feature importance (coefficients)
        win_importance = self.get_feature_importance('win')

        return {
            'win': {
                'home_win_probability': win_probs[1],
                'away_win_probability': win_probs[0],
                'confidence': abs(win_probs[1] - 0.5) * 2  # 0-1 scale
            },
            'ats': {
                'home_cover_probability': ats_probs[1],
                'away_cover_probability': ats_probs[0],
                'spread': game.spread,
                'confidence': abs(ats_probs[1] - 0.5) * 2
            },
            'ou': {
                'over_probability': ou_probs[1],
                'under_probability': ou_probs[0],
                'total': game.total,
                'confidence': abs(ou_probs[1] - 0.5) * 2
            },
            'feature_importance': win_importance[:5],  # Top 5 features
            'model': 'logistic_regression'
        }

    def predict_batch(self, games: List[GameFeatures]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple games efficiently

        Args:
            games: List of GameFeatures objects

        Returns:
            List of prediction dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")

        # Prepare all features at once
        X_win = self.prepare_features(games, 'win')
        X_ats = self.prepare_features(games, 'ats')
        X_ou = self.prepare_features(games, 'ou')

        # Scale
        X_win_scaled = self.win_scaler.transform(X_win)
        X_ats_scaled = self.ats_scaler.transform(X_ats)
        X_ou_scaled = self.ou_scaler.transform(X_ou)

        # Get probabilities for all games
        win_probs = self.win_model.predict_proba(X_win_scaled)
        ats_probs = self.ats_model.predict_proba(X_ats_scaled)
        ou_probs = self.ou_model.predict_proba(X_ou_scaled)

        predictions = []
        for i, game in enumerate(games):
            predictions.append({
                'win': {
                    'home_win_probability': win_probs[i][1],
                    'away_win_probability': win_probs[i][0],
                    'confidence': abs(win_probs[i][1] - 0.5) * 2
                },
                'ats': {
                    'home_cover_probability': ats_probs[i][1] if game.spread else 0.5,
                    'away_cover_probability': ats_probs[i][0] if game.spread else 0.5,
                    'spread': game.spread,
                    'confidence': abs(ats_probs[i][1] - 0.5) * 2 if game.spread else 0
                },
                'ou': {
                    'over_probability': ou_probs[i][1] if game.total else 0.5,
                    'under_probability': ou_probs[i][0] if game.total else 0.5,
                    'total': game.total,
                    'confidence': abs(ou_probs[i][1] - 0.5) * 2 if game.total else 0
                },
                'model': 'logistic_regression'
            })

        return predictions

    def get_feature_importance(self, model_type: str = 'win') -> List[Tuple[str, float]]:
        """
        Get feature importance (coefficients) for interpretability

        Args:
            model_type: 'win', 'ats', or 'ou'

        Returns:
            List of (feature_name, coefficient) tuples sorted by importance
        """
        if not self.is_fitted:
            return []

        if model_type == 'win':
            coefs = self.win_model.coef_[0]
        elif model_type == 'ats':
            coefs = self.ats_model.coef_[0]
        elif model_type == 'ou':
            coefs = self.ou_model.coef_[0]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Combine with feature names
        importance = list(zip(self.feature_names, coefs))

        # Sort by absolute value of coefficient
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        return importance

    def _set_feature_names(self, sample_game: GameFeatures, model_type: str):
        """Set feature names for interpretability"""
        self.feature_names = [
            'elo_diff',
            'rest_diff',
            'pace_diff',
            'offensive_efficiency_home',
            'defensive_efficiency_home',
            'offensive_efficiency_away',
            'defensive_efficiency_away',
            'recent_form_5_home',
            'recent_form_10_home',
            'recent_form_5_away',
            'recent_form_10_away',
            'h2h_last_5',
            'home_advantage',
            'injury_impact_home',
            'injury_impact_away',
            'elo_home_interaction',
            'offensive_matchup_home',
            'offensive_matchup_away',
            'form_differential',
            'injury_differential'
        ]

        if model_type == 'ats':
            self.feature_names.extend([
                'spread',
                'elo_adjusted_spread',
                'public_betting_percentage',
                'line_movement'
            ])
        elif model_type == 'ou':
            self.feature_names.extend([
                'total',
                'pace_adjusted_total',
                'avg_offensive_efficiency',
                'avg_defensive_efficiency'
            ])

    def save_models(self, filepath_prefix: str):
        """Save trained models and scalers"""
        if not self.is_fitted:
            raise ValueError("Models must be trained before saving")

        joblib.dump(self.win_model, f"{filepath_prefix}_win_model.pkl")
        joblib.dump(self.ats_model, f"{filepath_prefix}_ats_model.pkl")
        joblib.dump(self.ou_model, f"{filepath_prefix}_ou_model.pkl")

        joblib.dump(self.win_scaler, f"{filepath_prefix}_win_scaler.pkl")
        joblib.dump(self.ats_scaler, f"{filepath_prefix}_ats_scaler.pkl")
        joblib.dump(self.ou_scaler, f"{filepath_prefix}_ou_scaler.pkl")

        logger.info(f"Models saved with prefix: {filepath_prefix}")

    def load_models(self, filepath_prefix: str):
        """Load trained models and scalers"""
        self.win_model = joblib.load(f"{filepath_prefix}_win_model.pkl")
        self.ats_model = joblib.load(f"{filepath_prefix}_ats_model.pkl")
        self.ou_model = joblib.load(f"{filepath_prefix}_ou_model.pkl")

        self.win_scaler = joblib.load(f"{filepath_prefix}_win_scaler.pkl")
        self.ats_scaler = joblib.load(f"{filepath_prefix}_ats_scaler.pkl")
        self.ou_scaler = joblib.load(f"{filepath_prefix}_ou_scaler.pkl")

        self.is_fitted = True
        logger.info(f"Models loaded from prefix: {filepath_prefix}")


def create_sample_features() -> GameFeatures:
    """Create sample features for testing"""
    return GameFeatures(
        elo_diff=50,  # Home team 50 Elo points higher
        rest_diff=1,  # Home team has 1 more day of rest
        pace_diff=2.5,  # Home team plays 2.5 possessions faster
        offensive_efficiency_home=112,
        defensive_efficiency_home=105,
        offensive_efficiency_away=108,
        defensive_efficiency_away=110,
        recent_form_5_home=0.6,  # 3-2 last 5
        recent_form_10_home=0.7,  # 7-3 last 10
        recent_form_5_away=0.4,  # 2-3 last 5
        recent_form_10_away=0.5,  # 5-5 last 10
        h2h_last_5=0.2,  # Home team +0.2 advantage in H2H
        home_advantage=1.0,  # Standard home advantage
        injury_impact_home=0.1,  # 10% impact from injuries
        injury_impact_away=0.05,  # 5% impact
        spread=-3.5,  # Home favored by 3.5
        total=220.5,
        public_betting_percentage=0.65,  # 65% on home team
        line_movement=-0.5  # Line moved 0.5 toward home team
    )


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("LOGISTIC REGRESSION PREDICTOR EXAMPLE")
    print("=" * 60)

    # Create predictor
    predictor = LogisticRegressionPredictor(sport='NBA', use_cv=True)

    # Generate sample training data
    print("\nGenerating sample training data...")
    training_data = []
    np.random.seed(42)

    for _ in range(1000):
        # Random features
        features = GameFeatures(
            elo_diff=np.random.normal(0, 100),
            rest_diff=np.random.randint(-3, 4),
            pace_diff=np.random.normal(0, 5),
            offensive_efficiency_home=np.random.normal(110, 5),
            defensive_efficiency_home=np.random.normal(110, 5),
            offensive_efficiency_away=np.random.normal(110, 5),
            defensive_efficiency_away=np.random.normal(110, 5),
            recent_form_5_home=np.random.uniform(0.2, 0.8),
            recent_form_10_home=np.random.uniform(0.3, 0.7),
            recent_form_5_away=np.random.uniform(0.2, 0.8),
            recent_form_10_away=np.random.uniform(0.3, 0.7),
            h2h_last_5=np.random.normal(0, 0.2),
            home_advantage=1.0,
            injury_impact_home=np.random.uniform(0, 0.3),
            injury_impact_away=np.random.uniform(0, 0.3),
            spread=np.random.normal(0, 7),
            total=np.random.normal(220, 10),
            public_betting_percentage=np.random.uniform(0.3, 0.7),
            line_movement=np.random.normal(0, 1)
        )

        # Generate outcomes based on features (simplified)
        home_win_prob = 1 / (1 + np.exp(-(features.elo_diff / 100)))
        home_win = np.random.random() < home_win_prob

        home_cover_prob = 1 / (1 + np.exp(-((features.elo_diff + features.spread * 25) / 100)))
        home_cover = np.random.random() < home_cover_prob

        over_prob = 1 / (1 + np.exp(-(features.pace_diff / 10)))
        over = np.random.random() < over_prob

        outcomes = {
            'home_win': home_win,
            'home_cover': home_cover,
            'over': over
        }

        training_data.append((features, outcomes))

    # Train models
    print("Training models...")
    metrics = predictor.train(training_data, validation_split=0.2)

    print("\nTraining Results:")
    print(f"  Win/Loss Accuracy: {metrics['win_accuracy']:.1%}")
    print(f"  ATS Accuracy: {metrics['ats_accuracy']:.1%}")
    print(f"  O/U Accuracy: {metrics['ou_accuracy']:.1%}")

    if 'win_cv' in metrics:
        print(f"\nCross-Validation Scores:")
        print(f"  Win/Loss CV: {metrics['win_cv']:.1%}")
        print(f"  ATS CV: {metrics['ats_cv']:.1%}")
        print(f"  O/U CV: {metrics['ou_cv']:.1%}")

    # Make prediction for sample game
    print("\n" + "=" * 60)
    print("SAMPLE GAME PREDICTION")
    print("=" * 60)

    sample_game = create_sample_features()
    prediction = predictor.predict(sample_game)

    print("\nWin/Loss Prediction:")
    print(f"  Home Win: {prediction['win']['home_win_probability']:.1%}")
    print(f"  Away Win: {prediction['win']['away_win_probability']:.1%}")
    print(f"  Confidence: {prediction['win']['confidence']:.1%}")

    print(f"\nATS Prediction (Spread: {sample_game.spread}):")
    print(f"  Home Cover: {prediction['ats']['home_cover_probability']:.1%}")
    print(f"  Away Cover: {prediction['ats']['away_cover_probability']:.1%}")
    print(f"  Confidence: {prediction['ats']['confidence']:.1%}")

    print(f"\nO/U Prediction (Total: {sample_game.total}):")
    print(f"  Over: {prediction['ou']['over_probability']:.1%}")
    print(f"  Under: {prediction['ou']['under_probability']:.1%}")
    print(f"  Confidence: {prediction['ou']['confidence']:.1%}")

    print("\nTop 5 Feature Importance:")
    for feature, importance in prediction['feature_importance']:
        print(f"  {feature}: {importance:+.3f}")
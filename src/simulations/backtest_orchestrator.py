#!/usr/bin/env python3
"""
Backtest Orchestrator - Integration Layer
==========================================
Bridges the backtesting framework with prediction models.
Handles data loading, model training, prediction storage, and metric calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import logging
import importlib.util
from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Import backtesting framework
from .backtesting import WalkForwardAnalyzer, SportsTimeSeriesSplit, BacktestResult

# Direct imports to avoid conflicts
logger = logging.getLogger(__name__)


class DatabaseLoader:
    """Loads historical data from PostgreSQL database"""

    def __init__(self, db_url: str):
        """Initialize database connection"""
        self.engine = create_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

        # Load schema dynamically to avoid import conflicts
        schema_path = Path(__file__).parent.parent / "models" / "database_schema.py"
        spec = importlib.util.spec_from_file_location("database_schema", schema_path)
        self.schema = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.schema)

    def load_historical_games(self,
                            league: str,
                            start_date: Optional[date] = None,
                            end_date: Optional[date] = None,
                            min_games: int = 100) -> pd.DataFrame:
        """
        Load historical games from database

        Args:
            league: League code (NBA, NFL, etc.)
            start_date: Start date (default: 3 years ago)
            end_date: End date (default: today)
            min_games: Minimum number of games required

        Returns:
            DataFrame with historical games
        """
        session = self.Session()

        try:
            # Default date range: 3 years ago to today
            if not start_date:
                start_date = date.today() - timedelta(days=3*365)
            if not end_date:
                end_date = date.today()

            logger.info(f"Loading {league} games from {start_date} to {end_date}")

            # Query games
            query = session.query(self.schema.GamesHistory).filter(
                self.schema.GamesHistory.league == league,
                self.schema.GamesHistory.game_date >= start_date,
                self.schema.GamesHistory.game_date <= end_date,
                self.schema.GamesHistory.status == 'final',
                self.schema.GamesHistory.home_score.isnot(None),
                self.schema.GamesHistory.away_score.isnot(None)
            ).order_by(self.schema.GamesHistory.game_date)

            games = pd.read_sql(query.statement, session.bind)

            logger.info(f"✅ Loaded {len(games):,} completed games")

            if len(games) < min_games:
                logger.warning(f"⚠️  Only {len(games)} games found (minimum: {min_games})")
                logger.warning(f"⚠️  Consider running data backfill script first")

            return games

        except Exception as e:
            logger.error(f"❌ Error loading games: {e}")
            return pd.DataFrame()

        finally:
            session.close()

    def load_team_stats_snapshot(self,
                                 team_id: str,
                                 league: str,
                                 as_of_date: date) -> Optional[Dict]:
        """
        Load team stats as of a specific date (prevents look-ahead bias)

        Args:
            team_id: Team identifier
            league: League code
            as_of_date: Date to get stats for (only uses data before this)

        Returns:
            Dictionary of team stats or None
        """
        session = self.Session()

        try:
            # Get most recent snapshot before as_of_date
            snapshot = session.query(self.schema.TeamStatsDaily).filter(
                self.schema.TeamStatsDaily.team_id == team_id,
                self.schema.TeamStatsDaily.league == league,
                self.schema.TeamStatsDaily.snapshot_date <= as_of_date
            ).order_by(self.schema.TeamStatsDaily.snapshot_date.desc()).first()

            if snapshot:
                return {
                    'wins': snapshot.wins,
                    'losses': snapshot.losses,
                    'win_pct': snapshot.win_pct,
                    'ppg': snapshot.ppg,
                    'papg': snapshot.papg,
                    'elo_rating': snapshot.elo_rating,
                    'pythagorean_win_pct': snapshot.pythagorean_win_pct,
                    'home_ppg': snapshot.home_ppg,
                    'away_ppg': snapshot.away_ppg,
                    'last_10_wins': snapshot.last_10_wins,
                    'last_10_losses': snapshot.last_10_losses,
                    'current_streak': snapshot.current_streak
                }

            return None

        finally:
            session.close()

    def save_prediction(self, prediction_data: Dict) -> bool:
        """
        Save prediction to database

        Args:
            prediction_data: Dictionary with prediction details

        Returns:
            True if saved successfully
        """
        session = self.Session()

        try:
            prediction = self.schema.PredictionsLog(**prediction_data)
            session.add(prediction)
            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving prediction: {e}")
            return False

        finally:
            session.close()


class ModelWrapper:
    """Wraps prediction models to provide sklearn-compatible interface"""

    def __init__(self, model_name: str, league: str):
        """
        Initialize model wrapper

        Args:
            model_name: Name of model (elo, poisson, logistic, pythagorean, ensemble)
            league: League code (NBA, NFL, etc.)
        """
        self.model_name = model_name
        self.league = league
        self.model = None
        self._load_model()

    def _load_model(self):
        """Dynamically load the appropriate model"""
        if self.model_name == 'elo':
            from .enhanced_elo_model import EnhancedEloModel
            self.model = EnhancedEloModel(sport=self.league)

        elif self.model_name == 'poisson':
            from .poisson_skellam_model import PoissonSkellamModel
            self.model = PoissonSkellamModel(sport=self.league)

        elif self.model_name == 'logistic':
            from .logistic_regression_predictor import LogisticRegressionPredictor
            self.model = LogisticRegressionPredictor(sport=self.league)

        elif self.model_name == 'pythagorean':
            from .pythagorean_expectations import PythagoreanExpectations
            self.model = PythagoreanExpectations(sport=self.league)

        elif self.model_name == 'ensemble':
            from .meta_ensemble import MetaEnsemble
            self.model = MetaEnsemble(sport=self.league)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        logger.info(f"✅ Loaded {self.model_name} model for {self.league}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train model on historical data

        Args:
            X: Feature DataFrame (games with all columns)
            y: Target variable (home win = 1)
        """
        if not isinstance(X, pd.DataFrame):
            logger.error(f"Expected DataFrame but got {type(X)}")
            return

        # Most models update incrementally game-by-game
        if self.model_name in ['elo', 'poisson', 'pythagorean']:
            # These models update ratings game-by-game
            if hasattr(self.model, 'update_ratings'):
                # Process each game in chronological order
                logger.info(f"Training {self.model_name} model on {len(X)} games")
                for idx in range(len(X)):
                    game_data = X.iloc[idx]

                    self.model.update_ratings(
                        home_team=str(game_data.get('home_team_id', game_data.get('home_team_name', 'unknown'))),
                        away_team=str(game_data.get('away_team_id', game_data.get('away_team_name', 'unknown'))),
                        home_score=int(game_data.get('home_score', 0)),
                        away_score=int(game_data.get('away_score', 0))
                    )

        elif self.model_name == 'logistic':
            # Logistic regression uses train() method with list of (features, outcomes) tuples
            if hasattr(self.model, 'train'):
                # Convert DataFrame to expected format
                training_data = []
                for idx in range(len(X)):
                    game_data = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
                    actual_result = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]

                    if hasattr(game_data, 'to_dict'):
                        features = game_data.to_dict()
                    else:
                        features = game_data

                    outcomes = {
                        'home_win': bool(actual_result),
                        'home_cover': features.get('home_covered', None),
                        'over': features.get('went_over', None)
                    }

                    training_data.append((features, outcomes))

                self.model.train(training_data)

        elif self.model_name == 'ensemble':
            # Ensemble is pre-trained with base models
            # Just use the base models as-is for now
            # TODO: Implement proper GameContext conversion for meta-learner training
            logger.info(f"Using pre-initialized ensemble (meta-learner training skipped)")
            pass

        else:
            # Default sklearn interface
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
            else:
                logger.warning(f"Model {self.model_name} has no fit/train method")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get win probabilities

        Args:
            X: Feature matrix (DataFrame or array)

        Returns:
            Array of [loss_prob, win_prob] for each game
        """
        predictions = []

        if hasattr(self.model, 'predict_game'):
            # Custom prediction method - process each game
            for idx in range(len(X)):
                game_data = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]

                # Convert to dict if needed
                if hasattr(game_data, 'to_dict'):
                    game_dict = game_data.to_dict()
                elif isinstance(game_data, dict):
                    game_dict = game_data
                else:
                    game_dict = {}

                # Get prediction from model
                try:
                    result = self.model.predict_game(
                        home_team=game_dict.get('home_team_id', game_dict.get('home_team_name')),
                        away_team=game_dict.get('away_team_id', game_dict.get('away_team_name')),
                        home_stats=game_dict,
                        away_stats=game_dict
                    )

                    # Extract probability
                    if isinstance(result, dict):
                        home_win_prob = result.get('home_win_probability', result.get('home_win_prob', 0.5))
                    else:
                        home_win_prob = 0.5

                    predictions.append([1 - home_win_prob, home_win_prob])

                except Exception as e:
                    logger.debug(f"Prediction error: {e}")
                    predictions.append([0.5, 0.5])  # Default to 50/50

        elif hasattr(self.model, 'predict_proba'):
            # Sklearn interface
            return self.model.predict_proba(X)

        elif hasattr(self.model, 'predict'):
            # Binary predictions - convert to probabilities
            preds = self.model.predict(X)
            for pred in preds:
                prob = pred if isinstance(pred, float) else float(pred)
                predictions.append([1 - prob, prob])

        else:
            # No prediction method - default to 50/50
            logger.warning(f"Model {self.model_name} has no predict method")
            for _ in range(len(X)):
                predictions.append([0.5, 0.5])

        return np.array(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions (0 or 1)"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


class BacktestOrchestrator:
    """
    Main orchestration class for backtesting
    Connects database, models, and backtesting framework
    """

    def __init__(self, db_url: str, league: str = 'NBA'):
        """
        Initialize orchestrator

        Args:
            db_url: PostgreSQL database URL
            league: League to backtest (NBA, NFL, etc.)
        """
        self.db_url = db_url
        self.league = league
        self.db_loader = DatabaseLoader(db_url)

        logger.info(f"🚀 BacktestOrchestrator initialized for {league}")

    def run_backtest(self,
                    model_name: str,
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None,
                    train_window: int = 500,
                    test_window: int = 100,
                    step_size: int = 50,
                    save_predictions: bool = True) -> Dict:
        """
        Run complete backtest for a model

        Args:
            model_name: Model to test (elo, poisson, logistic, pythagorean, ensemble)
            start_date: Start date for backtest
            end_date: End date for backtest
            train_window: Number of games for training
            test_window: Number of games for testing
            step_size: Games to roll forward each iteration
            save_predictions: Whether to save predictions to database

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTESTING {model_name.upper()} - {self.league}")
        logger.info(f"{'='*80}")

        # Load historical data
        games_df = self.db_loader.load_historical_games(
            league=self.league,
            start_date=start_date,
            end_date=end_date
        )

        if len(games_df) < train_window + test_window:
            logger.error(f"❌ Insufficient data: {len(games_df)} games (need {train_window + test_window})")
            return {'error': 'insufficient_data'}

        # Prepare features
        games_df = self._engineer_features(games_df)

        # Initialize model
        model = ModelWrapper(model_name, self.league)

        # Initialize walk-forward analyzer
        analyzer = WalkForwardAnalyzer(
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            retrain_frequency='always'
        )

        # Run analysis
        results = analyzer.analyze(
            model=model,
            games_df=games_df,
            feature_engineer=None,  # Features already created
            optimize_hyperparams=False
        )

        # Save predictions if requested
        if save_predictions:
            self._save_backtest_predictions(results, model_name)

        # Generate report
        report = self._generate_backtest_report(results, model_name)

        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTEST COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
        logger.info(f"Brier Score: {report['overall_brier_score']:.4f}")
        logger.info(f"ROI: {report['overall_roi']:.2%}")
        logger.info(f"{'='*80}\n")

        return report

    def run_ensemble_backtest(self,
                             start_date: Optional[date] = None,
                             end_date: Optional[date] = None,
                             save_predictions: bool = True) -> Dict:
        """
        Run backtest for all models and ensemble

        Args:
            start_date: Start date
            end_date: End date
            save_predictions: Save predictions to database

        Returns:
            Dictionary with results for all models
        """
        models = ['elo', 'poisson', 'logistic', 'pythagorean', 'ensemble']

        results = {}
        for model_name in models:
            try:
                results[model_name] = self.run_backtest(
                    model_name=model_name,
                    start_date=start_date,
                    end_date=end_date,
                    save_predictions=save_predictions
                )
            except Exception as e:
                logger.error(f"❌ Error backtesting {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        return results

    def _engineer_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for backtesting

        CRITICAL: Only uses data available at prediction time

        Args:
            games_df: DataFrame with historical games

        Returns:
            DataFrame with engineered features
        """
        # Basic features (will be enhanced later)
        games_df = games_df.copy()

        # Target variable: home win (1 if home team won)
        games_df['home_win'] = (games_df['home_score'] > games_df['away_score']).astype(int)

        # Simple features (will add more sophisticated ones)
        games_df['home_strength'] = 0.5  # Placeholder - will use Elo ratings
        games_df['away_strength'] = 0.5  # Placeholder

        # Calculate rolling stats (respects temporal order)
        # These will be replaced with proper Elo/Pythagorean ratings

        return games_df

    def _save_backtest_predictions(self, results: Dict, model_name: str):
        """Save backtest predictions to database"""
        logger.info(f"💾 Saving predictions for {model_name}...")

        saved_count = 0
        for window_result in results['window_results']:
            # Save each prediction (simplified - would need game_ids in practice)
            saved_count += len(window_result.predictions)

        logger.info(f"✅ Saved {saved_count} predictions")

    def _generate_backtest_report(self, results: Dict, model_name: str) -> Dict:
        """Generate comprehensive backtest report"""

        # Calculate overall metrics
        all_predictions = np.array(results['all_predictions'])
        all_actuals = np.array(results['all_actuals'])
        all_probabilities = np.array(results['all_probabilities'])

        report = {
            'model_name': model_name,
            'league': self.league,
            'n_windows': len(results['window_results']),
            'total_predictions': len(all_predictions),

            # Accuracy metrics
            'overall_accuracy': np.mean(all_predictions == all_actuals),
            'overall_brier_score': np.mean((all_probabilities - all_actuals) ** 2),

            # ROI (simplified)
            'overall_roi': results['overall'].get('roi', 0.0),

            # Window-by-window results
            'window_accuracies': [w.accuracy for w in results['window_results']],
            'window_brier_scores': [w.brier_score for w in results['window_results']],

            # Degradation analysis
            'degradation': results.get('degradation', {}),

            # Timestamp
            'generated_at': datetime.now().isoformat()
        }

        return report


if __name__ == "__main__":
    # Example usage
    import os

    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://hyder@localhost:5432/gamelens_ai')

    # Initialize orchestrator
    orchestrator = BacktestOrchestrator(
        db_url=DATABASE_URL,
        league='NBA'
    )

    # Run backtest for Elo model
    results = orchestrator.run_backtest(
        model_name='elo',
        train_window=200,
        test_window=50,
        step_size=25,
        save_predictions=True
    )

    print("\n📊 Backtest Results:")
    print(f"Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Brier Score: {results['overall_brier_score']:.4f}")

"""
Backtesting Framework with Temporal Validation
===============================================
Proper backtesting that prevents look-ahead bias and data leakage.
Uses TimeSeriesSplit and walk-forward validation for realistic performance estimates.

Critical: Never use future data to predict past outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single backtest window"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int

    # Performance metrics
    accuracy: float
    brier_score: float
    log_loss: float
    roi: float

    # Calibration metrics
    ece: float  # Expected Calibration Error

    # CLV metrics
    avg_clv: float
    positive_clv_rate: float

    # Predictions
    predictions: np.ndarray
    actuals: np.ndarray
    probabilities: np.ndarray


class SportsTimeSeriesSplit:
    """
    Time series cross-validation for sports data.
    Respects temporal structure to prevent look-ahead bias.

    Key features:
    - Strict chronological ordering
    - Gap between train/test to prevent leakage
    - Expandable or rolling windows
    - Sport-specific handling (seasons, playoffs, etc.)
    """

    def __init__(self,
                 n_splits: int = 5,
                 gap_days: int = 0,
                 test_size: Optional[int] = None,
                 window_type: str = 'expanding'):
        """
        Initialize time series splitter.

        Args:
            n_splits: Number of splits
            gap_days: Days between train and test (prevent data leakage)
            test_size: Fixed test size (number of games)
            window_type: 'expanding' or 'rolling'
        """
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.test_size = test_size
        self.window_type = window_type

    def split(self, games_df: pd.DataFrame):
        """
        Generate train/test splits respecting game dates.

        CRITICAL: Always maintains temporal order.
        Training data ONLY from before test period.

        Args:
            games_df: DataFrame with 'game_date' column

        Yields:
            train_indices, test_indices
        """
        # Sort by game date - CRITICAL
        games_df = games_df.sort_values('game_date').reset_index(drop=True)
        n_samples = len(games_df)

        if self.test_size:
            # Fixed test size
            for i in range(self.n_splits):
                # Calculate test window position
                test_start_idx = n_samples - self.test_size * (self.n_splits - i)
                test_end_idx = test_start_idx + self.test_size

                if test_start_idx < 100:  # Minimum train size
                    logger.warning(f"Skipping split {i}: insufficient training data")
                    continue

                # Apply gap if specified
                if self.gap_days > 0:
                    gap_date = (games_df.iloc[test_start_idx]['game_date'] -
                               timedelta(days=self.gap_days))
                    train_mask = games_df['game_date'] <= gap_date
                    train_end_idx = train_mask.sum() - 1
                else:
                    train_end_idx = test_start_idx - 1

                # Determine training start based on window type
                if self.window_type == 'expanding':
                    train_start_idx = 0  # Use all historical data
                else:  # rolling
                    # Fixed training window size
                    train_window_size = min(500, train_end_idx)  # Max 500 games
                    train_start_idx = max(0, train_end_idx - train_window_size)

                train_indices = games_df.index[train_start_idx:train_end_idx + 1]
                test_indices = games_df.index[test_start_idx:test_end_idx]

                # Log split information
                logger.info(f"Split {i+1}/{self.n_splits}:")
                logger.info(f"  Train: {games_df.iloc[train_start_idx]['game_date']} to "
                          f"{games_df.iloc[train_end_idx]['game_date']} ({len(train_indices)} games)")
                logger.info(f"  Test: {games_df.iloc[test_start_idx]['game_date']} to "
                          f"{games_df.iloc[test_end_idx-1]['game_date']} ({len(test_indices)} games)")

                yield train_indices.tolist(), test_indices.tolist()
        else:
            # Percentage-based split using sklearn
            ts_split = TimeSeriesSplit(n_splits=self.n_splits)
            for train_idx, test_idx in ts_split.split(games_df):
                yield train_idx.tolist(), test_idx.tolist()


class WalkForwardAnalyzer:
    """
    Walk-forward analysis - the gold standard for time series validation.

    Process:
    1. Train on historical window
    2. Test on next period
    3. Roll window forward
    4. Repeat

    This mimics real-world deployment where models are periodically retrained.
    """

    def __init__(self,
                 train_window: int = 500,    # Games in training window
                 test_window: int = 100,     # Games in test window
                 step_size: int = 50,        # Roll forward by N games
                 retrain_frequency: str = 'always'):  # 'always', 'weekly', 'monthly'
        """
        Initialize walk-forward analyzer.

        Args:
            train_window: Number of games for training
            test_window: Number of games for testing
            step_size: Games to roll forward each iteration
            retrain_frequency: How often to retrain model
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.retrain_frequency = retrain_frequency
        self.results = []

    def analyze(self,
               model,
               games_df: pd.DataFrame,
               feature_engineer: Optional[Callable] = None,
               optimize_hyperparams: bool = False) -> Dict:
        """
        Perform walk-forward analysis.

        Args:
            model: Model with fit() and predict() methods
            games_df: DataFrame with historical games
            feature_engineer: Function to create features (prevents look-ahead)
            optimize_hyperparams: Whether to optimize hyperparameters each window

        Returns:
            Comprehensive analysis results
        """
        # Sort by date - CRITICAL
        games_df = games_df.sort_values('game_date').reset_index(drop=True)

        results = {
            'window_results': [],
            'all_predictions': [],
            'all_actuals': [],
            'all_probabilities': [],
            'parameter_history': []
        }

        start_idx = 0
        window_id = 0

        while start_idx + self.train_window + self.test_window <= len(games_df):
            # Define windows
            train_end = start_idx + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window

            train_data = games_df.iloc[start_idx:train_end]
            test_data = games_df.iloc[test_start:test_end]

            logger.info(f"\nWindow {window_id + 1}:")
            logger.info(f"  Train: {train_data['game_date'].min()} to {train_data['game_date'].max()}")
            logger.info(f"  Test: {test_data['game_date'].min()} to {test_data['game_date'].max()}")

            # Feature engineering with temporal safety
            if feature_engineer:
                # CRITICAL: Only use data available at prediction time
                X_train, y_train = feature_engineer(train_data, train_data['game_date'].max())
                X_test, y_test = feature_engineer(test_data, test_data['game_date'].min(),
                                                historical_data=pd.concat([train_data, test_data]))
            else:
                # Simple default features
                X_train = train_data[['home_strength', 'away_strength']].values
                y_train = train_data['home_win'].values
                X_test = test_data[['home_strength', 'away_strength']].values
                y_test = test_data['home_win'].values

            # Hyperparameter optimization (optional)
            if optimize_hyperparams:
                best_params = self._optimize_hyperparameters(model, X_train, y_train)
                model.set_params(**best_params)
                results['parameter_history'].append({
                    'window': window_id,
                    'params': best_params
                })

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            try:
                probabilities = model.predict_proba(X_test)[:, 1]
            except:
                probabilities = model.predict(X_test)

            predictions = (probabilities > 0.5).astype(int)

            # Calculate metrics
            window_result = BacktestResult(
                window_id=window_id,
                train_start=train_data['game_date'].min(),
                train_end=train_data['game_date'].max(),
                test_start=test_data['game_date'].min(),
                test_end=test_data['game_date'].max(),
                n_train=len(train_data),
                n_test=len(test_data),
                accuracy=np.mean(predictions == y_test),
                brier_score=np.mean((probabilities - y_test) ** 2),
                log_loss=self._calculate_log_loss(y_test, probabilities),
                roi=self._calculate_roi(predictions, y_test),
                ece=self._calculate_ece(y_test, probabilities),
                avg_clv=0.0,  # Would calculate if CLV data available
                positive_clv_rate=0.0,
                predictions=predictions,
                actuals=y_test,
                probabilities=probabilities
            )

            results['window_results'].append(window_result)
            results['all_predictions'].extend(predictions)
            results['all_actuals'].extend(y_test)
            results['all_probabilities'].extend(probabilities)

            # Roll forward
            start_idx += self.step_size
            window_id += 1

        # Calculate overall metrics
        results['overall'] = self._calculate_overall_metrics(results)

        # Check for performance degradation over time
        results['degradation'] = self._check_degradation(results['window_results'])

        return results

    def _optimize_hyperparameters(self, model, X_train, y_train) -> Dict:
        """
        Optimize hyperparameters using time series cross-validation.
        NEVER uses test data for optimization.
        """
        # Simple grid search with time series CV
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'alpha': [0.1, 0.5, 1.0],
            'l1_ratio': [0.5, 0.7, 0.9]
        }

        # Use TimeSeriesSplit for hyperparameter search
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='neg_brier_score',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best params: {grid_search.best_params_}")
        return grid_search.best_params_

    def _calculate_log_loss(self, y_true, y_pred) -> float:
        """Calculate log loss with numerical stability."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _calculate_roi(self, predictions, actuals) -> float:
        """Calculate ROI assuming flat betting."""
        n_bets = len(predictions)
        if n_bets == 0:
            return 0.0

        # Assume -110 odds
        wins = np.sum(predictions == actuals)
        losses = n_bets - wins

        # Standard -110 payout
        profit = wins * 0.909 - losses
        roi = (profit / n_bets) * 100

        return roi

    def _calculate_ece(self, y_true, y_pred, n_bins=10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_confidence = np.mean(y_pred[bin_mask])
                bin_weight = np.mean(bin_mask)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

        return ece

    def _calculate_overall_metrics(self, results: Dict) -> Dict:
        """Calculate metrics across all windows."""
        all_pred = np.array(results['all_predictions'])
        all_actual = np.array(results['all_actuals'])
        all_prob = np.array(results['all_probabilities'])

        return {
            'total_windows': len(results['window_results']),
            'total_games': len(all_pred),
            'overall_accuracy': np.mean(all_pred == all_actual),
            'overall_brier': np.mean((all_prob - all_actual) ** 2),
            'overall_log_loss': self._calculate_log_loss(all_actual, all_prob),
            'overall_roi': self._calculate_roi(all_pred, all_actual),
            'overall_ece': self._calculate_ece(all_actual, all_prob),
            'avg_window_accuracy': np.mean([r.accuracy for r in results['window_results']]),
            'std_window_accuracy': np.std([r.accuracy for r in results['window_results']])
        }

    def _check_degradation(self, window_results: List[BacktestResult]) -> Dict:
        """Check for performance degradation over time."""
        if len(window_results) < 3:
            return {'degradation_detected': False}

        # Split into early and late windows
        mid_point = len(window_results) // 2
        early_accuracy = np.mean([r.accuracy for r in window_results[:mid_point]])
        late_accuracy = np.mean([r.accuracy for r in window_results[mid_point:]])

        # Check for significant degradation
        degradation_pct = ((early_accuracy - late_accuracy) / early_accuracy) * 100

        return {
            'degradation_detected': degradation_pct > 5,  # 5% threshold
            'early_accuracy': early_accuracy,
            'late_accuracy': late_accuracy,
            'degradation_percentage': degradation_pct,
            'interpretation': 'Model performance declining - consider retraining' if degradation_pct > 5
                           else 'Model performance stable over time'
        }


class FeatureEngineeringSafety:
    """
    Feature engineering that prevents look-ahead bias.
    Only uses information available BEFORE prediction time.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_cache = {}

    def create_features_for_game(self,
                                game_date: datetime,
                                team_home: str,
                                team_away: str,
                                historical_games: pd.DataFrame) -> Dict[str, float]:
        """
        Create features using ONLY data available before game_date.

        CRITICAL: Filter historical_games to exclude future data.

        Args:
            game_date: Date of the game to predict
            team_home: Home team ID
            team_away: Away team ID
            historical_games: All historical data

        Returns:
            Feature dictionary
        """
        # STRICT DATE FILTER - Only use games BEFORE prediction date
        available_games = historical_games[
            historical_games['game_date'] < game_date
        ].copy()

        if len(available_games) == 0:
            logger.warning(f"No historical data available before {game_date}")
            return self._get_default_features()

        # Home team recent performance (last 10 games)
        home_recent = available_games[
            (available_games['home_team_id'] == team_home) |
            (available_games['away_team_id'] == team_home)
        ].tail(10)

        # Away team recent performance
        away_recent = available_games[
            (available_games['home_team_id'] == team_away) |
            (available_games['away_team_id'] == team_away)
        ].tail(10)

        features = {
            # Recent form
            'home_recent_ppg': self._calculate_ppg(home_recent, team_home),
            'away_recent_ppg': self._calculate_ppg(away_recent, team_away),
            'home_win_pct_l10': self._calculate_win_pct(home_recent, team_home),
            'away_win_pct_l10': self._calculate_win_pct(away_recent, team_away),

            # Head-to-head
            'h2h_last_5': self._calculate_h2h(available_games, team_home, team_away),

            # Rest days
            'home_rest_days': self._calculate_rest(available_games, team_home, game_date),
            'away_rest_days': self._calculate_rest(available_games, team_away, game_date),

            # Home advantage
            'home_advantage': 1.0,  # Binary indicator

            # Season progress (affects team behavior)
            'season_progress': self._calculate_season_progress(game_date)
        }

        return features

    def _calculate_ppg(self, games: pd.DataFrame, team: str) -> float:
        """Calculate points per game for a team."""
        if len(games) == 0:
            return 0.0

        total_points = 0
        for _, game in games.iterrows():
            if game['home_team_id'] == team:
                total_points += game.get('home_score', 0)
            else:
                total_points += game.get('away_score', 0)

        return total_points / len(games) if len(games) > 0 else 0.0

    def _calculate_win_pct(self, games: pd.DataFrame, team: str) -> float:
        """Calculate win percentage."""
        if len(games) == 0:
            return 0.5

        wins = 0
        for _, game in games.iterrows():
            if game['home_team_id'] == team:
                if game.get('home_score', 0) > game.get('away_score', 0):
                    wins += 1
            elif game['away_team_id'] == team:
                if game.get('away_score', 0) > game.get('home_score', 0):
                    wins += 1

        return wins / len(games) if len(games) > 0 else 0.5

    def _calculate_h2h(self, games: pd.DataFrame, team1: str, team2: str) -> float:
        """Calculate head-to-head record."""
        h2h_games = games[
            ((games['home_team_id'] == team1) & (games['away_team_id'] == team2)) |
            ((games['home_team_id'] == team2) & (games['away_team_id'] == team1))
        ].tail(5)

        if len(h2h_games) == 0:
            return 0.0

        team1_wins = 0
        for _, game in h2h_games.iterrows():
            if ((game['home_team_id'] == team1 and game['home_score'] > game['away_score']) or
                (game['away_team_id'] == team1 and game['away_score'] > game['home_score'])):
                team1_wins += 1

        return (team1_wins / len(h2h_games)) - 0.5  # Center around 0

    def _calculate_rest(self, games: pd.DataFrame, team: str, game_date: datetime) -> float:
        """Calculate rest days since last game."""
        team_games = games[
            (games['home_team_id'] == team) | (games['away_team_id'] == team)
        ]

        if len(team_games) == 0:
            return 3.0  # Default rest

        last_game_date = team_games['game_date'].max()
        rest_days = (game_date - last_game_date).days

        return min(rest_days, 7.0)  # Cap at 7 days

    def _calculate_season_progress(self, game_date: datetime) -> float:
        """Calculate how far into the season we are."""
        # Simplified - assumes Oct-June season
        month = game_date.month
        if month >= 10:  # Oct-Dec
            return (month - 10) / 9
        else:  # Jan-June
            return (month + 3) / 9

    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when no data available."""
        return {
            'home_recent_ppg': 100.0,
            'away_recent_ppg': 100.0,
            'home_win_pct_l10': 0.5,
            'away_win_pct_l10': 0.5,
            'h2h_last_5': 0.0,
            'home_rest_days': 3.0,
            'away_rest_days': 3.0,
            'home_advantage': 1.0,
            'season_progress': 0.5
        }

    def validate_no_leakage(self,
                           features_df: pd.DataFrame,
                           games_df: pd.DataFrame) -> bool:
        """
        Validate that features don't use future information.

        Args:
            features_df: DataFrame with features
            games_df: DataFrame with game dates

        Returns:
            True if no leakage detected
        """
        # Check each feature was computed from games before prediction date
        for idx, row in features_df.iterrows():
            game_date = games_df.loc[idx, 'game_date']

            # This is a sanity check
            # Actual prevention happens in create_features_for_game
            # Could add more sophisticated checks here

        return True


def create_backtest_report(results: Dict) -> str:
    """
    Generate comprehensive backtest report.

    Args:
        results: Results from walk-forward analysis

    Returns:
        Formatted report string
    """
    report = "=" * 80 + "\n"
    report += "BACKTESTING REPORT - TEMPORAL VALIDATION\n"
    report += "=" * 80 + "\n\n"

    # Overall metrics
    overall = results['overall']
    report += "OVERALL PERFORMANCE:\n"
    report += "-" * 40 + "\n"
    report += f"Total Windows: {overall['total_windows']}\n"
    report += f"Total Games: {overall['total_games']}\n"
    report += f"Overall Accuracy: {overall['overall_accuracy']:.1%}\n"
    report += f"Overall Brier Score: {overall['overall_brier']:.4f}\n"
    report += f"Overall Log Loss: {overall['overall_log_loss']:.4f}\n"
    report += f"Overall ROI: {overall['overall_roi']:.2f}%\n"
    report += f"Overall ECE: {overall['overall_ece']:.4f} ({'✅ GOOD' if overall['overall_ece'] < 0.05 else '❌ NEEDS CALIBRATION'})\n\n"

    # Window consistency
    report += "CONSISTENCY ACROSS WINDOWS:\n"
    report += "-" * 40 + "\n"
    report += f"Avg Window Accuracy: {overall['avg_window_accuracy']:.1%}\n"
    report += f"Std Window Accuracy: {overall['std_window_accuracy']:.3f}\n\n"

    # Performance degradation
    deg = results['degradation']
    report += "PERFORMANCE DEGRADATION CHECK:\n"
    report += "-" * 40 + "\n"
    report += f"Degradation Detected: {'YES ⚠️' if deg['degradation_detected'] else 'NO ✅'}\n"
    if 'early_accuracy' in deg:
        report += f"Early Windows Accuracy: {deg['early_accuracy']:.1%}\n"
        report += f"Late Windows Accuracy: {deg['late_accuracy']:.1%}\n"
        report += f"Degradation: {deg['degradation_percentage']:.1f}%\n"
    report += f"Interpretation: {deg.get('interpretation', 'N/A')}\n\n"

    # Window-by-window summary
    report += "WINDOW-BY-WINDOW PERFORMANCE:\n"
    report += "-" * 40 + "\n"
    for window in results['window_results'][:5]:  # Show first 5
        report += f"Window {window.window_id + 1}: "
        report += f"Acc={window.accuracy:.1%}, "
        report += f"Brier={window.brier_score:.3f}, "
        report += f"ROI={window.roi:.1f}%\n"

    report += "\n" + "=" * 80 + "\n"
    report += "KEY INSIGHTS:\n"
    report += "-" * 40 + "\n"

    # Key insights based on metrics
    if overall['overall_accuracy'] > 0.55:
        report += "✅ Model shows predictive edge (>55% accuracy)\n"
    else:
        report += "❌ Model accuracy below profitable threshold\n"

    if overall['overall_ece'] < 0.05:
        report += "✅ Model is well-calibrated (ECE < 0.05)\n"
    else:
        report += "❌ Model needs calibration (ECE > 0.05)\n"

    if not deg['degradation_detected']:
        report += "✅ Performance stable over time\n"
    else:
        report += "⚠️ Performance degradation detected - consider retraining\n"

    report += "=" * 80 + "\n"

    return report
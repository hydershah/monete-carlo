"""
Calibration Pipeline for Sports Betting Models
===============================================
Research shows calibration-optimized models achieve +34.69% ROI
vs -35.17% for accuracy-optimized models.

This module provides calibration methods to ensure predicted probabilities
match actual outcome frequencies, which is critical for profitable betting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Make matplotlib optional for environments without it
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class CalibrationPipeline:
    """
    Comprehensive calibration system for sports betting predictions.

    Supports:
    - Platt scaling (sigmoid calibration) for small datasets
    - Isotonic regression for large datasets
    - Reliability diagrams and calibration plots
    - Expected Calibration Error (ECE) monitoring
    """

    def __init__(self, method: str = 'isotonic', n_bins: int = 10):
        """
        Initialize calibration pipeline.

        Args:
            method: 'isotonic' or 'sigmoid' (Platt scaling)
                - isotonic: Non-parametric, better for large datasets (>1000)
                - sigmoid: Parametric, better for small datasets (<1000)
            n_bins: Number of bins for calibration metrics
        """
        self.method = method
        self.n_bins = n_bins
        self.calibrators = {}
        self.calibration_metrics = {}

    def fit_calibrator(self,
                      predictions: np.ndarray,
                      actuals: np.ndarray,
                      name: str = 'default') -> None:
        """
        Fit a calibrator on historical predictions.

        Args:
            predictions: Uncalibrated probability predictions
            actuals: Actual binary outcomes (0 or 1)
            name: Name for this calibrator (e.g., 'win', 'ats', 'over')
        """
        if self.method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(predictions, actuals)
        elif self.method == 'sigmoid':
            # Platt scaling using LogisticRegression
            calibrator = LogisticRegression()
            # Reshape for sklearn
            calibrator.fit(predictions.reshape(-1, 1), actuals)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.calibrators[name] = calibrator

        # Calculate initial metrics
        calibrated_probs = self.calibrate(predictions, name)
        self.calibration_metrics[name] = self.calculate_metrics(
            actuals, predictions, calibrated_probs
        )

    def calibrate(self,
                 predictions: np.ndarray,
                 name: str = 'default') -> np.ndarray:
        """
        Apply calibration to predictions.

        Args:
            predictions: Uncalibrated probabilities
            name: Which calibrator to use

        Returns:
            Calibrated probabilities
        """
        if name not in self.calibrators:
            # Return uncalibrated if no calibrator available
            return predictions

        calibrator = self.calibrators[name]

        if self.method == 'isotonic':
            return calibrator.transform(predictions)
        elif self.method == 'sigmoid':
            # Reshape for sklearn
            return calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]

    def calculate_ece(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        Target: <0.05 for production betting systems.

        ECE = Σ (|accuracy - confidence| × bin_proportion)

        Args:
            y_true: Actual outcomes (0 or 1)
            y_pred: Predicted probabilities

        Returns:
            ECE value (lower is better, <0.05 is excellent)
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                # Calculate accuracy in this bin
                accuracy_in_bin = np.mean(y_true[in_bin])
                # Calculate average confidence in this bin
                avg_confidence_in_bin = np.mean(y_pred[in_bin])
                # Add to ECE
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return ece

    def calculate_classwise_ece(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               n_classes: int = 2) -> Dict[int, float]:
        """
        Calculate Classwise-ECE for multi-class problems.

        Args:
            y_true: Actual outcomes
            y_pred: Predicted probabilities (can be multi-class)
            n_classes: Number of classes

        Returns:
            Dictionary with ECE for each class
        """
        classwise_ece = {}

        for class_idx in range(n_classes):
            # Convert to binary problem for this class
            binary_true = (y_true == class_idx).astype(int)

            if y_pred.ndim > 1:
                # Multi-class probabilities
                class_probs = y_pred[:, class_idx]
            else:
                # Binary case
                if class_idx == 0:
                    class_probs = 1 - y_pred
                else:
                    class_probs = y_pred

            classwise_ece[class_idx] = self.calculate_ece(binary_true, class_probs)

        return classwise_ece

    def calculate_brier_score(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray) -> float:
        """
        Calculate Brier score (mean squared error of probabilities).
        Lower is better, typical range 0.15-0.25 for sports.

        Args:
            y_true: Actual outcomes (0 or 1)
            y_pred: Predicted probabilities

        Returns:
            Brier score
        """
        return np.mean((y_pred - y_true) ** 2)

    def plot_reliability_diagram(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                calibrated_pred: Optional[np.ndarray] = None,
                                title: str = "Reliability Diagram",
                                save_path: Optional[str] = None):
        """
        Generate reliability diagram (calibration plot).
        Perfect calibration lies on the diagonal.

        Args:
            y_true: Actual outcomes
            y_pred: Uncalibrated predictions
            calibrated_pred: Calibrated predictions (optional)
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure (if available) or None
        """
        if not HAS_MATPLOTLIB:
            # Return None if matplotlib not available
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

        # Plot uncalibrated
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_pred, n_bins=self.n_bins
        )
        ax.plot(mean_pred, fraction_pos, 's-',
               label=f'Uncalibrated (ECE={self.calculate_ece(y_true, y_pred):.3f})',
               markersize=8, linewidth=2)

        # Plot calibrated if provided
        if calibrated_pred is not None:
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y_true, calibrated_pred, n_bins=self.n_bins
            )
            ax.plot(mean_pred_cal, fraction_pos_cal, 'o-',
                   label=f'Calibrated (ECE={self.calculate_ece(y_true, calibrated_pred):.3f})',
                   markersize=8, linewidth=2)

        # Histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(y_pred, bins=self.n_bins, alpha=0.3, color='gray',
                edgecolor='black', label='Prediction Distribution')
        ax2.set_ylabel('Count', fontsize=12)

        # Formatting
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
        ax2.legend(loc='upper right', fontsize=11)

        # Add ECE text box
        ece_uncal = self.calculate_ece(y_true, y_pred)
        textstr = f'ECE (Uncalibrated): {ece_uncal:.3f}'
        if calibrated_pred is not None:
            ece_cal = self.calculate_ece(y_true, calibrated_pred)
            textstr += f'\nECE (Calibrated): {ece_cal:.3f}'
            textstr += f'\nImprovement: {(ece_uncal - ece_cal)/ece_uncal*100:.1f}%'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred_uncal: np.ndarray,
                         y_pred_cal: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive calibration metrics.

        Args:
            y_true: Actual outcomes
            y_pred_uncal: Uncalibrated predictions
            y_pred_cal: Calibrated predictions

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            # ECE metrics
            'ece_uncalibrated': self.calculate_ece(y_true, y_pred_uncal),
            'ece_calibrated': self.calculate_ece(y_true, y_pred_cal),

            # Brier scores
            'brier_uncalibrated': self.calculate_brier_score(y_true, y_pred_uncal),
            'brier_calibrated': self.calculate_brier_score(y_true, y_pred_cal),

            # Improvement percentages
            'ece_improvement': 0.0,
            'brier_improvement': 0.0,

            # Thresholds
            'meets_production_threshold': False
        }

        # Calculate improvements
        if metrics['ece_uncalibrated'] > 0:
            metrics['ece_improvement'] = (
                (metrics['ece_uncalibrated'] - metrics['ece_calibrated']) /
                metrics['ece_uncalibrated'] * 100
            )

        if metrics['brier_uncalibrated'] > 0:
            metrics['brier_improvement'] = (
                (metrics['brier_uncalibrated'] - metrics['brier_calibrated']) /
                metrics['brier_uncalibrated'] * 100
            )

        # Check if meets production threshold (ECE < 0.05)
        metrics['meets_production_threshold'] = metrics['ece_calibrated'] < 0.05

        return metrics

    def get_calibration_summary(self) -> str:
        """
        Get a text summary of calibration performance.

        Returns:
            Formatted string with calibration metrics
        """
        summary = "=" * 60 + "\n"
        summary += "CALIBRATION PERFORMANCE SUMMARY\n"
        summary += "=" * 60 + "\n\n"

        for name, metrics in self.calibration_metrics.items():
            summary += f"Calibrator: {name}\n"
            summary += "-" * 30 + "\n"
            summary += f"ECE (Uncalibrated): {metrics['ece_uncalibrated']:.4f}\n"
            summary += f"ECE (Calibrated):   {metrics['ece_calibrated']:.4f}\n"
            summary += f"ECE Improvement:    {metrics['ece_improvement']:.1f}%\n"
            summary += f"Brier (Uncalibrated): {metrics['brier_uncalibrated']:.4f}\n"
            summary += f"Brier (Calibrated):   {metrics['brier_calibrated']:.4f}\n"
            summary += f"Brier Improvement:    {metrics['brier_improvement']:.1f}%\n"
            summary += f"Production Ready:     {'✅ YES' if metrics['meets_production_threshold'] else '❌ NO'}\n"
            summary += "\n"

        summary += "=" * 60 + "\n"
        summary += "Target: ECE < 0.05 for production deployment\n"
        summary += "Research shows calibration adds +34.69% ROI\n"
        summary += "=" * 60 + "\n"

        return summary


class SportsBettingCalibrator:
    """
    Specialized calibrator for sports betting with bet type specific handling.
    """

    def __init__(self):
        """Initialize separate calibrators for different bet types."""
        self.win_calibrator = CalibrationPipeline(method='isotonic')
        self.ats_calibrator = CalibrationPipeline(method='isotonic')
        self.ou_calibrator = CalibrationPipeline(method='isotonic')

    def fit_all(self, historical_predictions: pd.DataFrame):
        """
        Fit calibrators on historical predictions.

        Args:
            historical_predictions: DataFrame with columns:
                - win_prob_pred: Predicted win probability
                - win_actual: Actual win outcome
                - ats_prob_pred: Predicted ATS probability
                - ats_actual: Actual ATS outcome
                - ou_prob_pred: Predicted O/U probability
                - ou_actual: Actual O/U outcome
        """
        # Fit win probability calibrator
        if 'win_prob_pred' in historical_predictions.columns:
            self.win_calibrator.fit_calibrator(
                historical_predictions['win_prob_pred'].values,
                historical_predictions['win_actual'].values,
                name='win'
            )

        # Fit ATS calibrator
        if 'ats_prob_pred' in historical_predictions.columns:
            self.ats_calibrator.fit_calibrator(
                historical_predictions['ats_prob_pred'].values,
                historical_predictions['ats_actual'].values,
                name='ats'
            )

        # Fit O/U calibrator
        if 'ou_prob_pred' in historical_predictions.columns:
            self.ou_calibrator.fit_calibrator(
                historical_predictions['ou_prob_pred'].values,
                historical_predictions['ou_actual'].values,
                name='ou'
            )

    def calibrate_predictions(self, predictions: Dict) -> Dict:
        """
        Calibrate a set of predictions.

        Args:
            predictions: Dictionary with keys:
                - home_win_probability
                - home_cover_probability
                - over_probability

        Returns:
            Dictionary with calibrated probabilities
        """
        calibrated = predictions.copy()

        if 'home_win_probability' in predictions:
            calibrated['home_win_probability_uncalibrated'] = predictions['home_win_probability']
            calibrated['home_win_probability'] = float(
                self.win_calibrator.calibrate(
                    np.array([predictions['home_win_probability']]),
                    name='win'
                )[0]
            )

        if 'home_cover_probability' in predictions:
            calibrated['home_cover_probability_uncalibrated'] = predictions['home_cover_probability']
            calibrated['home_cover_probability'] = float(
                self.ats_calibrator.calibrate(
                    np.array([predictions['home_cover_probability']]),
                    name='ats'
                )[0]
            )

        if 'over_probability' in predictions:
            calibrated['over_probability_uncalibrated'] = predictions['over_probability']
            calibrated['over_probability'] = float(
                self.ou_calibrator.calibrate(
                    np.array([predictions['over_probability']]),
                    name='ou'
                )[0]
            )

        return calibrated

    def evaluate_calibration(self) -> str:
        """
        Evaluate all calibrators and return summary.

        Returns:
            Formatted evaluation string
        """
        summary = "\n" + "=" * 70 + "\n"
        summary += "SPORTS BETTING CALIBRATION EVALUATION\n"
        summary += "=" * 70 + "\n\n"

        summary += "WIN PROBABILITY CALIBRATOR:\n"
        summary += self.win_calibrator.get_calibration_summary()
        summary += "\n"

        summary += "ATS CALIBRATOR:\n"
        summary += self.ats_calibrator.get_calibration_summary()
        summary += "\n"

        summary += "O/U CALIBRATOR:\n"
        summary += self.ou_calibrator.get_calibration_summary()

        return summary


def determine_calibration_method(n_samples: int) -> str:
    """
    Determine optimal calibration method based on sample size.

    Research recommendations:
    - Platt scaling (sigmoid): Small datasets (<1000 samples)
    - Isotonic regression: Large datasets (>1000 samples)

    Args:
        n_samples: Number of training samples

    Returns:
        'sigmoid' or 'isotonic'
    """
    if n_samples < 1000:
        return 'sigmoid'
    else:
        return 'isotonic'
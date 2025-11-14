#!/usr/bin/env python3
"""
Bankroll Simulator with Kelly Criterion Testing
================================================
Validates Kelly betting strategies through backtesting simulation.
Tests Full Kelly, Half Kelly, Quarter Kelly, and custom fractions.

Key Features:
- Bankroll growth tracking
- Drawdown analysis
- Risk of ruin calculation
- Sharpe ratio computation
- Kelly fraction optimization

Research:
- Full Kelly (1.0x): Theoretically optimal but high volatility
- Half Kelly (0.5x): 75% of Full Kelly growth, much lower variance
- Quarter Kelly (0.25x): ~50% of growth, very low bankruptcy risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


@dataclass
class BetResult:
    """Single bet result"""
    date: date
    game_id: str
    bet_type: str  # 'win', 'ats', 'ou'

    # Bet details
    stake: float  # Percentage of bankroll
    stake_amount: float  # Actual $ amount
    odds: float  # Decimal odds (e.g., 1.91 for -110)
    edge: float  # Our predicted edge

    # Prediction
    predicted_probability: float
    confidence: float

    # Outcome
    won: bool
    profit_loss: float  # $ amount
    profit_loss_pct: float  # % of original stake

    # Bankroll tracking
    bankroll_before: float
    bankroll_after: float
    bankroll_growth: float  # % growth


@dataclass
class BankrollSimulationResult:
    """Results from bankroll simulation"""
    kelly_fraction: float
    starting_bankroll: float
    ending_bankroll: float
    total_return: float  # Percentage

    # Performance metrics
    total_bets: int
    winning_bets: int
    win_rate: float
    avg_profit_per_bet: float
    total_profit: float

    # Risk metrics
    max_drawdown: float  # Percentage
    max_drawdown_duration: int  # Days
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # Return / Max Drawdown

    # Bankruptcy analysis
    bankrupt: bool
    bankruptcy_date: Optional[date]
    times_below_50pct: int
    times_below_25pct: int

    # Detailed tracking
    bet_history: List[BetResult]
    bankroll_history: pd.DataFrame

    # Comparison to target
    meets_sharpe_target: bool  # >1.5 Sharpe ratio
    meets_roi_target: bool  # Varies by sport


class BankrollSimulator:
    """
    Simulates betting with Kelly Criterion
    Tests different Kelly fractions to find optimal risk/reward balance
    """

    # American odds to decimal conversion
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    # Decimal to American odds
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))

    @staticmethod
    def calculate_kelly_stake(win_prob: float,
                             odds: float,
                             kelly_fraction: float = 0.5,
                             min_edge: float = 0.02,
                             max_stake: float = 0.02) -> float:
        """
        Calculate Kelly stake size

        Args:
            win_prob: Our predicted win probability (0-1)
            odds: Decimal odds (e.g., 1.91 for -110)
            kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly)
            min_edge: Minimum edge required to bet
            max_stake: Maximum stake as % of bankroll

        Returns:
            Stake size as percentage of bankroll (0-1)
        """
        # Calculate edge
        # Edge = (win_prob * (odds - 1)) - (1 - win_prob)
        b = odds - 1  # Net odds (payout per unit)
        p = win_prob  # Win probability
        q = 1 - p     # Loss probability

        # Full Kelly formula: (bp - q) / b
        full_kelly = (b * p - q) / b

        # Apply fraction
        fractional_kelly = full_kelly * kelly_fraction

        # Calculate implied edge
        implied_prob = 1 / odds
        edge = win_prob - implied_prob

        # Safety checks
        if edge < min_edge:
            return 0.0  # Edge too small

        if fractional_kelly < 0:
            return 0.0  # Negative Kelly = no bet

        # Cap at maximum
        stake = min(fractional_kelly, max_stake)

        return max(0.0, stake)

    def simulate_betting(self,
                        predictions_df: pd.DataFrame,
                        kelly_fraction: float = 0.5,
                        starting_bankroll: float = 10000.0,
                        min_edge: float = 0.02,
                        max_stake: float = 0.02) -> BankrollSimulationResult:
        """
        Simulate betting with given Kelly fraction

        Args:
            predictions_df: DataFrame with predictions and outcomes
                Required columns: date, predicted_prob, actual_outcome, odds, game_id
            kelly_fraction: Kelly fraction to use (0.5 = Half Kelly, 0.25 = Quarter Kelly)
            starting_bankroll: Starting bankroll in $
            min_edge: Minimum edge to place bet
            max_stake: Maximum stake % per bet

        Returns:
            BankrollSimulationResult with full tracking
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SIMULATING: {kelly_fraction:.2f}x Kelly")
        logger.info(f"Starting bankroll: ${starting_bankroll:,.2f}")
        logger.info(f"{'='*80}")

        # Sort by date
        predictions_df = predictions_df.sort_values('date').reset_index(drop=True)

        # Initialize tracking
        bankroll = starting_bankroll
        bet_history = []
        bankroll_history = []

        max_bankroll = starting_bankroll
        max_drawdown = 0.0
        drawdown_start = None
        max_drawdown_duration = 0

        times_below_50pct = 0
        times_below_25pct = 0
        bankrupt = False
        bankruptcy_date = None

        # Simulate each bet
        for idx, row in predictions_df.iterrows():
            # Get prediction details
            predicted_prob = row['predicted_prob']
            actual_outcome = row['actual_outcome']  # 1 = won, 0 = lost
            odds = row.get('odds', 1.91)  # Default -110 in decimal
            game_id = row.get('game_id', f'game_{idx}')
            bet_date = row['date']
            confidence = row.get('confidence', 0.5)
            bet_type = row.get('bet_type', 'win')

            # Calculate stake
            stake_pct = self.calculate_kelly_stake(
                win_prob=predicted_prob,
                odds=odds,
                kelly_fraction=kelly_fraction,
                min_edge=min_edge,
                max_stake=max_stake
            )

            # Skip if stake is 0 (no edge or negative Kelly)
            if stake_pct == 0:
                continue

            # Place bet
            stake_amount = bankroll * stake_pct
            bankroll_before = bankroll

            # Determine outcome
            won = (actual_outcome == 1)

            if won:
                profit_loss = stake_amount * (odds - 1)
            else:
                profit_loss = -stake_amount

            # Update bankroll
            bankroll += profit_loss
            bankroll_growth = (bankroll / bankroll_before - 1)

            # Track bet
            bet_result = BetResult(
                date=bet_date,
                game_id=game_id,
                bet_type=bet_type,
                stake=stake_pct,
                stake_amount=stake_amount,
                odds=odds,
                edge=(predicted_prob - (1/odds)),
                predicted_probability=predicted_prob,
                confidence=confidence,
                won=won,
                profit_loss=profit_loss,
                profit_loss_pct=(profit_loss / stake_amount),
                bankroll_before=bankroll_before,
                bankroll_after=bankroll,
                bankroll_growth=bankroll_growth
            )

            bet_history.append(bet_result)

            # Track bankroll history
            bankroll_history.append({
                'date': bet_date,
                'bankroll': bankroll,
                'cumulative_return': (bankroll / starting_bankroll - 1),
                'bet_count': len(bet_history)
            })

            # Update max bankroll for drawdown calculation
            if bankroll > max_bankroll:
                max_bankroll = bankroll
                drawdown_start = None

            # Calculate current drawdown
            current_drawdown = (max_bankroll - bankroll) / max_bankroll

            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown

                if drawdown_start:
                    duration = (bet_date - drawdown_start).days
                    if duration > max_drawdown_duration:
                        max_drawdown_duration = duration

            if current_drawdown > 0 and drawdown_start is None:
                drawdown_start = bet_date
            elif current_drawdown == 0:
                drawdown_start = None

            # Check bankruptcy thresholds
            if bankroll < starting_bankroll * 0.5:
                times_below_50pct += 1

            if bankroll < starting_bankroll * 0.25:
                times_below_25pct += 1

            if bankroll <= 0:
                bankrupt = True
                bankruptcy_date = bet_date
                logger.warning(f"💀 BANKRUPT on {bet_date} after {len(bet_history)} bets")
                break

        # Create bankroll history DataFrame
        bankroll_df = pd.DataFrame(bankroll_history)

        # Calculate performance metrics
        total_return = (bankroll / starting_bankroll - 1)
        winning_bets = sum(1 for bet in bet_history if bet.won)
        win_rate = winning_bets / len(bet_history) if bet_history else 0
        total_profit = bankroll - starting_bankroll
        avg_profit_per_bet = total_profit / len(bet_history) if bet_history else 0

        # Calculate Sharpe ratio
        if len(bankroll_df) > 1:
            returns = bankroll_df['cumulative_return'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Calculate Sortino ratio (only downside volatility)
        if len(bankroll_df) > 1:
            returns = bankroll_df['cumulative_return'].pct_change().dropna()
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = 0.0

        # Calculate Calmar ratio (Return / Max Drawdown)
        if max_drawdown > 0:
            calmar_ratio = total_return / max_drawdown
        else:
            calmar_ratio = float('inf') if total_return > 0 else 0.0

        # Check targets
        meets_sharpe_target = sharpe_ratio >= 1.5
        meets_roi_target = total_return >= 0.10  # 10% minimum

        result = BankrollSimulationResult(
            kelly_fraction=kelly_fraction,
            starting_bankroll=starting_bankroll,
            ending_bankroll=bankroll,
            total_return=total_return,
            total_bets=len(bet_history),
            winning_bets=winning_bets,
            win_rate=win_rate,
            avg_profit_per_bet=avg_profit_per_bet,
            total_profit=total_profit,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            bankrupt=bankrupt,
            bankruptcy_date=bankruptcy_date,
            times_below_50pct=times_below_50pct,
            times_below_25pct=times_below_25pct,
            bet_history=bet_history,
            bankroll_history=bankroll_df,
            meets_sharpe_target=meets_sharpe_target,
            meets_roi_target=meets_roi_target
        )

        # Log results
        logger.info(f"\n{'='*80}")
        logger.info(f"RESULTS: {kelly_fraction:.2f}x Kelly")
        logger.info(f"{'='*80}")
        logger.info(f"Ending bankroll: ${bankroll:,.2f}")
        logger.info(f"Total return: {total_return:.2%}")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f} {'✅' if meets_sharpe_target else '❌'} (target: >1.5)")
        logger.info(f"Max drawdown: {max_drawdown:.2%}")
        logger.info(f"Calmar ratio: {calmar_ratio:.2f}")
        logger.info(f"Bankrupt: {bankrupt}")
        logger.info(f"{'='*80}\n")

        return result

    def compare_kelly_fractions(self,
                               predictions_df: pd.DataFrame,
                               kelly_fractions: List[float] = [1.0, 0.75, 0.5, 0.25, 0.125],
                               starting_bankroll: float = 10000.0) -> Dict[float, BankrollSimulationResult]:
        """
        Compare multiple Kelly fractions

        Args:
            predictions_df: Predictions and outcomes
            kelly_fractions: List of Kelly fractions to test
            starting_bankroll: Starting bankroll

        Returns:
            Dictionary mapping fraction to results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPARING KELLY FRACTIONS")
        logger.info(f"Testing: {kelly_fractions}")
        logger.info(f"{'='*80}\n")

        results = {}

        for fraction in kelly_fractions:
            result = self.simulate_betting(
                predictions_df=predictions_df,
                kelly_fraction=fraction,
                starting_bankroll=starting_bankroll
            )
            results[fraction] = result

        # Print comparison table
        self._print_comparison_table(results)

        return results

    def _print_comparison_table(self, results: Dict[float, BankrollSimulationResult]):
        """Print comparison table of Kelly fractions"""

        logger.info(f"\n{'='*80}")
        logger.info("KELLY FRACTION COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"{'Fraction':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Bankrupt':>10}")
        logger.info("-" * 80)

        for fraction, result in sorted(results.items(), reverse=True):
            logger.info(
                f"{fraction:<12.2f} "
                f"{result.total_return:>9.2%} "
                f"{result.sharpe_ratio:>8.2f} "
                f"{result.max_drawdown:>9.2%} "
                f"{'YES' if result.bankrupt else 'NO':>10}"
            )

        logger.info("=" * 80)

        # Recommendation
        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        best_return = max(results.items(), key=lambda x: x[1].total_return if not x[1].bankrupt else -float('inf'))

        logger.info(f"\n📊 RECOMMENDATIONS:")
        logger.info(f"  Best Sharpe ratio: {best_sharpe[0]:.2f}x Kelly ({best_sharpe[1].sharpe_ratio:.2f})")
        logger.info(f"  Best return: {best_return[0]:.2f}x Kelly ({best_return[1].total_return:.2%})")
        logger.info(f"  ⚠️  Full Kelly (1.0x): Generally too risky - high bankruptcy rate")
        logger.info(f"  ✅ Half Kelly (0.5x): Good balance of growth and safety")
        logger.info(f"  ✅ Quarter Kelly (0.25x): Very safe, lower returns\n")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate sample predictions
    np.random.seed(42)
    n_bets = 1000

    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_bets),
        'game_id': [f'game_{i}' for i in range(n_bets)],
        'predicted_prob': np.random.uniform(0.5, 0.65, n_bets),  # Positive edge
        'actual_outcome': np.random.binomial(1, 0.55, n_bets),  # 55% true win rate
        'odds': np.random.uniform(1.85, 1.95, n_bets),  # Around -110
        'confidence': np.random.uniform(0.6, 0.9, n_bets),
        'bet_type': ['win'] * n_bets
    })

    # Run simulation
    simulator = BankrollSimulator()

    results = simulator.compare_kelly_fractions(
        predictions_df=sample_data,
        kelly_fractions=[1.0, 0.5, 0.25],
        starting_bankroll=10000.0
    )

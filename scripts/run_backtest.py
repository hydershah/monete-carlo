#!/usr/bin/env python3
"""
Automated Backtesting Pipeline
===============================
Complete end-to-end backtesting workflow:
1. Load historical data
2. Train base models (Elo, Poisson, Logistic, Pythagorean)
3. Generate out-of-sample predictions
4. Train meta-ensemble
5. Train calibrators
6. Run walk-forward validation
7. Calculate performance metrics
8. Generate comprehensive report

Usage:
    python scripts/run_backtest.py --league NBA --years 3
    python scripts/run_backtest.py --league NFL --start-date 2020-01-01 --model ensemble
    python scripts/run_backtest.py --all-models --save-predictions
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import logging
import json
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulations.backtest_orchestrator import BacktestOrchestrator
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive backtesting pipeline'
    )

    # League selection
    parser.add_argument(
        '--league',
        type=str,
        choices=['NBA', 'NFL', 'MLB', 'NHL', 'NCAAF', 'NCAAB'],
        default='NBA',
        help='League to backtest'
    )

    # Date range
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). Default: 3 years ago'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). Default: today'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='Number of years to backtest (if start-date not specified)'
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['elo', 'poisson', 'logistic', 'pythagorean', 'ensemble', 'all'],
        default='all',
        help='Model to backtest'
    )

    # Backtest parameters
    parser.add_argument(
        '--train-window',
        type=int,
        default=500,
        help='Number of games for training window'
    )

    parser.add_argument(
        '--test-window',
        type=int,
        default=100,
        help='Number of games for test window'
    )

    parser.add_argument(
        '--step-size',
        type=int,
        default=50,
        help='Number of games to roll forward each iteration'
    )

    # Output options
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to database'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='backtest_results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'html', 'all'],
        default='json',
        help='Output format for results'
    )

    # Database
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.getenv('DATABASE_URL', 'postgresql://hyder@localhost:5432/gamelens_ai'),
        help='Database URL'
    )

    args = parser.parse_args()

    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    else:
        start_date = date.today() - timedelta(days=args.years * 365)

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = date.today()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("=" * 80)
    logger.info("BACKTESTING PIPELINE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"League: {args.league}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Models: {args.model}")
    logger.info(f"Train window: {args.train_window} games")
    logger.info(f"Test window: {args.test_window} games")
    logger.info(f"Step size: {args.step_size} games")
    logger.info(f"Save predictions: {args.save_predictions}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Initialize orchestrator
    orchestrator = BacktestOrchestrator(
        db_url=args.db_url,
        league=args.league
    )

    # Determine which models to run
    if args.model == 'all':
        models = ['elo', 'poisson', 'logistic', 'pythagorean', 'ensemble']
    else:
        models = [args.model]

    # Run backtests
    results = {}
    for model_name in models:
        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTESTING: {model_name.upper()}")
        logger.info(f"{'='*80}\n")

        try:
            model_results = orchestrator.run_backtest(
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
                train_window=args.train_window,
                test_window=args.test_window,
                step_size=args.step_size,
                save_predictions=args.save_predictions
            )

            results[model_name] = model_results

            # Save individual model results
            _save_results(
                results=model_results,
                model_name=model_name,
                output_dir=output_dir,
                format=args.format
            )

        except Exception as e:
            logger.error(f"❌ Error backtesting {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[model_name] = {'error': str(e)}

    # Generate comparison report
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING COMPARISON REPORT")
    logger.info(f"{'='*80}\n")

    comparison = _generate_comparison_report(results, args.league)

    # Save comparison
    comparison_file = output_dir / f"comparison_{args.league}_{date.today()}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"✅ Comparison report saved: {comparison_file}")

    # Print summary
    _print_summary(results)

    logger.info(f"\n{'='*80}")
    logger.info("✅ BACKTESTING PIPELINE COMPLETE")
    logger.info(f"{'='*80}\n")
    logger.info(f"Results saved to: {output_dir}")


def _save_results(results: Dict,
                 model_name: str,
                 output_dir: Path,
                 format: str = 'json'):
    """Save results in specified format(s)"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{model_name}_{timestamp}"

    # JSON format
    if format in ['json', 'all']:
        json_file = output_dir / f"{base_filename}.json"

        # Convert numpy arrays to lists for JSON serialization
        json_results = _prepare_for_json(results)

        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"  ✅ JSON saved: {json_file}")

    # CSV format
    if format in ['csv', 'all']:
        csv_file = output_dir / f"{base_filename}.csv"

        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'model': [model_name],
            'accuracy': [results.get('overall_accuracy', 0)],
            'brier_score': [results.get('overall_brier_score', 0)],
            'roi': [results.get('overall_roi', 0)],
            'n_predictions': [results.get('total_predictions', 0)],
            'n_windows': [results.get('n_windows', 0)]
        })

        summary_df.to_csv(csv_file, index=False)
        logger.info(f"  ✅ CSV saved: {csv_file}")

    # HTML format (basic table)
    if format in ['html', 'all']:
        html_file = output_dir / f"{base_filename}.html"

        html_content = _generate_html_report(results, model_name)

        with open(html_file, 'w') as f:
            f.write(html_content)

        logger.info(f"  ✅ HTML saved: {html_file}")


def _prepare_for_json(obj):
    """Recursively convert numpy arrays to lists for JSON serialization"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _prepare_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_prepare_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj


def _generate_comparison_report(results: Dict, league: str) -> Dict:
    """Generate comparison report across all models"""

    comparison = {
        'league': league,
        'generated_at': datetime.now().isoformat(),
        'models': {}
    }

    for model_name, model_results in results.items():
        if 'error' in model_results:
            comparison['models'][model_name] = {'error': model_results['error']}
            continue

        comparison['models'][model_name] = {
            'accuracy': model_results.get('overall_accuracy', 0),
            'brier_score': model_results.get('overall_brier_score', 0),
            'roi': model_results.get('overall_roi', 0),
            'n_predictions': model_results.get('total_predictions', 0),
            'n_windows': model_results.get('n_windows', 0)
        }

    # Rank models
    valid_models = {
        k: v for k, v in comparison['models'].items()
        if 'error' not in v
    }

    if valid_models:
        comparison['rankings'] = {
            'by_accuracy': sorted(
                valid_models.items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            ),
            'by_brier_score': sorted(
                valid_models.items(),
                key=lambda x: x[1]['brier_score']
            ),
            'by_roi': sorted(
                valid_models.items(),
                key=lambda x: x[1]['roi'],
                reverse=True
            )
        }

    return comparison


def _generate_html_report(results: Dict, model_name: str) -> str:
    """Generate basic HTML report"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report - {model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        </style>
    </head>
    <body>
        <h1>Backtest Report: {model_name.upper()}</h1>

        <h2>Overall Performance</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td class="metric">{results.get('overall_accuracy', 0):.2%}</td>
            </tr>
            <tr>
                <td>Brier Score</td>
                <td class="metric">{results.get('overall_brier_score', 0):.4f}</td>
            </tr>
            <tr>
                <td>ROI</td>
                <td class="metric">{results.get('overall_roi', 0):.2%}</td>
            </tr>
            <tr>
                <td>Total Predictions</td>
                <td>{results.get('total_predictions', 0):,}</td>
            </tr>
            <tr>
                <td>Number of Windows</td>
                <td>{results.get('n_windows', 0)}</td>
            </tr>
        </table>

        <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """

    return html


def _print_summary(results: Dict):
    """Print summary table to console"""

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 80)

    # Header
    logger.info(f"{'Model':<15} {'Accuracy':>12} {'Brier':>10} {'ROI':>10} {'Predictions':>12}")
    logger.info("-" * 80)

    # Results
    for model_name, model_results in results.items():
        if 'error' in model_results:
            logger.info(f"{model_name:<15} {'ERROR':<45}")
            continue

        accuracy = model_results.get('overall_accuracy', 0)
        brier = model_results.get('overall_brier_score', 0)
        roi = model_results.get('overall_roi', 0)
        n_pred = model_results.get('total_predictions', 0)

        logger.info(
            f"{model_name:<15} "
            f"{accuracy:>11.2%} "
            f"{brier:>10.4f} "
            f"{roi:>9.2%} "
            f"{n_pred:>12,}"
        )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()

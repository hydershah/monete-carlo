#!/usr/bin/env python3
"""
Comprehensive Backtest Report Generator
========================================
Generates detailed reports comparing all models and Kelly strategies.

Output formats:
- HTML with charts
- JSON with full data
- CSV summaries
- Console summary

Includes:
- Model performance comparison
- Kelly criterion analysis
- Target metric validation (73-75% NBA, 71.5% NFL, <0.20 Brier, >1.5 Sharpe)
- Visualization charts
- Recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """Generate comprehensive backtest reports"""

    # Target metrics by sport
    TARGET_METRICS = {
        'NBA': {
            'accuracy': 0.735,  # 73-75%
            'accuracy_range': (0.73, 0.75),
            'brier_score': 0.20,  # <0.20
            'sharpe_ratio': 1.5   # >1.5
        },
        'NFL': {
            'accuracy': 0.715,  # 71.5%
            'accuracy_range': (0.70, 0.73),
            'brier_score': 0.20,
            'sharpe_ratio': 1.5
        },
        'MLB': {
            'accuracy': 0.580,  # Baseball is harder
            'accuracy_range': (0.55, 0.60),
            'brier_score': 0.22,
            'sharpe_ratio': 1.3
        },
        'NHL': {
            'accuracy': 0.590,
            'accuracy_range': (0.56, 0.62),
            'brier_score': 0.21,
            'sharpe_ratio': 1.4
        }
    }

    def __init__(self, league: str = 'NBA'):
        """
        Initialize report generator

        Args:
            league: League for target metrics
        """
        self.league = league
        self.targets = self.TARGET_METRICS.get(league, self.TARGET_METRICS['NBA'])

    def generate_full_report(self,
                            model_results: Dict,
                            kelly_results: Optional[Dict] = None,
                            output_dir: Path = Path('backtest_results'),
                            include_charts: bool = True) -> Dict:
        """
        Generate comprehensive backtest report

        Args:
            model_results: Results from all models
            kelly_results: Results from Kelly testing
            output_dir: Directory to save reports
            include_charts: Generate chart images

        Returns:
            Dictionary with report data
        """
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING COMPREHENSIVE BACKTEST REPORT")
        logger.info(f"{'='*80}\n")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate sections
        report = {
            'metadata': self._generate_metadata(),
            'executive_summary': self._generate_executive_summary(model_results, kelly_results),
            'model_comparison': self._generate_model_comparison(model_results),
            'kelly_analysis': self._generate_kelly_analysis(kelly_results) if kelly_results else None,
            'target_validation': self._validate_targets(model_results),
            'recommendations': self._generate_recommendations(model_results, kelly_results),
            'detailed_results': model_results
        }

        # Save reports
        self._save_json_report(report, output_dir)
        self._save_html_report(report, output_dir, include_charts)
        self._save_csv_summaries(report, output_dir)

        # Print to console
        self._print_console_summary(report)

        logger.info(f"\n✅ Reports saved to: {output_dir}\n")

        return report

    def _generate_metadata(self) -> Dict:
        """Generate report metadata"""
        return {
            'generated_at': datetime.now().isoformat(),
            'league': self.league,
            'target_metrics': self.targets,
            'generator_version': '1.0.0'
        }

    def _generate_executive_summary(self,
                                   model_results: Dict,
                                   kelly_results: Optional[Dict]) -> Dict:
        """Generate executive summary"""

        # Find best model
        best_model = None
        best_accuracy = 0
        best_sharpe = 0

        for model_name, results in model_results.items():
            if 'error' in results:
                continue

            accuracy = results.get('overall_accuracy', 0)
            sharpe = results.get('sharpe_ratio', 0)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name

            if sharpe > best_sharpe:
                best_sharpe = sharpe

        # Kelly summary
        kelly_summary = None
        if kelly_results:
            # Find best Kelly fraction
            best_kelly = max(
                kelly_results.items(),
                key=lambda x: x[1].sharpe_ratio if not x[1].bankrupt else -1
            )
            kelly_summary = {
                'best_fraction': best_kelly[0],
                'sharpe_ratio': best_kelly[1].sharpe_ratio,
                'total_return': best_kelly[1].total_return,
                'max_drawdown': best_kelly[1].max_drawdown
            }

        return {
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'best_sharpe': best_sharpe,
            'meets_targets': best_accuracy >= self.targets['accuracy'] and best_sharpe >= self.targets['sharpe_ratio'],
            'kelly_recommendation': kelly_summary
        }

    def _generate_model_comparison(self, model_results: Dict) -> pd.DataFrame:
        """Generate model comparison table"""

        comparison_data = []

        for model_name, results in model_results.items():
            if 'error' in results:
                continue

            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': results.get('overall_accuracy', 0),
                'Brier Score': results.get('overall_brier_score', 0),
                'ROI': results.get('overall_roi', 0),
                'Sharpe Ratio': results.get('sharpe_ratio', 0),
                'Predictions': results.get('total_predictions', 0),
                'Windows': results.get('n_windows', 0)
            })

        df = pd.DataFrame(comparison_data)

        # Sort by accuracy descending
        df = df.sort_values('Accuracy', ascending=False)

        return df

    def _generate_kelly_analysis(self, kelly_results: Dict) -> Dict:
        """Generate Kelly criterion analysis"""

        analysis = {
            'fractions_tested': list(kelly_results.keys()),
            'results': {}
        }

        for fraction, result in kelly_results.items():
            analysis['results'][fraction] = {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'bankrupt': result.bankrupt,
                'calmar_ratio': result.calmar_ratio
            }

        # Determine recommendation
        # Prefer fractions with good Sharpe and no bankruptcy
        safe_fractions = {
            k: v for k, v in kelly_results.items()
            if not v.bankrupt
        }

        if safe_fractions:
            best = max(safe_fractions.items(), key=lambda x: x[1].sharpe_ratio)
            analysis['recommended_fraction'] = best[0]
            analysis['recommendation_reason'] = f"Best Sharpe ratio ({best[1].sharpe_ratio:.2f}) with no bankruptcy"
        else:
            analysis['recommended_fraction'] = min(kelly_results.keys())
            analysis['recommendation_reason'] = "All fractions led to bankruptcy - use minimum"

        return analysis

    def _validate_targets(self, model_results: Dict) -> Dict:
        """Validate against target metrics"""

        validation = {
            'league': self.league,
            'targets': self.targets,
            'results': {}
        }

        for model_name, results in model_results.items():
            if 'error' in results:
                continue

            accuracy = results.get('overall_accuracy', 0)
            brier = results.get('overall_brier_score', 1.0)
            sharpe = results.get('sharpe_ratio', 0)

            meets_accuracy = (
                self.targets['accuracy_range'][0] <= accuracy <= self.targets['accuracy_range'][1]
            )
            meets_brier = brier < self.targets['brier_score']
            meets_sharpe = sharpe >= self.targets['sharpe_ratio']

            validation['results'][model_name] = {
                'accuracy': {
                    'value': accuracy,
                    'target': self.targets['accuracy'],
                    'meets_target': meets_accuracy,
                    'difference': accuracy - self.targets['accuracy']
                },
                'brier_score': {
                    'value': brier,
                    'target': self.targets['brier_score'],
                    'meets_target': meets_brier,
                    'difference': brier - self.targets['brier_score']
                },
                'sharpe_ratio': {
                    'value': sharpe,
                    'target': self.targets['sharpe_ratio'],
                    'meets_target': meets_sharpe,
                    'difference': sharpe - self.targets['sharpe_ratio']
                },
                'overall_pass': meets_accuracy and meets_brier and meets_sharpe
            }

        return validation

    def _generate_recommendations(self,
                                 model_results: Dict,
                                 kelly_results: Optional[Dict]) -> Dict:
        """Generate recommendations"""

        recommendations = {
            'model_selection': [],
            'kelly_fraction': None,
            'next_steps': [],
            'warnings': []
        }

        # Model recommendations
        passing_models = []
        for model_name, results in model_results.items():
            if 'error' in results:
                continue

            accuracy = results.get('overall_accuracy', 0)
            sharpe = results.get('sharpe_ratio', 0)

            if accuracy >= self.targets['accuracy'] and sharpe >= self.targets['sharpe_ratio']:
                passing_models.append(model_name)

        if passing_models:
            recommendations['model_selection'].append(
                f"✅ Models meeting targets: {', '.join(passing_models)}"
            )
            recommendations['model_selection'].append(
                f"Recommend using: {passing_models[0].upper()}"
            )
        else:
            recommendations['model_selection'].append(
                "❌ No models meet all targets - need more training data or hyperparameter tuning"
            )

        # Kelly recommendations
        if kelly_results:
            kelly_analysis = self._generate_kelly_analysis(kelly_results)
            recommended_fraction = kelly_analysis['recommended_fraction']
            recommendations['kelly_fraction'] = {
                'fraction': recommended_fraction,
                'reason': kelly_analysis['recommendation_reason']
            }

            if recommended_fraction >= 0.5:
                recommendations['warnings'].append(
                    "⚠️  Half Kelly or higher has significant risk - monitor drawdowns closely"
                )

        # Next steps
        recommendations['next_steps'] = [
            "1. Review model performance across different time periods",
            "2. Test on out-of-sample data from current season",
            "3. Implement real-time CLV tracking",
            "4. Set up automated model retraining pipeline",
            "5. Start with paper trading before live deployment"
        ]

        return recommendations

    def _save_json_report(self, report: Dict, output_dir: Path):
        """Save JSON report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"backtest_report_{self.league}_{timestamp}.json"

        # Convert non-serializable objects
        def convert(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        serializable_report = json.loads(
            json.dumps(report, default=convert)
        )

        with open(filename, 'w') as f:
            json.dump(serializable_report, f, indent=2)

        logger.info(f"  ✅ JSON report: {filename}")

    def _save_html_report(self, report: Dict, output_dir: Path, include_charts: bool):
        """Save HTML report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"backtest_report_{self.league}_{timestamp}.html"

        html = self._generate_html(report, include_charts)

        with open(filename, 'w') as f:
            f.write(html)

        logger.info(f"  ✅ HTML report: {filename}")

    def _generate_html(self, report: Dict, include_charts: bool) -> str:
        """Generate HTML report"""

        model_comparison_df = report['model_comparison']

        # Create HTML table
        if isinstance(model_comparison_df, pd.DataFrame):
            table_html = model_comparison_df.to_html(index=False, classes='table', float_format=lambda x: f'{x:.4f}')
        else:
            table_html = "<p>No comparison data available</p>"

        # Executive summary
        exec_summary = report['executive_summary']

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {self.league}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
                h2 {{ color: #34495e; margin-top: 40px; }}
                .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #ecf0f1; border-radius: 8px; min-width: 200px; }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #2ecc71; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
                td {{ border: 1px solid #ddd; padding: 12px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .pass {{ color: #2ecc71; font-weight: bold; }}
                .fail {{ color: #e74c3c; font-weight: bold; }}
                .recommendation {{ background: #fff9e6; border-left: 4px solid #f39c12; padding: 15px; margin: 20px 0; }}
                .warning {{ background: #ffe6e6; border-left: 4px solid #e74c3c; padding: 15px; margin: 20px 0; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏀 Backtest Report: {self.league}</h1>

                <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>

                <h2>📊 Executive Summary</h2>

                <div class="metric">
                    <div class="metric-label">Best Model</div>
                    <div class="metric-value">{exec_summary['best_model'] or 'N/A'}</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Best Accuracy</div>
                    <div class="metric-value">{exec_summary['best_accuracy']:.2%}</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Best Sharpe</div>
                    <div class="metric-value">{exec_summary['best_sharpe']:.2f}</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Meets Targets</div>
                    <div class="metric-value {'pass' if exec_summary['meets_targets'] else 'fail'}">
                        {'✅ YES' if exec_summary['meets_targets'] else '❌ NO'}
                    </div>
                </div>

                <h2>📈 Model Comparison</h2>
                {table_html}

                <h2>🎯 Target Validation</h2>
                <p>Target metrics for {self.league}:</p>
                <ul>
                    <li>Accuracy: {self.targets['accuracy']:.1%}</li>
                    <li>Brier Score: <{self.targets['brier_score']:.2f}</li>
                    <li>Sharpe Ratio: >{self.targets['sharpe_ratio']:.1f}</li>
                </ul>

                <h2>💡 Recommendations</h2>
                <div class="recommendation">
                    <h3>Model Selection:</h3>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in report['recommendations']['model_selection'])}
                    </ul>
                </div>

                {''.join(f'<div class="warning">⚠️  {warning}</div>' for warning in report['recommendations'].get('warnings', []))}

                <h2>🚀 Next Steps</h2>
                <ol>
                    {''.join(f'<li>{step}</li>' for step in report['recommendations']['next_steps'])}
                </ol>

                <div class="footer">
                    <p>Generated by GameLens.ai Backtesting System v1.0</p>
                    <p>Target: 73-75% NBA | 71.5% NFL | <0.20 Brier | >1.5 Sharpe</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _save_csv_summaries(self, report: Dict, output_dir: Path):
        """Save CSV summaries"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Model comparison
        if isinstance(report['model_comparison'], pd.DataFrame):
            filename = output_dir / f"model_comparison_{self.league}_{timestamp}.csv"
            report['model_comparison'].to_csv(filename, index=False)
            logger.info(f"  ✅ CSV summary: {filename}")

    def _print_console_summary(self, report: Dict):
        """Print summary to console"""

        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTEST SUMMARY - {self.league}")
        logger.info(f"{'='*80}\n")

        # Executive summary
        exec_summary = report['executive_summary']
        logger.info(f"Best Model: {exec_summary['best_model']}")
        logger.info(f"Best Accuracy: {exec_summary['best_accuracy']:.2%}")
        logger.info(f"Best Sharpe: {exec_summary['best_sharpe']:.2f}")
        logger.info(f"Meets Targets: {'✅ YES' if exec_summary['meets_targets'] else '❌ NO'}\n")

        # Model comparison
        if isinstance(report['model_comparison'], pd.DataFrame):
            logger.info("MODEL COMPARISON:")
            logger.info(report['model_comparison'].to_string(index=False))

        logger.info(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Sample results
    sample_results = {
        'elo': {
            'overall_accuracy': 0.725,
            'overall_brier_score': 0.185,
            'overall_roi': 0.08,
            'sharpe_ratio': 1.6,
            'total_predictions': 500,
            'n_windows': 10
        },
        'ensemble': {
            'overall_accuracy': 0.745,
            'overall_brier_score': 0.175,
            'overall_roi': 0.12,
            'sharpe_ratio': 1.8,
            'total_predictions': 500,
            'n_windows': 10
        }
    }

    generator = BacktestReportGenerator(league='NBA')
    report = generator.generate_full_report(
        model_results=sample_results,
        output_dir=Path('test_reports')
    )

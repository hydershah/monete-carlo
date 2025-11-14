#!/usr/bin/env python3
"""
Historical Data Ingestion Script
=================================
Fetches historical sports data from ESPN API and populates the database
for backtesting purposes.

Usage:
    # Fetch 3 seasons of NBA data (default)
    python scripts/ingest_historical_data.py

    # Fetch 5 seasons of NFL data
    python scripts/ingest_historical_data.py --league nfl --seasons 5

    # Fetch multiple leagues
    python scripts/ingest_historical_data.py --league nba nfl mlb

    # Save to CSV only (no database)
    python scripts/ingest_historical_data.py --csv-only

    # Use custom database URL
    python scripts/ingest_historical_data.py --db-url postgresql://user:pass@host/db
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.historical_data_fetcher import HistoricalDataFetcher
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fetch historical sports data for backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 3 seasons of NBA data (default)
  python scripts/ingest_historical_data.py

  # Fetch 5 seasons of NFL data
  python scripts/ingest_historical_data.py --league nfl --seasons 5

  # Fetch multiple leagues
  python scripts/ingest_historical_data.py --league nba nfl mlb --seasons 2

  # Save to CSV only (no database)
  python scripts/ingest_historical_data.py --csv-only

  # Custom output directory
  python scripts/ingest_historical_data.py --output-dir ./my_data
        """
    )

    parser.add_argument(
        '--league',
        nargs='+',
        default=['nba'],
        choices=['nba', 'nfl', 'mlb', 'nhl'],
        help='League(s) to fetch data for (default: nba)'
    )

    parser.add_argument(
        '--seasons',
        type=int,
        default=3,
        help='Number of seasons to fetch (default: 3)'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        help='Starting season year (default: current_year - seasons)'
    )

    parser.add_argument(
        '--csv-only',
        action='store_true',
        help='Save to CSV only, skip database insertion'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/historical',
        help='Output directory for CSV files (default: data/historical)'
    )

    parser.add_argument(
        '--db-url',
        type=str,
        help='Database URL (default: from DATABASE_URL env var)'
    )

    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Delay between API requests in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate CSV files, do not fetch new data'
    )

    return parser.parse_args()


def print_header(text: str):
    """Print formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def print_validation_report(validation: dict, league: str):
    """Print data validation report."""
    print(f"\n📊 Data Validation Report - {league.upper()}")
    print("-" * 70)
    print(f"Total Games:       {validation['total_games']:,}")
    print(f"Date Range:        {validation['date_range']['start']} to {validation['date_range']['end']}")
    print(f"Span:              {validation['date_range']['span_days']:,} days")
    print(f"Seasons:           {validation['seasons_covered']}")
    print(f"Teams:             {validation['teams_count']}")

    print(f"\nQuality Checks:")
    print(f"  Duplicate Games: {validation['quality_checks']['duplicate_games']}")
    print(f"  Invalid Scores:  {validation['quality_checks']['invalid_scores']}")
    print(f"  Future Dates:    {validation['quality_checks']['future_dates']}")

    print(f"\nGames per Season:")
    for season, count in sorted(validation['completeness']['games_per_season'].items()):
        print(f"  {season}-{int(season)+1}: {count:,} games")

    print(f"\nMissing Data:")
    print(f"  Scores:          {validation['completeness']['missing_scores']}")
    print(f"  Spreads:         {validation['completeness']['missing_odds']['spread']}")
    print(f"  Totals:          {validation['completeness']['missing_odds']['total']}")


def main():
    """Main execution function."""
    args = parse_args()

    print_header("🏀 HISTORICAL DATA INGESTION FOR BACKTESTING 🏀")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize fetcher
    logger.info(f"Initializing fetcher (rate limit: {args.rate_limit}s)")
    fetcher = HistoricalDataFetcher(
        db_url=args.db_url,
        rate_limit_delay=args.rate_limit
    )

    # Process each league
    total_games_fetched = 0
    total_games_saved = 0

    for league in args.league:
        print_header(f"Processing {league.upper()}")

        logger.info(f"League: {league.upper()}")
        logger.info(f"Seasons: {args.seasons}")
        if args.start_year:
            logger.info(f"Start Year: {args.start_year}")

        # Fetch data
        try:
            logger.info(f"\n🔄 Fetching {league.upper()} data...")
            logger.info(f"⏱️  This may take 10-20 minutes due to API rate limiting...")

            games = fetcher.fetch_historical_games(
                league=league,
                seasons=args.seasons,
                start_year=args.start_year
            )

            total_games_fetched += len(games)

            if not games:
                logger.warning(f"⚠️  No games fetched for {league.upper()}")
                continue

            # Validate data
            validation = fetcher.validate_data_quality(games)
            print_validation_report(validation, league)

            # Save to CSV
            csv_filename = f"{league}_historical_{args.seasons}seasons.csv"
            csv_path = os.path.join(args.output_dir, csv_filename)

            logger.info(f"\n💾 Saving to CSV: {csv_path}")
            fetcher.save_to_csv(games, csv_path)

            # Save to database (unless csv-only mode)
            if not args.csv_only:
                logger.info(f"\n💾 Saving to database...")
                saved_count = fetcher.save_to_database(games)
                total_games_saved += saved_count

                if saved_count < len(games):
                    logger.warning(
                        f"⚠️  Only {saved_count}/{len(games)} games saved to database"
                    )
            else:
                logger.info("📄 CSV-only mode: Skipping database insertion")

        except Exception as e:
            logger.error(f"❌ Error processing {league.upper()}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print_header("✅ INGESTION COMPLETE")

    print(f"Leagues Processed:     {', '.join([l.upper() for l in args.league])}")
    print(f"Total Games Fetched:   {total_games_fetched:,}")

    if not args.csv_only:
        print(f"Total Games Saved:     {total_games_saved:,}")

    print(f"Output Directory:      {args.output_dir}")

    print("\n🎉 Historical data is ready for backtesting!")
    print("\nNext steps:")
    print("  1. Review the CSV files in the output directory")
    print("  2. Run backtesting with: python -m src.simulations.backtesting")
    print("  3. Check the validation reports for data quality\n")


if __name__ == "__main__":
    main()

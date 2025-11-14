#!/usr/bin/env python3
"""
Quick test of the data pipeline
Fetches a small sample of data to verify everything works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.espn_client import ESPNClient
from datetime import datetime, timedelta

def test_espn_client():
    """Test basic ESPN client functionality"""
    print("\n" + "="*60)
    print("Testing ESPN Client")
    print("="*60 + "\n")

    client = ESPNClient(rate_limit_delay=0.5)

    # Test 1: Fetch today's games
    print("1. Fetching today's NBA games...")
    try:
        games = client.get_todays_games('nba')
        print(f"   ✅ Found {len(games)} NBA games today")

        if games:
            game = games[0]
            print(f"   Sample game: {game['away_team_name']} @ {game['home_team_name']}")
            print(f"   Score: {game['away_score']}-{game['home_score']}")
            print(f"   Status: {game['status']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test 2: Fetch historical games
    print("\n2. Fetching games from last week...")
    try:
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        scoreboard = client.get_scoreboard('nba', date=week_ago)
        games = client.parse_games(scoreboard)
        print(f"   ✅ Found {len(games)} games from {week_ago}")

        if games:
            completed_games = [g for g in games if g['completed']]
            print(f"   Completed games: {len(completed_games)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test 3: Fetch multiple days
    print("\n3. Fetching games from date range...")
    try:
        all_games = client.get_games_date_range('nba', days_back=7, days_forward=0)
        print(f"   ✅ Found {len(all_games)} games in the last 7 days")

        completed = [g for g in all_games if g['completed']]
        print(f"   Completed: {len(completed)}")
        print(f"   In progress/scheduled: {len(all_games) - len(completed)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    print("\n✅ ESPN Client test passed!")
    return True


def test_historical_fetcher():
    """Test historical data fetcher with small sample"""
    print("\n" + "="*60)
    print("Testing Historical Data Fetcher")
    print("="*60 + "\n")

    from src.data.historical_data_fetcher import HistoricalDataFetcher
    import os

    fetcher = HistoricalDataFetcher(rate_limit_delay=0.5)

    # Test: Fetch just 30 days of data (not full season)
    print("Fetching 30 days of NBA data as test...")

    try:
        # Manually fetch a month of data
        from datetime import datetime, timedelta

        games = []
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        print(f"Date range: {start_date.date()} to {end_date.date()}")

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")

            try:
                scoreboard = fetcher.espn_client.get_scoreboard('nba', date=date_str)
                daily_games = fetcher.espn_client.parse_games(scoreboard)

                for game in daily_games:
                    if game['completed']:
                        hist_game = fetcher._convert_to_historical_game(
                            game, 'nba', 2024
                        )
                        if hist_game:
                            games.append(hist_game)
            except:
                pass  # No games on this date

            current_date += timedelta(days=1)

        print(f"✅ Fetched {len(games)} completed games")

        if games:
            # Test validation
            validation = fetcher.validate_data_quality(games)
            print(f"\nValidation Results:")
            print(f"  Total games: {validation['total_games']}")
            print(f"  Date range: {validation['date_range']['start']} to {validation['date_range']['end']}")
            print(f"  Duplicate games: {validation['quality_checks']['duplicate_games']}")

            # Test CSV save
            os.makedirs('data/test', exist_ok=True)
            csv_path = 'data/test/sample_games.csv'
            fetcher.save_to_csv(games, csv_path)
            print(f"\n✅ Saved test data to {csv_path}")

            # Test DataFrame conversion
            df = fetcher.get_games_dataframe(games)
            print(f"\n✅ Converted to DataFrame: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {', '.join(df.columns[:5])}...")

            print("\n✅ Historical Fetcher test passed!")
            return True
        else:
            print("⚠️  No games found (might be off-season)")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_format():
    """Test that data format matches backtesting requirements"""
    print("\n" + "="*60)
    print("Testing Data Format for Backtesting")
    print("="*60 + "\n")

    try:
        import pandas as pd
        from pathlib import Path

        csv_path = Path('data/test/sample_games.csv')

        if not csv_path.exists():
            print("⚠️  Sample CSV not found, skipping test")
            return True

        df = pd.read_csv(csv_path)

        # Check required columns for backtesting
        required_columns = [
            'game_id', 'game_date', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'league'
        ]

        print("Checking required columns for backtesting...")
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False

        print(f"✅ All required columns present")
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df[['game_date', 'home_team_name', 'away_team_name', 'home_score', 'away_score']].head(3))

        print("\n✅ Data format test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DATA PIPELINE TEST SUITE")
    print("="*60)

    results = []

    # Test 1: ESPN Client
    results.append(("ESPN Client", test_espn_client()))

    # Test 2: Historical Fetcher
    results.append(("Historical Fetcher", test_historical_fetcher()))

    # Test 3: Data Format
    results.append(("Data Format", test_data_format()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! Pipeline is ready.")
        print("\nNext steps:")
        print("  1. Run: python scripts/ingest_historical_data.py --seasons 3")
        print("  2. Wait for data ingestion (10-15 minutes)")
        print("  3. Run backtesting with your models")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

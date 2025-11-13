"""
Command-line interface for sports prediction system.
"""

import argparse
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
from tabulate import tabulate

# Load environment variables
load_dotenv()

from .models import init_db, get_db_context, Game, Prediction, Team
from .data.ingestion import DataIngestion, TheOddsAPIClient
from .simulations.hybrid_predictor import HybridPredictor, predict_todays_games


def setup_logger():
    """Configure logger for CLI."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )


def cmd_init_db(args):
    """Initialize database tables."""
    logger.info("Initializing database...")
    init_db()
    logger.info("✓ Database initialized successfully")


def cmd_fetch(args):
    """Fetch game data from APIs."""
    sport = args.sport
    date_str = args.date

    logger.info(f"Fetching {sport.upper()} data...")

    # Initialize data ingestion
    odds_client = None
    if args.with_odds:
        try:
            odds_client = TheOddsAPIClient()
            logger.info("✓ TheOddsAPI client initialized")
        except ValueError as e:
            logger.warning(f"Could not initialize odds client: {e}")

    ingestion = DataIngestion(odds_client=odds_client)

    with get_db_context() as db:
        # Fetch teams
        logger.info("Fetching teams...")
        teams_count = ingestion.ingest_teams(sport, db)
        logger.info(f"✓ Fetched {teams_count} teams")

        # Fetch games
        if date_str == "today":
            date_str = datetime.now().strftime("%Y%m%d")
        elif date_str == "tomorrow":
            date_str = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")

        logger.info(f"Fetching games for {date_str}...")
        games_count = ingestion._ingest_games_with_db(sport, date_str, db)
        logger.info(f"✓ Fetched {games_count} games")

        # Fetch odds if requested
        if args.with_odds and odds_client:
            logger.info("Fetching odds...")
            odds_count = ingestion._ingest_odds_with_db(sport, db)
            logger.info(f"✓ Updated {odds_count} games with odds")

    logger.info("✓ Data fetch complete")


def cmd_predict(args):
    """Make predictions for games."""
    sport = args.sport
    use_gpt = not args.no_gpt

    if use_gpt:
        logger.info("GPT analysis enabled (will use API tokens)")
    else:
        logger.info("GPT analysis disabled")

    # Get predictions
    logger.info(f"Predicting {sport.upper()} games...")
    predictions = predict_todays_games(sport, use_gpt=use_gpt)

    if not predictions:
        logger.warning("No games to predict")
        return

    # Display predictions
    print(f"\n{'=' * 80}")
    print(f"  {sport.upper()} PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'=' * 80}\n")

    for pred in predictions:
        game = pred["game"]
        prediction = pred["prediction"]
        rec = pred["recommendation"]

        print(f"{game['away_team']} @ {game['home_team']}")
        print(f"  Time: {game['date']}")
        print(f"  Prediction: {prediction['final_home_win_prob']:.1%} home / "
              f"{prediction['final_away_win_prob']:.1%} away")
        print(f"  Confidence: {prediction['confidence_level']:.1%}")
        print(f"  Expected Score: {prediction['predicted_home_score']:.1f} - "
              f"{prediction['predicted_away_score']:.1f}")

        if use_gpt and prediction.get('gpt_adjustment'):
            print(f"  GPT Adjustment: {prediction['gpt_adjustment']:+.2%}")

        print(f"  Recommendation: {rec}")
        print()

    logger.info(f"✓ Predicted {len(predictions)} games")


def cmd_results(args):
    """View prediction results."""
    sport = args.sport
    days = args.days

    with get_db_context() as db:
        # Get recent predictions
        from sqlalchemy import and_, cast, Date
        from datetime import date

        start_date = date.today() - timedelta(days=days)

        predictions = db.query(Prediction).join(Game).filter(
            and_(
                Game.sport == sport.lower(),
                cast(Prediction.prediction_date, Date) >= start_date
            )
        ).order_by(Prediction.prediction_date.desc()).limit(20).all()

        if not predictions:
            logger.warning(f"No predictions found for last {days} days")
            return

        # Display results
        table_data = []
        for pred in predictions:
            game = pred.game

            # Determine if prediction was correct
            if game.home_score is not None and game.away_score is not None:
                actual_home_win = game.home_score > game.away_score
                predicted_home_win = pred.final_home_win_prob > 0.5
                correct = "✓" if actual_home_win == predicted_home_win else "✗"
                result = f"{game.home_score}-{game.away_score}"
            else:
                correct = "-"
                result = "Pending"

            table_data.append([
                pred.prediction_date.strftime("%Y-%m-%d"),
                f"{game.away_team.abbreviation} @ {game.home_team.abbreviation}",
                f"{pred.final_home_win_prob:.1%}",
                result,
                correct,
                f"{pred.confidence_level:.1%}",
            ])

        print(f"\n{'=' * 80}")
        print(f"  PREDICTION RESULTS - Last {days} days")
        print(f"{'=' * 80}\n")

        headers = ["Date", "Matchup", "Predicted", "Result", "Correct", "Confidence"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

        # Calculate accuracy
        correct_count = sum(1 for row in table_data if row[4] == "✓")
        total_completed = sum(1 for row in table_data if row[4] in ["✓", "✗"])

        if total_completed > 0:
            accuracy = correct_count / total_completed
            print(f"\nAccuracy: {correct_count}/{total_completed} ({accuracy:.1%})")


def cmd_list_games(args):
    """List games in database."""
    sport = args.sport
    date_str = args.date

    with get_db_context() as db:
        from sqlalchemy import and_, cast, Date
        from datetime import date

        if date_str == "today":
            target_date = date.today()
        elif date_str == "tomorrow":
            target_date = date.today() + timedelta(days=1)
        else:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        games = db.query(Game).filter(
            and_(
                Game.sport == sport.lower(),
                cast(Game.game_date, Date) == target_date
            )
        ).all()

        if not games:
            logger.warning(f"No {sport.upper()} games found for {target_date}")
            return

        print(f"\n{'=' * 80}")
        print(f"  {sport.upper()} GAMES - {target_date}")
        print(f"{'=' * 80}\n")

        for game in games:
            status = game.status
            if game.home_score is not None:
                score = f"{game.away_score}-{game.home_score}"
            else:
                score = ""
                time_str = game.game_date.strftime("%I:%M %p") if game.game_date else ""

            print(f"{game.away_team.name} @ {game.home_team.name}")
            print(f"  Status: {status}  {score}  {time_str}")
            print(f"  Venue: {game.venue or 'N/A'}")
            print()


def cmd_stats(args):
    """Show system statistics."""
    with get_db_context() as db:
        from sqlalchemy import func

        stats = {
            "teams": db.query(func.count(Team.id)).scalar(),
            "games": db.query(func.count(Game.id)).scalar(),
            "predictions": db.query(func.count(Prediction.id)).scalar(),
        }

        # Get prediction accuracy
        correct_preds = db.query(func.count(Prediction.id)).filter(
            Prediction.prediction_correct == True
        ).scalar()

        total_evaluated = db.query(func.count(Prediction.id)).filter(
            Prediction.prediction_correct.isnot(None)
        ).scalar()

        accuracy = (correct_preds / total_evaluated * 100) if total_evaluated > 0 else 0

        print(f"\n{'=' * 50}")
        print("  SYSTEM STATISTICS")
        print(f"{'=' * 50}\n")
        print(f"  Teams:       {stats['teams']:,}")
        print(f"  Games:       {stats['games']:,}")
        print(f"  Predictions: {stats['predictions']:,}")
        print(f"\n  Accuracy:    {correct_preds}/{total_evaluated} ({accuracy:.1f}%)")
        print()


def main():
    """Main CLI entry point."""
    setup_logger()

    parser = argparse.ArgumentParser(
        description="Sports Prediction System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database
  python -m src.cli init

  # Fetch NBA games for today
  python -m src.cli fetch --sport nba --date today --with-odds

  # Make predictions
  python -m src.cli predict --sport nba

  # View results
  python -m src.cli results --sport nba --days 7

  # List games
  python -m src.cli games --sport nba --date today
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Init command
    parser_init = subparsers.add_parser("init", help="Initialize database")
    parser_init.set_defaults(func=cmd_init_db)

    # Fetch command
    parser_fetch = subparsers.add_parser("fetch", help="Fetch game data")
    parser_fetch.add_argument("--sport", default="nba", help="Sport (nba, nfl, mlb, etc.)")
    parser_fetch.add_argument("--date", default="today", help="Date (YYYYMMDD, today, tomorrow)")
    parser_fetch.add_argument("--with-odds", action="store_true", help="Fetch betting odds")
    parser_fetch.set_defaults(func=cmd_fetch)

    # Predict command
    parser_predict = subparsers.add_parser("predict", help="Make predictions")
    parser_predict.add_argument("--sport", default="nba", help="Sport")
    parser_predict.add_argument("--no-gpt", action="store_true", help="Disable GPT analysis")
    parser_predict.set_defaults(func=cmd_predict)

    # Results command
    parser_results = subparsers.add_parser("results", help="View prediction results")
    parser_results.add_argument("--sport", default="nba", help="Sport")
    parser_results.add_argument("--days", type=int, default=7, help="Days to look back")
    parser_results.set_defaults(func=cmd_results)

    # Games command
    parser_games = subparsers.add_parser("games", help="List games")
    parser_games.add_argument("--sport", default="nba", help="Sport")
    parser_games.add_argument("--date", default="today", help="Date (YYYY-MM-DD, today, tomorrow)")
    parser_games.set_defaults(func=cmd_list_games)

    # Stats command
    parser_stats = subparsers.add_parser("stats", help="Show system statistics")
    parser_stats.set_defaults(func=cmd_stats)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

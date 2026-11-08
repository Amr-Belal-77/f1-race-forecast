"""Command-line entry point. All paths with spaces work when quoted by the shell."""

import argparse
import sys
from pathlib import Path

import requests

from .data import collect_history
from .pipeline import (
    DEFAULT_CONFIG,
    evaluate,
    load_config,
    predict,
    prepare_features,
    run_pipeline,
    train,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="F1 data-to-prediction research pipeline")
    parser.add_argument(
        "command", choices=["run", "collect", "features", "train", "evaluate", "predict"]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Race TOML config")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts"), help="Generated artifacts directory"
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh API cache for collect")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "collect":
            data = config["data"]
            frame = collect_history(
                data.get("start_season", 2018),
                data.get("end_season", config["race"]["season"]),
                data["cutoff"],
                args.output / "api_cache",
                args.refresh,
            )
            # Collection produces a separate file; the committed frozen input is preserved.
            args.output.mkdir(parents=True, exist_ok=True)
            destination = args.output / "collected_history.csv"
            frame.to_csv(destination, index=False)
            print(
                f"Collected {len(frame)} rows to {destination}. Set data.history in your config to use it."
            )
        elif args.command == "run":
            run_pipeline(config, args.output)
        else:
            history, features = prepare_features(config, args.output)
            if args.command == "train":
                train(config, args.output, features)
            elif args.command == "evaluate":
                evaluate(config, args.output, features)
            elif args.command == "predict":
                predict(config, args.output, history)
            print(f"{args.command.capitalize()} complete: {args.output.resolve()}")
    except (OSError, ValueError, KeyError, requests.RequestException) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0

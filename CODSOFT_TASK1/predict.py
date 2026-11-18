"""Command-line genre prediction with the saved pipeline.

Examples:
    python predict.py --plot "A detective hunts a serial killer through the rainy city."
    python predict.py --title "Space Farm" --plot "Farmers on Mars fight a dust storm." --top-k 5
    python predict.py --input "data/raw/Genre Classification Dataset/test_data.txt" --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import MODEL_PATH, TOP_K
from src.data_loader import load_dataset
from src.predictor import GenrePredictor, InvalidInputError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", help="Plot summary to classify")
    parser.add_argument("--title", default="", help="Optional movie title")
    parser.add_argument("--input", help="File in test_data.txt format (ID ::: TITLE ::: DESCRIPTION)")
    parser.add_argument("--output", default="predictions.csv", help="CSV path for --input mode")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args(argv)

    if not args.plot and not args.input:
        parser.error("provide --plot or --input")

    try:
        predictor = GenrePredictor.load(args.model_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.input:
        df, report = load_dataset(args.input, "test")
        results = predictor.predict_batch(df["title"], df["description"], top_k=args.top_k)
        out = pd.DataFrame(
            {
                "id": df["id"],
                "title": df["title"],
                "predicted_genre": [r["genre"] for r in results],
                "confidence": [round(r["confidence"], 4) for r in results],
            }
        )
        out.to_csv(args.output, index=False)
        print(f"Wrote {len(out)} predictions to {args.output} ({len(report.malformed)} malformed lines skipped)")
        return 0

    try:
        result = predictor.predict(args.plot, title=args.title, top_k=args.top_k)
    except InvalidInputError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    print(f"Predicted genre: {result['genre']}  (confidence {result['confidence']:.1%})")
    print(f"Top {len(result['top_k'])}:")
    for rank, item in enumerate(result["top_k"], 1):
        print(f"  {rank}. {item['genre']:<12} {item['score']:.1%}")
    print("Note: predictions are model estimates, not guaranteed classifications.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

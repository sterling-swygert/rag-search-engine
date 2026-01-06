import argparse
import json
from lib.hybrid_search import *
from lib import constants

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalizes a list of values")
    normalize_parser.add_argument("values", type=float, nargs="+", help="Text to embed")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search (Not Implemented)")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, help="Weight for BM25 score", default=constants.DEAFAULT_ALPHA)
    weighted_search_parser.add_argument("--limit", type=int, help="Number of results to return", default=constants.DEFAULT_SEARCH_LIMIT)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_values = normalize_values(args.values)
            for v in normalized_values:
                print(f"* {v:.4f}")

        case "weighted-search":
            with open(constants.MOVIES_PATH, "r") as f:
                documents = json.loads(f.read()).get('movies', [])
            results = HybridSearch(documents).weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results):
                doc_id = res[0]
                scores = res[1]
                doc = next((doc for doc in documents if doc['id'] == doc_id), None)
                if doc:
                    print(f"{i}. {doc['title']}")
                    print(f"  Hybrid Score: {scores['hybrid_score']:.4f}")
                    print(f"  BM25: {scores['bm25_score']:.4f}, Semantic: {scores['semantic_score']:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
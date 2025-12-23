#!/usr/bin/env python3

import argparse
import math

from indexing.inverted import InvertedIndex, tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID to get term frequency in", default=None)
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to get", default=None)
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")

    args = parser.parse_args()

    index = InvertedIndex()


    match args.command:
        case "search":
            index.load()
            print(f"Searching for: {args.query}")
            tokens = tokenize(args.query)
            count = 0
            matched_movies = []
            matched_movie_ids = []
            for token in tokens:
                current_matched_movie_ids = index.get_documents(token)
                for movie_id in current_matched_movie_ids:
                    if movie_id in matched_movie_ids:
                        continue
                    matched_movies.append(index.docmap[movie_id])
                    matched_movie_ids.append(movie_id)
                    count += 1
                if count >= 5:
                    break

            for movie in sorted(matched_movies[:5], key=lambda x: x['id']):
                print(f"Found: {movie.get('title')}")

        case "build":
            print("Building the index...")
            index.build()
            index.save()
            print("Index built and saved.")
            # Example query after building

        case "tf":
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)

        case "idf":
            index.load()
            token = tokenize(args.term)[0]
            df = len(index.get_documents(token))
            N = len(index.docmap)
            idf = math.log((N + 1) / (df + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            index.load()
            token = tokenize(args.term)[0]
            tf = index.get_tf(args.doc_id, token)
            df = len(index.get_documents(token))
            N = len(index.docmap)
            idf = math.log((N + 1) / (df + 1))
            tf_idf = tf * idf
            print(f"TF-IDF of term '{args.term}' in document {args.doc_id}: {tf_idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

import argparse
import json
import re

from lib import constants
from lib.semantic_search import verify_model, embed_text, SemanticSearch, ChunkedSemanticSearch
from lib.utils import chunk_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model loads correctly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text using the semantic search model")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings can be created and loaded")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query and display its embedding")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, help="Number of results to return", default=constants.DEFAULT_SEARCH_LIMIT)

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, help="Size of each chunk", default=constants.DEFAULT_CHUNK_SIZE)
    chunk_parser.add_argument("--overlap", type=int, help="Size of overlap between neighboring chunks", default=constants.DEFAULT_OVERLAP_SIZE)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text semantically using the model")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, help="Maximum size of each chunk", default=constants.DEFAULT_SEMANTIC_CHUNK_SIZE)   
    semantic_chunk_parser.add_argument("--overlap", type=int, help="Size of overlap between neighboring chunks", default=constants.DEFAULT_OVERLAP_SIZE)

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed text chunks using the semantic search model")
    

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            semanticSearch = SemanticSearch()
            with open(constants.MOVIES_PATH, "r") as f:
                documents = json.loads(f.read()).get('movies', [])
                embeddings = semanticSearch.load_or_create_embeddings(documents)
                print(f"Number of docs:   {len(documents)}")
                print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
        case "embedquery":
            semanticSearch = SemanticSearch()
            semanticSearch.embed_query(args.query)
        case "search":
            semanticSearch = SemanticSearch()
            with open(constants.MOVIES_PATH, "r") as f:
                documents = json.loads(f.read()).get('movies', [])
                semanticSearch.load_or_create_embeddings(documents)
                results = semanticSearch.search(args.query, limit=args.limit)
                for i, result in enumerate(results):
                    print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                    print(f"    {result['description']}\n")
        case "chunk":
            words = args.text.split()
            chunks = chunk_text(words, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")

        case "semantic_chunk":
            semanticSearch = SemanticSearch()
            chunks = chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")

        case "embed_chunks":
            chunkedsSmanticSearch = ChunkedSemanticSearch()
            with open(constants.MOVIES_PATH, "r") as f:
                documents = json.loads(f.read()).get('movies', [])
                chunk_embeddings = chunkedsSmanticSearch.build_chunk_embeddings(documents)
            print(f"Generated {len(chunk_embeddings)} chunked embeddings")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
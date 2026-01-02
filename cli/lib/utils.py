import json
import re
from typing import List

from lib import constants


def load_movies() -> list[dict]:
    with open(constants.MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(constants.STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
    

def fixed_size_chunking(
    text: str,
    chunk_size: int = constants.DEFAULT_CHUNK_SIZE,
    overlap: int = constants.DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks
    

def chunk_text(
    text: str,
    chunk_size: int = constants.DEFAULT_CHUNK_SIZE,
    overlap: int = constants.DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk(
    text: str,
    max_chunk_size: int = constants.DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = constants.DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = constants.DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = constants.DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


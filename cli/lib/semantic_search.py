
import json
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List, Dict, Any

from . import constants
from .utils import *


class SemanticSearch(object):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def generate_embedding(self, text) -> List[Tensor]:
        if text.strip() == "":
            raise ValueError("Input text cannot be empty")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.documents_map = {doc['id']: doc for doc in documents}
        doc_strings = [doc["title"] + ': ' + doc["description"] for doc in documents]
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        if not os.path.exists(constants.CACHE_DIR):
            os.makedirs(constants.CACHE_DIR)
        np.save(constants.EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.documents_map = {doc['id'] for doc in documents}
        need_to_build = True
        if os.path.exists(constants.EMBEDDINGS_PATH):
            self.embeddings = np.load(constants.EMBEDDINGS_PATH)
            if self.embeddings.shape[0] == len(documents):
                need_to_build = False
        if need_to_build:
            self.build_embeddings(documents)
        return self.embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.generate_embedding(query)
        print(f"Query: {query}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Shape: {embedding.shape}")
        return embedding
    
    def search(self, query: str, limit: int = 5) -> List[dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        similarity_score_docs_lst = []

        for doc, emb in zip(self.documents, self.embeddings):
            similarity = cosine_similarity(query_embedding, emb)
            similarity_score_docs_lst.append((similarity, doc))

        similarity_score_docs_lst.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": sim, 
                "title": doc["title"],
                "description": doc["description"]
            } 
            for sim, doc in similarity_score_docs_lst[:limit]
        ]  


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.documents_map = {doc['id']: doc for doc in documents}
        chunks = []
        chunks_metadata = []
        count = 0
        for doc in documents:
            description = doc.get("description")
            if description:
                doc_chunks = semantic_chunk(description)
                for i, chunk in enumerate(doc_chunks):
                    count += 1
                    chunks.append(chunk)
                    chunks_metadata.append({
                        "doc_id": doc['id'],
                        "chunk_index": i,
                        "total_chunk_idx": count,
                        "total_chunks": len(doc_chunks)
                    })

        self.chunk_metadata = chunks_metadata
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)

        if not os.path.exists(constants.CACHE_DIR):
            os.makedirs(constants.CACHE_DIR)

        np.save(constants.CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(constants.CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: List[dict]) -> np.ndarray:
        self.documents = documents
        self.documents_map = {doc['id']: doc for doc in documents}
        if os.path.exists(constants.CHUNK_EMBEDDINGS_PATH) and os.path.exists(constants.CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(constants.CHUNK_EMBEDDINGS_PATH)
            with open(constants.CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data.get("chunks", [])
        else:
            self.build_chunk_embeddings(documents)
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.generate_embedding(query)
        chunk_score_lst = []
        for chunk_meta, chunk_emb in zip(self.chunk_metadata, self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_emb)
            chunk_score_lst.append(
                {
                    "movie_idx": chunk_meta["doc_id"],
                    "chunk_index": chunk_meta["chunk_index"],
                    "score": similarity
                }
            )

        score_map = {}
        for item in chunk_score_lst:
            movie_idx = item["movie_idx"]
            score = item["score"]
            if movie_idx not in score_map or score > score_map[movie_idx]:
                score_map[movie_idx] = score
        top_sorted_idx = [idx for idx in dict(sorted(score_map.items(), key=lambda x: x[1], reverse=True)).keys()][:limit]

        results = []
        for idx in top_sorted_idx:
            doc = self.documents_map[idx]
            results.append({
                "id": idx,
                "score": round(score_map[idx], constants.DEFAULT_SCORE_PRECISION),
                "title": doc["title"],
                "description": doc["description"][:constants.DEFAULT_DESCRIPTION_PREVIEW_LENGTH],
                "metadata": {}
            })
        return results


def verify_model():
    semanticSearch = SemanticSearch()
    print(f"Model loaded: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")


def embed_text(text: str):
    semanticSearch = SemanticSearch()
    print(f"Text: {text}")
    embedding = semanticSearch.generate_embedding(text)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    with open(constants.MOVIES_PATH, "r") as f:
        documents = json.loads(f.read()).get('movies', [])
    
    chunkedSemanticSearch = ChunkedSemanticSearch()
    chunkedSemanticSearch.load_or_create_chunk_embeddings(documents=documents)

    chunkedSemanticSearch.search_chunks("superhero action movie", limit=5)
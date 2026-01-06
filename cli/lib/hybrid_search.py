import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        normalized_bm25_scores = normalize_values([score for _, score in bm25_results])
        normalized_bm25_scores_map = {
            doc_id: score for (doc_id, _), score in zip(bm25_results, normalized_bm25_scores)
        }

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        normalized_semantic_scores = normalize_values([res['score'] for res in semantic_results])
        normalized_semantic_scores_map = {
            item["id"]: score for item, score in zip(semantic_results, normalized_semantic_scores)
        }

        all_scores = {
            doc_id: {""
            "hybrid_score": hybrid_score(
                normalized_bm25_scores_map.get(doc_id, 0.0),
                normalized_semantic_scores_map.get(doc_id, 0.0),
                alpha
            ),
            "bm25_score": normalized_bm25_scores_map.get(doc_id, 0.0),
            "semantic_score": normalized_semantic_scores_map.get(doc_id, 0.0)
            }
            for doc_id in set(normalized_bm25_scores_map.keys()) | set(normalized_semantic_scores_map.keys())
        }
        return sorted(all_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)[:limit]




    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

def normalize_values(values: list[float]) -> list[float]:
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val == 0:
        return [1.0 for _ in values]
    return [(val - min_val) / (max_val - min_val) for val in values]


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score
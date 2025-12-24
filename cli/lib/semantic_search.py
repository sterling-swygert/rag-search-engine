
from sentence_transformers import SentenceTransformer


class SemanticSearch(object):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


def verify_model():
    semanticSearch = SemanticSearch()
    print(f"Model loaded: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")

from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List


class SemanticSearch(object):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text) -> List[Tensor]:
        if text.strip() == "":
            raise ValueError("Input text cannot be empty")
        return self.model.encode([text])[0]


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
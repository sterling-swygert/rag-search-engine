
from collections import Counter
import json
from nltk.stem import PorterStemmer
import math
import os
from pathlib import Path
import pickle as pkl
import string

from . import constants
from . import utils

stemmer = PorterStemmer()



def clean_text(text: str) -> str:
    return text.translate(str.maketrans({char: "" for char in string.punctuation})).lower()

def tokenize(text: str) -> list[str]:
    stopwords = utils.load_stopwords()
    return [stemmer.stem(s) for s in clean_text(text).split() if (s != '' and s not in stopwords)]



class TokenException(Exception):
    pass


class InvertedIndex(object):
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequences = {}
        self.doc_lengths = {}
        self.index_path = os.path.join(constants.CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(constants.CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(constants.CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(constants.CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.term_frequences[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)
            if token not in self.term_frequences[doc_id]:
                self.term_frequences[doc_id][token] = 0
            self.term_frequences[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths) if self.doc_lengths else 0.0
    
    def get_documents(self, term):
        return sorted(self.index.get(clean_text(term), []))

    def build(self):
        with open(os.path.join(constants.DATA_DIR, "movies.json"), "r") as f:
            movies_data = json.loads(f.read()).get('movies', [])
            for movie in movies_data:
                doc_id = movie.get('id')
                self.docmap[doc_id] = movie
                self.__add_document(doc_id, f"{movie.get('title', '')} {movie.get('description', '')}")

    def save(self) -> None:
        Path(constants.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        pkl.dump(self.index, open(self.index_path, "wb"))
        pkl.dump(self.docmap, open(self.docmap_path, "wb"))
        pkl.dump(self.term_frequences, open(self.term_frequencies_path, "wb"))
        pkl.dump(self.doc_lengths, open(self.doc_lengths_path, "wb"))

    def load(self) -> None:
        for filename in ["index", "docmap", "term_frequencies", "doc_lengths"]:
            if not Path(os.path.join(constants.CACHE_DIR, f"{filename}.pkl")).exists():
                raise FileNotFoundError(f"Cached {filename} file not found. Please build the index first.")
        
        self.index = pkl.load(open(self.index_path, "rb"))
        self.docmap = pkl.load(open(self.docmap_path, "rb"))
        self.term_frequences = pkl.load(open(self.term_frequencies_path, "rb"))
        self.doc_lengths = pkl.load(open(self.doc_lengths_path, "rb"))

    def get_tf(self, doc_id, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise TokenException("Only single term is allowed for term frequency calculation.") 
        return self.term_frequences.get(doc_id, {}).get(tokens[0], 0)
    
    def get_tf_idf(self, doc_id, term: str) -> float:
        token = tokenize(term)[0]
        tf = self.get_tf(doc_id, token)
        df = len(self.get_documents(token))
        N = len(self.docmap)
        idf = math.log((N + 1) / (df + 1))
        tf_idf = tf * idf
        return tf_idf
    
    def get_bm25_tf(self, doc_id, term, k1=constants.BM25_K1, b=constants.BM25_B) -> float:
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise TokenException("Only single term is allowed for BM25 IDF calculation.")
        df = len(self.get_documents(tokens[0]))
        N = len(self.docmap)
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf
    
    def bm25(self, doc_id, term) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        scores = {}
        for doc_id in self.docmap.keys():
            scores[doc_id] = 0.0
            for token in tokens:
                scores[doc_id] += self.bm25(doc_id, token)

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:limit]
    

if __name__ == "__main__":
    with open("data/movies.json", "r") as f:
        print("opening data/movies.json")

from collections import Counter
import json
from nltk.stem import PorterStemmer
from pathlib import Path
import pickle as pkl
import string

stemmer = PorterStemmer()

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    return text.translate(str.maketrans({char: "" for char in string.punctuation})).lower()

def tokenize(text: str) -> list[str]:
    return [stemmer.stem(s) for s in clean_text(text).split() if s != '']

with open("data/stopwords.txt", "r") as f:
    stopwords = [stemmer.stem(clean_text(s)) for s in f.read().splitlines()]


class InvertedIndex(object):
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequences = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.term_frequences[doc_id] = Counter()
        for token in tokens:
            if token in stopwords:
                continue
            if token not in self.index:
                self.index[token] = []
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)
            if token not in self.term_frequences[doc_id]:
                self.term_frequences[doc_id][token] = 0
            self.term_frequences[doc_id][token] += 1
    
    def get_documents(self, term):
        return sorted(self.index.get(clean_text(term), []))

    def build(self):
        with open("data/movies.json", "r") as f:
            movies_data = json.loads(f.read()).get('movies', [])
            for movie in movies_data:
                doc_id = movie.get('id')
                title = movie.get('title', '')
                self.docmap[doc_id] = movie
                self.__add_document(doc_id, f"{movie.get('title', '')} {movie.get('description', '')}")

    def save(self) -> None:
        Path("cache").mkdir(parents=True, exist_ok=True)
        pkl.dump(self.index, open("cache/index.pkl", "wb"))
        pkl.dump(self.docmap, open("cache/docmap.pkl", "wb"))
        pkl.dump(self.term_frequences, open("cache/term_frequencies.pkl", "wb"))

    def load(self) -> None:
        for filename in ["index", "docmap", "term_frequencies"]:
            if not Path(f"cache/{filename}.pkl").exists():
                raise FileNotFoundError(f"Cached {filename} file not found. Please build the index first.")
        self.index = pkl.load(open("cache/index.pkl", "rb"))
        self.docmap = pkl.load(open("cache/docmap.pkl", "rb"))
        self.term_frequences = pkl.load(open("cache/term_frequencies.pkl", "rb"))

    def get_tf(self, doc_id, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise 
        return self.term_frequences.get(doc_id, {}).get(tokens[0], 0)
    

if __name__ == "__main__":
    with open("data/movies.json", "r") as f:
        print("opening data/movies.json")
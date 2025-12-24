import os

BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_SEARCH_LIMIT = 5

CACHE_DIR = "cache"
DATA_DIR = "data"
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.json")
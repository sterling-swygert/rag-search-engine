import json

from lib import constants


def load_movies() -> list[dict]:
    with open(constants.MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(constants.STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
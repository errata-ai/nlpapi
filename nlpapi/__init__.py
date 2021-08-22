import pysbd
import spacy

from fastapi import FastAPI, HTTPException
from pysbd.languages import LANGUAGE_CODES

app = FastAPI()
nlp = {
    "en": spacy.load("en_core_web_sm"),
    "de": spacy.load("de_core_news_sm"),
    "es": spacy.load("es_core_news_sm"),
}


class NoLangError(HTTPException):
    """Raised when an unknown (i.e., unsupported) language code is given."""

    # NOTE:  A two-character (ISO 639-1) code is expected.
    reason = "The language code '{0}' is not recognized."

    def __init__(self, lang: str, status_code: int = 404):
        super().__init__(status_code, self.reason.format(lang))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/tag")
def tag(lang: str, text: str):
    model = nlp.get(lang)
    if not model:
        raise NoLangError(lang)

    tokens = []
    for token in model(text, disable=["ner", "parser", "lemmatizer", "textcat"]):
        tokens.append(
            {
                "Text": token.text,
                "Tag": token.tag_ if lang in ["en", "de"] else token.pos_,
            }
        )

    return {"tokens": tokens}


@app.post("/segment")
def segment(lang: str, text: str):
    if lang not in LANGUAGE_CODES:
        raise NoLangError(lang)

    seg = pysbd.Segmenter(language=lang, clean=False)
    return {"sents": seg.segment(text)}

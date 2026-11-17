"""Análisis lingüístico en español con spaCy.

Un único análisis del texto produce un TextAnalysis inmutable con oraciones,
tokens, lemas y palabras vacías. Está pensado para compartirse entre
Summarizer, KeywordExtractor y KeyPoints, de modo que el texto se procese
una sola vez.

El analizador trabaja sobre el texto que recibe (normalmente la salida de
TextNormalizer y EmojiProcessor); no conoce esos módulos ni depende de ellos.
Todas las posiciones (start, end) son índices de carácter en ese texto.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc, Token

SPACY_MODEL = "es_core_news_sm"
# ner no se usa en ninguna fase y es la parte más lenta del pipeline
EXCLUDED_COMPONENTS = ("ner",)
BOUNDARY_COMPONENT = "brieflyx_sentence_starts"
MAX_TEXT_LENGTH = 1_000_000
MIN_CONTENT_LENGTH = 2

# spaCy las marca como vacías; se señalan aparte para no perder la negación.
# "sin" queda fuera: es preposición y aparece en "sin embargo"
NEGATION_WORDS = frozenset({
    "no", "nunca", "jamás", "ni", "tampoco", "nadie", "nada",
    "ninguno", "ninguna", "ningún",
})
# ruido de chats y tratamientos de cortesía que no aportan contenido
EXTRA_STOP_WORDS = frozenset({"jaja", "sr", "sra", "srta", "dr", "dra", "lic", "ing"})
ELLIPSES = frozenset({"...", "…"})
OPENING_MARKS = frozenset({"¿", "¡"})
SENTENCE_TERMINATORS = frozenset({".", "!", "?"}) | ELLIPSES


class LanguageModelNotFoundError(RuntimeError):
    """El modelo de spaCy para español no está instalado."""


@dataclass(frozen=True)
class AnalyzedToken:
    """Un token con su información lingüística."""

    text: str
    lemma: str
    pos: str
    start: int
    end: int
    is_stop: bool
    is_punct: bool
    is_negation: bool

    @property
    def is_word(self) -> bool:
        """Palabra con al menos una letra (excluye signos y números puros)."""
        return any(char.isalpha() for char in self.text)

    @property
    def is_content(self) -> bool:
        """Palabra con significado propio: no vacía, no signo, lema útil."""
        return self.is_word and not self.is_stop and len(self.lemma) >= MIN_CONTENT_LENGTH


@dataclass(frozen=True)
class AnalyzedSentence:
    """Una oración del texto con sus tokens."""

    index: int
    text: str
    start: int
    end: int
    tokens: tuple[AnalyzedToken, ...]

    @property
    def words(self) -> list[str]:
        return [token.text for token in self.tokens if token.is_word]

    @property
    def content_lemmas(self) -> list[str]:
        """Lemas de las palabras con contenido, en orden de aparición."""
        return [token.lemma for token in self.tokens if token.is_content]

    @property
    def lemmatized_text(self) -> str:
        """Representación normalizada para TF-IDF y similitud entre oraciones."""
        return " ".join(self.content_lemmas)

    @property
    def has_negation(self) -> bool:
        return any(token.is_negation for token in self.tokens)


@dataclass(frozen=True)
class TextAnalysis:
    """Resultado inmutable del análisis lingüístico de un texto."""

    text: str
    sentences: tuple[AnalyzedSentence, ...]

    @property
    def is_empty(self) -> bool:
        return not self.sentences

    @property
    def tokens(self) -> list[AnalyzedToken]:
        return [token for sentence in self.sentences for token in sentence.tokens]

    @property
    def word_count(self) -> int:
        return sum(len(sentence.words) for sentence in self.sentences)

    @property
    def sentence_texts(self) -> list[str]:
        return [sentence.text for sentence in self.sentences]

    @property
    def content_lemmas(self) -> list[str]:
        return [lemma for sentence in self.sentences for lemma in sentence.content_lemmas]

    @property
    def lemma_frequencies(self) -> dict[str, int]:
        """Frecuencia de cada lema con contenido, de mayor a menor."""
        return dict(Counter(self.content_lemmas).most_common())


@Language.component(BOUNDARY_COMPONENT)
def fix_sentence_starts(doc: Doc) -> Doc:
    """Ajusta los inicios de oración que el sentencizer no resuelve en español.

    - Tras un salto de línea empieza una oración nueva: en chats cada línea
      suele ser un mensaje sin punto final.
    - "¿" y "¡" abren la oración siguiente; el sentencizer los dejaría
      pegados al final de la anterior ("¿Qué haces? ¡No...").
    """
    for token in doc[:-1]:
        following = doc[token.i + 1]
        if token.is_space and "\n" in token.text:
            following.is_sent_start = True
        elif token.text in OPENING_MARKS and following.is_sent_start and not token.is_sent_start:
            token.is_sent_start = True
            following.is_sent_start = False
    return doc


@lru_cache(maxsize=2)
def load_language_model(model_name: str = SPACY_MODEL) -> Language:
    """Carga el modelo de spaCy una sola vez por proceso.

    Raises:
        LanguageModelNotFoundError: si el modelo no está instalado.
    """
    try:
        nlp = spacy.load(model_name, exclude=list(EXCLUDED_COMPONENTS))
    except OSError as error:
        raise LanguageModelNotFoundError(
            f"no se encontró el modelo '{model_name}'. "
            f"instálalo con: python -m spacy download {model_name}"
        ) from error
    # el parser segmenta de forma irregular; con todos los límites fijados por
    # reglas antes de él, ya no agrega cortes propios
    nlp.add_pipe(
        "sentencizer",
        before="parser",
        config={"punct_chars": sorted(set(Sentencizer.default_punct_chars) | ELLIPSES)},
    )
    nlp.add_pipe(BOUNDARY_COMPONENT, after="sentencizer")
    nlp.max_length = MAX_TEXT_LENGTH
    return nlp


class TextAnalyzer:
    """Tokeniza, segmenta y lematiza textos en español."""

    def __init__(self, model_name: str = SPACY_MODEL) -> None:
        self._nlp = load_language_model(model_name)
        self._stop_words = self._nlp.Defaults.stop_words | EXTRA_STOP_WORDS

    def analyze(self, text: str) -> TextAnalysis:
        """Analiza ``text`` y devuelve sus oraciones, tokens y lemas.

        Raises:
            TypeError: si ``text`` no es una cadena.
            ValueError: si el texto supera MAX_TEXT_LENGTH caracteres.
        """
        if not isinstance(text, str):
            raise TypeError("el texto a analizar debe ser una cadena")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"el texto supera el máximo de {MAX_TEXT_LENGTH:,} caracteres"
            )
        if not text.strip():
            return TextAnalysis(text, ())

        doc = self._nlp(text)
        sentences = []
        for start, end in self._sentence_bounds(doc):
            sentence = self._build_sentence(doc, start, end, len(sentences))
            if sentence is not None:
                sentences.append(sentence)
        return TextAnalysis(text, tuple(sentences))

    def _sentence_bounds(self, doc: Doc) -> list[tuple[int, int]]:
        """Límites de oración (en tokens), corrigiendo cortes falsos.

        Si la oración siguiente empieza en minúscula y la anterior acaba en una
        abreviatura ("a las 3 p.m. y habló") o en puntos suspensivos
        ("y bueno... no sé"), se unen. Nunca se une a través de un salto de línea.
        """
        bounds: list[tuple[int, int]] = []
        for span in doc.sents:
            if bounds and self._continues_previous(doc, bounds[-1], span.start):
                bounds[-1] = (bounds[-1][0], span.end)
            else:
                bounds.append((span.start, span.end))
        return bounds

    @staticmethod
    def _continues_previous(doc: Doc, previous: tuple[int, int], start: int) -> bool:
        last = next(
            (doc[i] for i in range(previous[1] - 1, previous[0] - 1, -1) if not doc[i].is_space),
            None,
        )
        if last is None or (doc[start - 1].is_space and "\n" in doc[start - 1].text):
            return False
        first = next((t for t in doc[start:] if not t.is_space), None)
        if first is None or not first.text[:1].islower():
            return False
        # "y bueno... no sé" sigue siendo la misma oración
        return last.text in ELLIPSES or _is_abbreviation_end(doc, last)

    def _build_sentence(
        self, doc: Doc, start: int, end: int, index: int
    ) -> AnalyzedSentence | None:
        """Construye una oración sin espacios en los bordes; None si queda vacía."""
        tokens = [token for token in doc[start:end] if not token.is_space]
        if not tokens:
            return None
        char_start = tokens[0].idx
        char_end = tokens[-1].idx + len(tokens[-1].text)
        return AnalyzedSentence(
            index=index,
            text=doc.text[char_start:char_end],
            start=char_start,
            end=char_end,
            tokens=tuple(self._build_token(token) for token in tokens),
        )

    def _build_token(self, token: Token) -> AnalyzedToken:
        lower = token.lower_
        # "dr." → "dr": el punto de la abreviatura no forma parte del lema
        lemma = token.lemma_.lower().rstrip(".") or lower
        return AnalyzedToken(
            text=token.text,
            lemma=lemma,
            pos=token.pos_,
            start=token.idx,
            end=token.idx + len(token.text),
            is_stop=lower in self._stop_words or lemma in self._stop_words,
            is_punct=token.is_punct,
            is_negation=lower in NEGATION_WORDS,
        )


def _is_abbreviation_end(doc: Doc, token: Token) -> bool:
    """Indica si ``token`` cierra una abreviatura y no una oración.

    Cubre "Sra." (un solo token) y "p.m." (spaCy lo parte en "p.m" + ".").
    """
    if token.text in SENTENCE_TERMINATORS:
        if token.text != "." or token.i == 0:
            return False
        previous = doc[token.i - 1]
        return not previous.whitespace_ and "." in previous.text
    return token.text.endswith(".")

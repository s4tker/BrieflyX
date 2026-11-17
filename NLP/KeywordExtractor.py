"""Extracción de palabras clave con TF-IDF sobre lemas.

Flujo:

1. TextAnalyzer aporta oraciones, lemas, categorías gramaticales y palabras
   vacías (no se vuelve a tokenizar ni lematizar).
2. Candidatos: sustantivos, nombres propios y adjetivos, más frases de dos
   palabras contiguas ("cambio climático") que se repiten en el texto.
3. Cada oración es un documento TF-IDF. La relevancia de un término es la
   suma de sus pesos en todas las oraciones (frecuencia × idf): pesa que el
   término se repita, pero el idf frena a los que aparecen en casi todas
   las oraciones sin distinguir ninguna.
4. Las variantes de un mismo término ("economía", "económico") se agrupan
   por su raíz (stemmer Snowball de NLTK) y se muestran una sola vez.
5. Si una frase contiene una palabra clave ya elegida, la frase la sustituye.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from NLP.TextAnalyzer import AnalyzedSentence, AnalyzedToken, TextAnalysis, TextAnalyzer

DEFAULT_TOP_N = 10
KEYWORD_POS = frozenset({"NOUN", "PROPN", "ADJ"})
PHRASE_POS_PATTERNS = frozenset({
    ("NOUN", "ADJ"),
    ("PROPN", "ADJ"),
    ("NOUN", "NOUN"),
    ("PROPN", "PROPN"),
})
# una frase debe repetirse para considerarse un concepto y no una casualidad
MIN_PHRASE_FREQUENCY = 2
# tildes y diéresis no distinguen variantes; la ñ sí se conserva
ACCENT_MARKS = frozenset({"́", "̈"})
STEMMER_LANGUAGE = "spanish"


@dataclass(frozen=True)
class Keyword:
    """Una palabra clave con su relevancia.

    ``score`` está normalizado: la palabra más relevante vale 1.0.
    ``lemmas`` contiene los términos agrupados (lemas o frases lematizadas).
    """

    term: str
    score: float
    frequency: int
    lemmas: tuple[str, ...]
    variants: tuple[str, ...]

    @property
    def is_phrase(self) -> bool:
        return " " in self.term


@dataclass(frozen=True)
class KeywordResult:
    """Palabras clave ordenadas de mayor a menor relevancia."""

    keywords: tuple[Keyword, ...]

    @property
    def is_empty(self) -> bool:
        return not self.keywords

    @property
    def terms(self) -> list[str]:
        return [keyword.term for keyword in self.keywords]


@dataclass
class _Group:
    """Acumulador de un término y sus variantes durante la extracción."""

    key: str
    score: float = 0.0
    frequency: int = 0
    first_position: int = 0
    lemma_counts: Counter = field(default_factory=Counter)
    surface_counts: Counter = field(default_factory=Counter)
    proper_noun_count: int = 0

    @property
    def is_phrase(self) -> bool:
        return " " in self.key

    @property
    def display(self) -> str:
        """Lema más frecuente; los nombres propios conservan su escritura."""
        surface = self.surface_counts.most_common(1)[0][0]
        if self.proper_noun_count * 2 > self.frequency and surface[:1].isupper():
            return surface
        return self.lemma_counts.most_common(1)[0][0]


class KeywordExtractor:
    """Identifica los términos más relevantes de un texto."""

    def __init__(self, analyzer: TextAnalyzer | None = None) -> None:
        self._analyzer = analyzer

    def extract(self, text: str, top_n: int = DEFAULT_TOP_N) -> KeywordResult:
        """Analiza ``text`` y devuelve sus ``top_n`` palabras clave.

        Raises:
            TypeError: si ``text`` no es una cadena.
            ValueError: si ``top_n`` no es un entero positivo.
        """
        _validate_top_n(top_n)
        if self._analyzer is None:
            self._analyzer = TextAnalyzer()
        return self.extract_from_analysis(self._analyzer.analyze(text), top_n)

    def extract_from_analysis(
        self, analysis: TextAnalysis, top_n: int = DEFAULT_TOP_N
    ) -> KeywordResult:
        """Extrae palabras clave de un texto ya analizado."""
        _validate_top_n(top_n)
        documents = [sentence_terms(sentence) for sentence in analysis.sentences]
        if not any(documents):
            # sin sustantivos ni adjetivos: se aceptan los demás lemas con contenido
            documents = [
                [token.lemma for token in sentence.tokens if token.is_content]
                for sentence in analysis.sentences
            ]
        if not any(documents):
            return KeywordResult(())

        term_scores = _tfidf_scores(documents)
        groups = self._group_terms(analysis.sentences, documents, term_scores)
        return KeywordResult(self._select(groups, top_n))

    @staticmethod
    def _group_terms(
        sentences: tuple[AnalyzedSentence, ...],
        documents: list[list[str]],
        term_scores: dict[str, float],
    ) -> dict[str, _Group]:
        """Agrupa los términos por raíz y acumula puntuación y frecuencia."""
        frequencies = Counter(term for document in documents for term in document)
        groups: dict[str, _Group] = {}
        position = 0
        for sentence, document in zip(sentences, documents):
            surfaces = _surface_forms(sentence)
            for term in document:
                position += 1
                if term not in term_scores:
                    continue
                if " " in term and frequencies[term] < MIN_PHRASE_FREQUENCY:
                    continue
                key = variant_key(term)
                group = groups.get(key)
                if group is None:
                    group = groups[key] = _Group(key, first_position=position)
                if group.lemma_counts[term] == 0:
                    group.score += term_scores[term]
                group.lemma_counts[term] += 1
                group.frequency += 1
                surface, pos = surfaces.get(term, (term, ""))
                group.surface_counts[surface] += 1
                group.proper_noun_count += pos == "PROPN"
        return groups

    @staticmethod
    def _select(groups: dict[str, _Group], top_n: int) -> tuple[Keyword, ...]:
        """Elige las ``top_n`` palabras clave; una frase sustituye a sus palabras."""
        ranked = sorted(
            groups.values(),
            key=lambda g: (-round(g.score, 9), -g.frequency, g.first_position),
        )
        chosen: list[tuple[_Group, float]] = []
        for group in ranked:
            stems = set(group.key.split())
            if group.is_phrase:
                covered = [
                    i for i, (other, _) in enumerate(chosen)
                    if not other.is_phrase and other.key in stems
                ]
                if covered:
                    # la frase hereda el lugar y la puntuación del término que precisa
                    inherited = max(group.score, chosen[covered[0]][1])
                    chosen[covered[0]] = (group, inherited)
                    for i in reversed(covered[1:]):
                        del chosen[i]
                    continue
            elif any(other.is_phrase and group.key in other.key.split() for other, _ in chosen):
                continue
            if len(chosen) < top_n:
                chosen.append((group, group.score))

        chosen.sort(key=lambda item: -item[1])
        top_score = chosen[0][1] if chosen and chosen[0][1] > 0 else 1.0
        return tuple(
            Keyword(
                term=group.display,
                score=round(score / top_score, 4),
                frequency=group.frequency,
                lemmas=tuple(sorted(group.lemma_counts)),
                variants=tuple(sorted(group.surface_counts)),
            )
            for group, score in chosen[:top_n]
        )


def sentence_terms(sentence: AnalyzedSentence) -> list[str]:
    """Términos candidatos de una oración: lemas relevantes y frases de dos palabras.

    Compartida con KeyPoints para comprobar qué palabras clave aparecen en
    cada oración con el mismo criterio que se usó para extraerlas.
    """
    tokens = sentence.tokens
    terms = [token.lemma for token in tokens if _is_candidate(token)]
    for first, second in zip(tokens, tokens[1:]):
        if (
            first.is_content
            and second.is_content
            and (first.pos, second.pos) in PHRASE_POS_PATTERNS
        ):
            terms.append(f"{first.lemma} {second.lemma}")
    return terms


def variant_key(term: str) -> str:
    """Clave que agrupa variantes: raíz de cada palabra, sin tildes.

    Se aplica el stemmer antes de quitar tildes porque Snowball reconoce los
    sufijos acentuados ("economía" y "económico" → "econom").
    """
    return " ".join(_strip_accents(_stem(word)) for word in term.split())


def _is_candidate(token: AnalyzedToken) -> bool:
    return token.is_content and token.pos in KEYWORD_POS


def _surface_forms(sentence: AnalyzedSentence) -> dict[str, tuple[str, str]]:
    """Mapa término → (forma escrita, categoría) para palabras y frases de la oración.

    Una frase cuenta como nombre propio solo si sus dos palabras lo son.
    """
    tokens = sentence.tokens
    forms = {token.lemma: (token.text, token.pos) for token in tokens}
    for first, second in zip(tokens, tokens[1:]):
        pos = "PROPN" if first.pos == second.pos == "PROPN" else ""
        forms[f"{first.lemma} {second.lemma}"] = (f"{first.text} {second.text}", pos)
    return forms


def _tfidf_scores(documents: list[list[str]]) -> dict[str, float]:
    """Suma de los pesos TF-IDF de cada término en todas las oraciones.

    Sin normalizar por oración: así una palabra aislada en una oración corta
    no se infla, y la puntuación equivale a frecuencia × especificidad (idf).
    """
    vectorizer = TfidfVectorizer(analyzer=_identity, norm=None)
    matrix = vectorizer.fit_transform(documents)
    totals = matrix.sum(axis=0).A1
    return dict(zip(vectorizer.get_feature_names_out(), (float(value) for value in totals)))


def _identity(document: list[str]) -> list[str]:
    return document


def _strip_accents(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word)
    kept = "".join(char for char in decomposed if char not in ACCENT_MARKS)
    return unicodedata.normalize("NFC", kept)


@lru_cache(maxsize=4096)
def _stem(word: str) -> str:
    return _stemmer().stem(word)


@lru_cache(maxsize=1)
def _stemmer() -> SnowballStemmer:
    return SnowballStemmer(STEMMER_LANGUAGE)


def _validate_top_n(top_n: int) -> None:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n debe ser un entero positivo")

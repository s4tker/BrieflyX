"""Identificación de ideas principales.

Una idea principal es una oración central (TextRank) que además concentra
los conceptos más relevantes (palabras clave). La puntuación combina ambas
señales normalizadas:

    puntuación = TEXTRANK_WEIGHT · textrank + KEYWORD_WEIGHT · cobertura

donde la cobertura es la suma de las puntuaciones de las palabras clave que
aparecen en la oración. Así se distingue del resumen, que solo usa TextRank
y una proporción del texto: las ideas principales son pocas, sustanciales y
llevan asociados los conceptos que contienen.

Se reutilizan la matriz de similitud, PageRank y la selección sin
redundancia de Summarizer, y los términos candidatos de KeywordExtractor;
no se reimplementa ninguno de esos algoritmos.
"""

from __future__ import annotations

from dataclasses import dataclass

from NLP.KeywordExtractor import KeywordExtractor, KeywordResult, sentence_terms
from NLP.Summarizer import (
    ScoredSentence,
    select_diverse_sentences,
    sentence_similarity_matrix,
    textrank_scores,
)
from NLP.TextAnalyzer import AnalyzedSentence, TextAnalysis, TextAnalyzer

DEFAULT_KEY_POINT_COUNT = 5
TEXTRANK_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4
# una idea principal debe tener cuerpo; las oraciones más cortas solo
# entran si no hay otras disponibles
MIN_KEY_POINT_WORDS = 4
# más estricto que en el resumen: dos ideas que comparten su concepto
# principal ya cuentan como la misma idea
KEY_POINT_REDUNDANCY_THRESHOLD = 0.25
# palabras clave que se consideran para la cobertura de cada oración
KEYWORDS_FOR_COVERAGE = 15


@dataclass(frozen=True)
class KeyPoint:
    """Una idea principal: oración original, puntuación y conceptos que contiene."""

    index: int
    text: str
    start: int
    end: int
    score: float
    textrank_score: float
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class KeyPointsResult:
    """Ideas principales en el orden en que aparecen en el texto."""

    source_text: str
    points: tuple[KeyPoint, ...]

    @property
    def is_empty(self) -> bool:
        return not self.points

    @property
    def texts(self) -> list[str]:
        return [point.text for point in self.points]


class KeyPointIdentifier:
    """Detecta las oraciones que expresan las ideas centrales de un texto."""

    def __init__(
        self,
        analyzer: TextAnalyzer | None = None,
        keyword_extractor: KeywordExtractor | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._keyword_extractor = keyword_extractor or KeywordExtractor(analyzer)

    def identify(self, text: str, count: int = DEFAULT_KEY_POINT_COUNT) -> KeyPointsResult:
        """Analiza ``text`` y devuelve hasta ``count`` ideas principales.

        Raises:
            TypeError: si ``text`` no es una cadena.
            ValueError: si ``count`` no es un entero positivo.
        """
        _validate_count(count)
        if self._analyzer is None:
            self._analyzer = TextAnalyzer()
        return self.identify_from_analysis(self._analyzer.analyze(text), count)

    def identify_from_analysis(
        self,
        analysis: TextAnalysis,
        count: int = DEFAULT_KEY_POINT_COUNT,
        keywords: KeywordResult | None = None,
        ranking: tuple[ScoredSentence, ...] | None = None,
    ) -> KeyPointsResult:
        """Identifica ideas principales de un texto ya analizado.

        ``keywords`` y ``ranking`` (de SummaryResult) son opcionales: si se
        pasan, se reutilizan en lugar de calcularse de nuevo.
        """
        _validate_count(count)
        sentences = analysis.sentences
        if not sentences:
            return KeyPointsResult(analysis.text, ())

        if keywords is None:
            keywords = self._keyword_extractor.extract_from_analysis(
                analysis, top_n=KEYWORDS_FOR_COVERAGE
            )
        similarity = sentence_similarity_matrix(sentences)
        if ranking is not None and len(ranking) == len(sentences):
            centrality = [item.score for item in ranking]
        else:
            centrality = textrank_scores(similarity)

        matched = [_matched_keywords(sentence, keywords) for sentence in sentences]
        coverage = [sum(score for _, score in found) for found in matched]
        scores = _combine(centrality, coverage)

        eligible = [len(s.words) >= MIN_KEY_POINT_WORDS for s in sentences]
        if any(eligible):
            # las oraciones demasiado cortas quedan al final del orden
            selection_scores = [
                score if ok else score - 2.0 for score, ok in zip(scores, eligible)
            ]
            target = min(count, sum(eligible))
        else:
            selection_scores = scores
            target = min(count, len(sentences))

        chosen = select_diverse_sentences(
            sentences, selection_scores, similarity, target, KEY_POINT_REDUNDANCY_THRESHOLD
        )
        points = tuple(
            KeyPoint(
                index=sentences[i].index,
                text=sentences[i].text,
                start=sentences[i].start,
                end=sentences[i].end,
                score=round(scores[i], 4),
                textrank_score=centrality[i],
                keywords=tuple(term for term, _ in matched[i]),
            )
            for i in sorted(chosen)
        )
        return KeyPointsResult(analysis.text, points)


def _matched_keywords(
    sentence: AnalyzedSentence, keywords: KeywordResult
) -> list[tuple[str, float]]:
    """Palabras clave presentes en la oración, con su puntuación."""
    terms = set(sentence_terms(sentence)) | set(sentence.content_lemmas)
    return [
        (keyword.term, keyword.score)
        for keyword in keywords.keywords
        if terms.intersection(keyword.lemmas)
    ]


def _combine(centrality: list[float], coverage: list[float]) -> list[float]:
    """Mezcla centralidad y cobertura, cada una normalizada a [0, 1]."""
    return [
        TEXTRANK_WEIGHT * c + KEYWORD_WEIGHT * k
        for c, k in zip(_normalize(centrality), _normalize(coverage))
    ]


def _normalize(values: list[float]) -> list[float]:
    """Divide por el máximo; si todo es cero, devuelve ceros (sin dividir entre cero)."""
    top = max(values, default=0.0)
    if top <= 0:
        return [0.0] * len(values)
    return [value / top for value in values]


def _validate_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count debe ser un entero positivo")

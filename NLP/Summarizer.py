"""Resumen extractivo con TextRank.

Flujo:

1. TextAnalyzer segmenta el texto y lematiza cada oración (sin palabras vacías).
2. Cada oración se representa con un vector TF-IDF de sus lemas.
3. La similitud coseno entre vectores define las aristas de un grafo.
4. PageRank (NetworkX) puntúa la importancia de cada oración.
5. Se eligen las mejor puntuadas, descartando las redundantes, y se
   devuelven en su orden original.

El resumen es extractivo: cada oración elegida es un fragmento literal del
texto analizado, recuperado por sus posiciones, sin reescritura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from NLP.TextAnalyzer import AnalyzedSentence, TextAnalysis, TextAnalyzer

DEFAULT_RATIO = 0.3
PAGERANK_DAMPING = 0.85
PAGERANK_MAX_ITERATIONS = 200
PAGERANK_TOLERANCE = 1.0e-6
# por encima de este coseno dos oraciones dicen prácticamente lo mismo
REDUNDANCY_THRESHOLD = 0.8
# diferencias menores se consideran empate entre puntuaciones
SCORE_PRECISION = 9


@dataclass(frozen=True)
class ScoredSentence:
    """Una oración del texto con su puntuación TextRank."""

    index: int
    text: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class SummaryResult:
    """Resultado inmutable del resumen.

    ``sentences`` son las oraciones elegidas en orden original;
    ``ranking`` contiene todas las oraciones con su puntuación, en orden original.
    """

    source_text: str
    sentences: tuple[ScoredSentence, ...]
    ranking: tuple[ScoredSentence, ...]
    original_word_count: int
    summary_word_count: int

    @property
    def is_empty(self) -> bool:
        return not self.sentences

    @property
    def summary_text(self) -> str:
        return " ".join(sentence.text for sentence in self.sentences)

    @property
    def original_sentence_count(self) -> int:
        return len(self.ranking)

    @property
    def reduction_percentage(self) -> float:
        """Porcentaje de palabras eliminadas respecto al texto original."""
        if self.original_word_count == 0:
            return 0.0
        kept = self.summary_word_count / self.original_word_count
        return round((1 - kept) * 100, 1)


class Summarizer:
    """Genera resúmenes extractivos con TextRank."""

    def __init__(self, analyzer: TextAnalyzer | None = None) -> None:
        self._analyzer = analyzer

    def summarize(
        self,
        text: str,
        sentence_count: int | None = None,
        ratio: float | None = None,
    ) -> SummaryResult:
        """Analiza ``text`` y lo resume.

        Args:
            sentence_count: cantidad exacta de oraciones del resumen.
            ratio: proporción de oraciones a conservar (0.3 = 30 %).
                Si no se indica ninguno de los dos, se usa DEFAULT_RATIO.

        Raises:
            TypeError: si ``text`` no es una cadena.
            ValueError: si los parámetros de longitud no son válidos.
        """
        _validate_length(sentence_count, ratio)
        if self._analyzer is None:
            self._analyzer = TextAnalyzer()
        return self.summarize_analysis(self._analyzer.analyze(text), sentence_count, ratio)

    def summarize_analysis(
        self,
        analysis: TextAnalysis,
        sentence_count: int | None = None,
        ratio: float | None = None,
    ) -> SummaryResult:
        """Resume un texto ya analizado, sin volver a procesarlo."""
        _validate_length(sentence_count, ratio)
        sentences = analysis.sentences
        word_count = analysis.word_count
        if not sentences:
            return SummaryResult(analysis.text, (), (), word_count, 0)

        # una sola matriz para puntuar y para filtrar redundancia
        similarity = _similarity_matrix(sentences)
        scores = _score(similarity)
        ranking = tuple(
            ScoredSentence(s.index, s.text, s.start, s.end, score)
            for s, score in zip(sentences, scores)
        )
        target = _target_size(len(sentences), sentence_count, ratio)
        chosen = self._select(sentences, scores, similarity, target)

        selected = tuple(ranking[index] for index in sorted(chosen))
        summary_words = sum(len(sentences[index].words) for index in chosen)
        return SummaryResult(analysis.text, selected, ranking, word_count, summary_words)

    @staticmethod
    def _select(
        sentences: tuple[AnalyzedSentence, ...],
        scores: list[float],
        similarity: np.ndarray,
        target: int,
    ) -> list[int]:
        """Elige hasta ``target`` oraciones por puntuación, evitando redundancia.

        Los empates se resuelven a favor de la oración con más contenido y,
        después, de la que aparece antes.
        """
        order = sorted(
            range(len(sentences)),
            key=lambda i: (
                -round(scores[i], SCORE_PRECISION),
                -len(sentences[i].content_lemmas),
                i,
            ),
        )
        chosen: list[int] = []
        seen_texts: set[str] = set()
        for index in order:
            if len(chosen) == target:
                break
            key = " ".join(sentences[index].words).casefold()
            if key in seen_texts:
                continue
            if any(similarity[index][other] >= REDUNDANCY_THRESHOLD for other in chosen):
                continue
            chosen.append(index)
            seen_texts.add(key)
        # nunca vacío si hay contenido: la mejor puntuada siempre entra
        return chosen or order[:1]


def _validate_length(sentence_count: int | None, ratio: float | None) -> None:
    if sentence_count is not None and ratio is not None:
        raise ValueError("indica sentence_count o ratio, no ambos")
    if sentence_count is not None:
        if isinstance(sentence_count, bool) or not isinstance(sentence_count, int):
            raise ValueError("sentence_count debe ser un entero")
        if sentence_count < 1:
            raise ValueError("sentence_count debe ser al menos 1")
    if ratio is not None:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("ratio debe ser un número")
        if not 0 < ratio <= 1:
            raise ValueError("ratio debe estar entre 0 (excluido) y 1")


def _target_size(total: int, sentence_count: int | None, ratio: float | None) -> int:
    """Cantidad de oraciones del resumen: al menos 1 y como mucho el total."""
    if sentence_count is not None:
        return min(sentence_count, total)
    proportion = DEFAULT_RATIO if ratio is None else ratio
    return max(1, min(total, math.ceil(total * proportion)))


def _score(similarity: np.ndarray) -> list[float]:
    """Puntuación de cada oración; con una sola no hace falta grafo."""
    if similarity.shape[0] == 1:
        return [1.0]
    return _pagerank(similarity)


def _similarity_matrix(sentences: tuple[AnalyzedSentence, ...]) -> np.ndarray:
    """Matriz de similitud coseno entre oraciones (diagonal en cero).

    Las oraciones sin lemas de contenido tienen vector nulo y similitud 0
    con todas; si ninguna tiene contenido, la matriz es completamente nula.
    """
    size = len(sentences)
    documents = [sentence.lemmatized_text for sentence in sentences]
    if not any(documents):
        return np.zeros((size, size))
    vectorizer = TfidfVectorizer(analyzer=str.split)
    vectors = vectorizer.fit_transform(documents)
    # cosine_similarity devuelve 0 para vectores nulos, sin dividir entre cero
    similarity = cosine_similarity(vectors)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _pagerank(similarity: np.ndarray) -> list[float]:
    """Aplica PageRank sobre el grafo ponderado por similitud."""
    size = similarity.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    for i in range(size):
        for j in range(i + 1, size):
            if similarity[i][j] > 0:
                graph.add_edge(i, j, weight=float(similarity[i][j]))

    if graph.number_of_edges() == 0:
        return [1.0 / size] * size
    try:
        ranks = nx.pagerank(
            graph,
            alpha=PAGERANK_DAMPING,
            weight="weight",
            max_iter=PAGERANK_MAX_ITERATIONS,
            tol=PAGERANK_TOLERANCE,
        )
    except nx.PowerIterationFailedConvergence:
        # alternativa estable: centralidad por suma de similitudes
        strengths = similarity.sum(axis=1)
        total = strengths.sum()
        return [float(value / total) for value in strengths]
    return [ranks[i] for i in range(size)]

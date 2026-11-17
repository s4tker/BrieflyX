"""Pruebas del motor de resumen TextRank."""

import pytest

from NLP.Summarizer import DEFAULT_RATIO, SummaryResult, Summarizer
from NLP.TextAnalyzer import TextAnalyzer

CLIMATE_TEXT = (
    "El cambio climático es uno de los mayores desafíos de este siglo. "
    "Las temperaturas globales han aumentado de forma constante desde la revolución industrial. "
    "Este aumento de las temperaturas provoca el deshielo de los glaciares y la subida del nivel del mar. "
    "Muchas ciudades costeras ya sufren inundaciones más frecuentes por la subida del nivel del mar. "
    "Ayer llovió en mi barrio y el partido de fútbol se suspendió. "
    "Los científicos coinciden en que la quema de combustibles fósiles es la principal causa del cambio climático. "
    "Reducir el uso de combustibles fósiles exige invertir en energías renovables como la solar y la eólica. "
    "Varios países ya producen gran parte de su electricidad con energías renovables. "
    "Mi primo se compró una bicicleta nueva la semana pasada. "
    "Sin una acción coordinada, los efectos del cambio climático serán cada vez más graves."
)
OFF_TOPIC = (
    "Ayer llovió en mi barrio y el partido de fútbol se suspendió.",
    "Mi primo se compró una bicicleta nueva la semana pasada.",
)


@pytest.fixture(scope="module")
def summarizer() -> Summarizer:
    return Summarizer(TextAnalyzer())


def assert_extractive(result: SummaryResult, text: str) -> None:
    """Cada oración elegida es un fragmento literal del texto, en orden."""
    for sentence in result.sentences:
        assert text[sentence.start:sentence.end] == sentence.text
        assert sentence.text in text
    indexes = [sentence.index for sentence in result.sentences]
    assert indexes == sorted(indexes)


class TestShortTexts:
    def test_two_sentences(self, summarizer):
        text = "El sol es una estrella. La luna gira alrededor de la tierra."
        result = summarizer.summarize(text, sentence_count=1)
        assert len(result.sentences) == 1
        assert_extractive(result, text)

    def test_single_sentence_skips_graph(self, summarizer):
        text = "Una sola oración con contenido relevante."
        result = summarizer.summarize(text)
        assert result.summary_text == text
        assert [s.score for s in result.ranking] == [1.0]
        assert result.reduction_percentage == 0.0

    def test_few_sentences_never_exceed_total(self, summarizer):
        text = "El gato duerme. El perro ladra."
        result = summarizer.summarize(text, sentence_count=10)
        assert len(result.sentences) == 2


class TestLongTexts:
    def test_long_text_is_reduced(self, summarizer):
        paragraphs = [CLIMATE_TEXT] * 1 + [
            f"El informe número {i} analiza el impacto económico de la sequía en la agricultura regional."
            for i in range(40)
        ]
        text = " ".join(paragraphs)
        result = summarizer.summarize(text, ratio=0.1)
        assert result.original_sentence_count == 50
        assert 1 <= len(result.sentences) <= 5
        assert result.reduction_percentage > 50
        assert_extractive(result, text)

    def test_ranking_covers_every_sentence(self, summarizer):
        result = summarizer.summarize(CLIMATE_TEXT)
        assert result.original_sentence_count == 10
        assert [s.index for s in result.ranking] == list(range(10))
        assert sum(s.score for s in result.ranking) == pytest.approx(1.0)


class TestRelevance:
    def test_off_topic_sentences_score_lowest(self, summarizer):
        result = summarizer.summarize(CLIMATE_TEXT, sentence_count=4)
        lowest = sorted(result.ranking, key=lambda s: s.score)[:2]
        assert {s.text for s in lowest} == set(OFF_TOPIC)
        assert not set(OFF_TOPIC) & {s.text for s in result.sentences}

    def test_selection_follows_scores_not_position(self, summarizer):
        result = summarizer.summarize(CLIMATE_TEXT, sentence_count=4)
        chosen = {s.index for s in result.sentences}
        assert chosen != {0, 1, 2, 3}
        best = max(result.ranking, key=lambda s: s.score)
        assert best.index in chosen

    def test_central_idea_wins_over_isolated_ones(self, summarizer):
        text = (
            "La biblioteca abre a las nueve. "
            "La biblioteca municipal ofrece libros y talleres gratuitos. "
            "Los talleres de la biblioteca municipal son para niños y adultos. "
            "Mañana hará calor."
        )
        result = summarizer.summarize(text, sentence_count=1)
        assert "biblioteca" in result.summary_text


class TestRedundancy:
    def test_repeated_sentences_are_selected_once(self, summarizer):
        text = (
            "El proyecto terminó a tiempo. El proyecto terminó a tiempo. "
            "El presupuesto se redujo un diez por ciento. El equipo celebró el resultado."
        )
        result = summarizer.summarize(text, sentence_count=3)
        texts = [s.text for s in result.sentences]
        assert texts.count("El proyecto terminó a tiempo.") <= 1
        assert len(texts) == len(set(texts))

    def test_near_duplicates_are_filtered(self, summarizer):
        text = (
            "La empresa lanzó un nuevo teléfono móvil. "
            "La empresa lanzó un nuevo teléfono móvil ayer. "
            "Las ventas del producto superaron las expectativas del mercado."
        )
        result = summarizer.summarize(text, sentence_count=2)
        lanzamientos = [s for s in result.sentences if "lanzó" in s.text]
        assert len(lanzamientos) == 1

    def test_all_identical_sentences_return_one(self, summarizer):
        result = summarizer.summarize("ok\nok\nok")
        assert result.summary_text == "ok"


class TestLengthConfiguration:
    @pytest.mark.parametrize("count", [1, 2, 3, 5])
    def test_sentence_count(self, summarizer, count):
        result = summarizer.summarize(CLIMATE_TEXT, sentence_count=count)
        assert len(result.sentences) == count
        assert_extractive(result, CLIMATE_TEXT)

    @pytest.mark.parametrize(("ratio", "expected"), [(0.1, 1), (0.3, 3), (0.5, 5), (1.0, 10)])
    def test_ratio(self, summarizer, ratio, expected):
        result = summarizer.summarize(CLIMATE_TEXT, ratio=ratio)
        assert len(result.sentences) == expected

    def test_default_ratio(self, summarizer):
        result = summarizer.summarize(CLIMATE_TEXT)
        assert DEFAULT_RATIO == 0.3
        assert len(result.sentences) == 3

    def test_smaller_ratio_reduces_more(self, summarizer):
        short = summarizer.summarize(CLIMATE_TEXT, ratio=0.2)
        long = summarizer.summarize(CLIMATE_TEXT, ratio=0.6)
        assert short.reduction_percentage > long.reduction_percentage
        assert short.summary_word_count < long.summary_word_count

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sentence_count": 0},
            {"sentence_count": -2},
            {"sentence_count": 2.5},
            {"sentence_count": True},
            {"ratio": 0},
            {"ratio": 1.5},
            {"ratio": -0.1},
            {"ratio": "0.3"},
            {"sentence_count": 2, "ratio": 0.3},
        ],
    )
    def test_invalid_length_parameters(self, summarizer, kwargs):
        with pytest.raises(ValueError):
            summarizer.summarize(CLIMATE_TEXT, **kwargs)


class TestOrder:
    def test_summary_keeps_original_order(self, summarizer):
        result = summarizer.summarize(CLIMATE_TEXT, sentence_count=5)
        positions = [CLIMATE_TEXT.index(s.text) for s in result.sentences]
        assert positions == sorted(positions)

    def test_summary_text_joins_selected_sentences(self, summarizer):
        result = summarizer.summarize(CLIMATE_TEXT, sentence_count=3)
        assert result.summary_text == " ".join(s.text for s in result.sentences)


class TestIrregularInput:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_empty_input(self, summarizer, text):
        result = summarizer.summarize(text)
        assert result.is_empty
        assert result.summary_text == ""
        assert result.reduction_percentage == 0.0

    def test_rejects_non_string(self, summarizer):
        with pytest.raises(TypeError):
            summarizer.summarize(None)  # type: ignore[arg-type]

    def test_irregular_punctuation(self, summarizer):
        text = "el examen fue dificil!!! la profe explico todo mal... no entendi nada???\nel examen tenia preguntas de temas que no vimos"
        result = summarizer.summarize(text, sentence_count=2)
        assert len(result.sentences) == 2
        assert_extractive(result, text)

    def test_text_without_punctuation(self, summarizer):
        text = "un texto largo sin ningun signo de puntuacion que igual debe resumirse"
        result = summarizer.summarize(text)
        assert result.summary_text == text

    def test_sentences_without_content(self, summarizer):
        text = "Hola. Sí. Bueno. Gracias."
        result = summarizer.summarize(text, sentence_count=2)
        assert len(result.sentences) == 2
        assert_extractive(result, text)


class TestDisconnectedGraph:
    def test_no_shared_terms_gives_equal_scores(self, summarizer):
        text = "Los gatos duermen. Mañana llueve. Compré pan."
        result = summarizer.summarize(text, sentence_count=1)
        scores = {round(s.score, 9) for s in result.ranking}
        assert len(scores) == 1
        assert len(result.sentences) == 1

    def test_ties_prefer_more_content(self, summarizer):
        text = "Hola. Los astronautas repararon el telescopio espacial."
        result = summarizer.summarize(text, sentence_count=1)
        assert result.summary_text == "Los astronautas repararon el telescopio espacial."

    def test_partially_connected_graph(self, summarizer):
        text = (
            "El tren de alta velocidad une ambas ciudades. "
            "El tren de alta velocidad tarda dos horas. "
            "Las flores del jardín son amarillas."
        )
        result = summarizer.summarize(text, sentence_count=1)
        assert "tren" in result.summary_text


class TestReuseAnalysis:
    def test_summarize_analysis_matches_summarize(self, summarizer):
        analysis = TextAnalyzer().analyze(CLIMATE_TEXT)
        from_analysis = summarizer.summarize_analysis(analysis, sentence_count=3)
        from_text = summarizer.summarize(CLIMATE_TEXT, sentence_count=3)
        assert from_analysis == from_text

    def test_result_is_immutable(self, summarizer):
        result = summarizer.summarize(CLIMATE_TEXT)
        with pytest.raises(AttributeError):
            result.sentences = ()  # type: ignore[misc]

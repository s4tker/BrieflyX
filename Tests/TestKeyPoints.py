"""Pruebas de la identificación de ideas principales."""

import pytest

from NLP.EmojiProcessor import EmojiProcessor
from NLP.KeyPoints import DEFAULT_KEY_POINT_COUNT, KeyPointIdentifier, KeyPointsResult
from NLP.KeywordExtractor import KeywordExtractor
from NLP.Summarizer import Summarizer
from NLP.TextAnalyzer import TextAnalyzer
from NLP.TextNormalizer import TextNormalizer
from Tests.TestSummarizer import CLIMATE_TEXT, OFF_TOPIC


@pytest.fixture(scope="module")
def analyzer() -> TextAnalyzer:
    return TextAnalyzer()


@pytest.fixture(scope="module")
def identifier(analyzer) -> KeyPointIdentifier:
    return KeyPointIdentifier(analyzer)


def assert_original(result: KeyPointsResult, text: str) -> None:
    """Cada idea es una oración literal del texto y se respeta el orden."""
    for point in result.points:
        assert text[point.start:point.end] == point.text
    indexes = [point.index for point in result.points]
    assert indexes == sorted(indexes)


class TestSeveralIdeas:
    def test_distinct_ideas_are_covered(self, identifier):
        result = identifier.identify(CLIMATE_TEXT, count=3)
        joined = " ".join(result.texts)
        assert "cambio climático" in joined
        assert "combustibles fósiles" in joined or "energías renovables" in joined
        assert "nivel del mar" in joined or "temperaturas" in joined

    def test_off_topic_sentences_are_excluded(self, identifier):
        result = identifier.identify(CLIMATE_TEXT, count=5)
        assert not set(OFF_TOPIC) & set(result.texts)

    def test_not_simply_the_first_sentences(self, identifier):
        result = identifier.identify(CLIMATE_TEXT, count=3)
        assert [p.index for p in result.points] != [0, 1, 2]

    def test_points_list_their_concepts(self, identifier):
        result = identifier.identify(CLIMATE_TEXT, count=3)
        assert all(point.keywords for point in result.points)
        for point in result.points:
            for keyword in point.keywords:
                assert keyword.split()[0][:4].lower() in point.text.lower()


class TestRedundancy:
    def test_similar_sentences_yield_one_idea(self, identifier):
        text = (
            "La subida del nivel del mar amenaza las ciudades costeras. "
            "Las ciudades costeras sufren por la subida del nivel del mar. "
            "Los gobiernos invierten en energías renovables para reducir emisiones."
        )
        result = identifier.identify(text, count=2)
        costeras = [t for t in result.texts if "costeras" in t]
        assert len(costeras) == 1
        assert any("renovables" in t for t in result.texts)

    def test_repeated_sentences_are_selected_once(self, identifier):
        text = (
            "El proyecto de software terminó a tiempo. "
            "El proyecto de software terminó a tiempo. "
            "El cliente aprobó el diseño final de la aplicación."
        )
        result = identifier.identify(text, count=3)
        assert len(result.texts) == len(set(result.texts))


class TestFewSentences:
    def test_single_sentence(self, identifier):
        text = "La biblioteca municipal amplía su horario durante los exámenes."
        assert identifier.identify(text).texts == [text]

    def test_short_sentences_only_when_needed(self, identifier):
        text = "Hola. Los astronautas repararon el telescopio espacial en órbita. Gracias."
        assert identifier.identify(text, count=1).texts == [
            "Los astronautas repararon el telescopio espacial en órbita."
        ]

    def test_only_short_sentences_still_returns_something(self, identifier):
        result = identifier.identify("Hola. Sí. Gracias.", count=2)
        assert len(result.points) == 2


class TestOrderAndOriginality:
    def test_original_order(self, identifier):
        result = identifier.identify(CLIMATE_TEXT, count=4)
        assert_original(result, CLIMATE_TEXT)

    def test_original_wording(self, identifier):
        result = identifier.identify(CLIMATE_TEXT, count=4)
        for text in result.texts:
            assert text in CLIMATE_TEXT

    def test_chat_ideas_come_from_prepared_text(self, identifier, analyzer):
        chat = (
            "oye vieron lo del examen de matemáticas?? 😡\n"
            "el examen de matemáticas fue super difícil, nadie terminó a tiempo\n"
            "jajaja yo tmb 😂\n"
            "la profe dijo q el examen de matemáticas se puede recuperar el viernes\n"
            "ok"
        )
        normalized = TextNormalizer().normalize(chat).normalized_text
        prepared = EmojiProcessor().process(normalized).processed_text
        result = identifier.identify_from_analysis(analyzer.analyze(prepared), count=2)
        assert_original(result, prepared)
        assert "el examen de matemáticas fue super difícil, nadie terminó a tiempo" in result.texts


class TestConfiguration:
    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_count(self, identifier, count):
        assert len(identifier.identify(CLIMATE_TEXT, count=count).points) == count

    def test_default_count(self, identifier):
        result = identifier.identify(CLIMATE_TEXT)
        assert 1 <= len(result.points) <= DEFAULT_KEY_POINT_COUNT

    @pytest.mark.parametrize("count", [0, -2, 1.5, False])
    def test_invalid_count(self, identifier, count):
        with pytest.raises(ValueError):
            identifier.identify(CLIMATE_TEXT, count=count)


class TestSharedInformation:
    def test_reuses_keywords_and_ranking(self, identifier, analyzer):
        analysis = analyzer.analyze(CLIMATE_TEXT)
        keywords = KeywordExtractor(analyzer).extract_from_analysis(analysis, top_n=15)
        ranking = Summarizer(analyzer).summarize_analysis(analysis).ranking
        shared = identifier.identify_from_analysis(analysis, 3, keywords=keywords, ranking=ranking)
        alone = identifier.identify(CLIMATE_TEXT, count=3)
        assert shared == alone

    def test_textrank_score_matches_summarizer(self, identifier, analyzer):
        ranking = Summarizer(analyzer).summarize(CLIMATE_TEXT).ranking
        result = identifier.identify(CLIMATE_TEXT, count=3)
        for point in result.points:
            assert point.textrank_score == pytest.approx(ranking[point.index].score)


class TestEdgeCases:
    @pytest.mark.parametrize("text", ["", "  \n "])
    def test_empty_input(self, identifier, text):
        result = identifier.identify(text)
        assert result.is_empty
        assert result.texts == []

    def test_rejects_non_string(self, identifier):
        with pytest.raises(TypeError):
            identifier.identify(None)  # type: ignore[arg-type]

    def test_content_never_gives_empty_result(self, identifier):
        assert not identifier.identify("ok\nok\nok").is_empty

    def test_result_is_immutable(self, identifier):
        result = identifier.identify(CLIMATE_TEXT)
        with pytest.raises(AttributeError):
            result.points = ()  # type: ignore[misc]

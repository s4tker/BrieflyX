"""Pruebas del extractor de palabras clave."""

import pytest

from NLP.KeywordExtractor import (
    DEFAULT_TOP_N,
    KeywordExtractor,
    KeywordResult,
    variant_key,
)
from NLP.TextAnalyzer import TextAnalyzer
from Tests.TestSummarizer import CLIMATE_TEXT


@pytest.fixture(scope="module")
def analyzer() -> TextAnalyzer:
    return TextAnalyzer()


@pytest.fixture(scope="module")
def extractor(analyzer) -> KeywordExtractor:
    return KeywordExtractor(analyzer)


def terms(extractor: KeywordExtractor, text: str, top_n: int = DEFAULT_TOP_N) -> list[str]:
    return extractor.extract(text, top_n=top_n).terms


class TestShortAndLongTexts:
    def test_short_text(self, extractor):
        result = terms(extractor, "El telescopio espacial captó una galaxia lejana.")
        assert "telescopio" in result
        assert result

    def test_long_text_main_concept_first(self, extractor):
        assert terms(extractor, CLIMATE_TEXT)[0] == "cambio climático"

    def test_long_text_finds_topic_concepts(self, extractor):
        found = set(terms(extractor, CLIMATE_TEXT, top_n=8))
        assert {"cambio climático", "combustible fósil", "energía renovable", "temperatura"} <= found

    def test_off_topic_words_rank_low(self, extractor):
        found = terms(extractor, CLIMATE_TEXT, top_n=8)
        assert not {"barrio", "bicicleta", "primo", "fútbol"} & set(found)


class TestRelevance:
    def test_repeated_concept_ranks_high(self, extractor):
        text = (
            "La vacuna protege contra la gripe. La vacuna se aplica en otoño. "
            "Los centros de salud ofrecen la vacuna gratis."
        )
        assert terms(extractor, text)[0] == "vacuna"

    def test_not_only_frequency(self, extractor):
        """El idf reduce la ventaja de un término presente en todas las oraciones.

        "informe" aparece 3 veces y "sequía" 2: por frecuencia la distancia
        sería 3/2, pero la relevancia la acorta porque "informe" no distingue
        ninguna oración.
        """
        text = (
            "El informe describe la sequía en el norte. "
            "El informe describe las inundaciones en el sur. "
            "El informe describe la sequía y sus efectos en la agricultura."
        )
        result = extractor.extract(text).keywords
        scores = {k.term: k.score for k in result}
        frequencies = {k.term: k.frequency for k in result}
        frequency_ratio = frequencies["informe"] / frequencies["sequía"]
        score_ratio = scores["informe"] / scores["sequía"]
        assert score_ratio < frequency_ratio
        # "sur" e "inundación" aparecen 1 vez, pero en la oración que no habla de sequía
        assert scores["sequía"] > scores["inundación"]

    def test_scores_sorted_and_normalized(self, extractor):
        keywords = extractor.extract(CLIMATE_TEXT).keywords
        scores = [k.score for k in keywords]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 1.0
        assert all(0 < score <= 1 for score in scores)

    def test_frequency_is_reported(self, extractor):
        keyword = extractor.extract(CLIMATE_TEXT).keywords[0]
        assert keyword.frequency == 3


class TestStopWords:
    def test_text_full_of_stop_words(self, extractor):
        text = "Y entonces él lo hizo y ella también lo hizo porque sí, pero no sé."
        found = terms(extractor, text)
        assert not {"y", "él", "lo", "ella", "también", "porque", "pero", "no"} & set(found)

    def test_only_stop_words_returns_empty(self, extractor):
        assert extractor.extract("y él lo de la que").is_empty

    def test_keywords_are_nouns_or_adjectives(self, extractor):
        text = "Los investigadores publicaron rápidamente un estudio innovador sobre robótica."
        found = terms(extractor, text)
        assert "rápidamente" not in found
        assert "publicar" not in found
        assert {"estudio", "robótica"} <= set(found)


class TestTechnicalTerms:
    def test_technical_vocabulary(self, extractor):
        text = (
            "El algoritmo de aprendizaje automático entrena una red neuronal. "
            "La red neuronal ajusta sus pesos con descenso de gradiente. "
            "El algoritmo evalúa la red neuronal con datos de validación."
        )
        found = terms(extractor, text, top_n=5)
        assert "red neuronal" in found
        assert "algoritmo" in found

    def test_proper_nouns_keep_their_spelling(self, extractor):
        text = (
            "La NASA lanzó un telescopio. El telescopio de la NASA observará galaxias. "
            "Las Naciones Unidas felicitaron a la NASA. Las Naciones Unidas apoyan la ciencia."
        )
        found = terms(extractor, text)
        assert found[0] == "NASA"
        assert "Naciones Unidas" in found


class TestVariants:
    def test_inflections_are_merged(self, extractor):
        text = "Los gatos duermen. El gato come. Un gato maúlla."
        result = extractor.extract(text).keywords
        assert [k.term for k in result].count("gato") == 1
        gato = next(k for k in result if k.term == "gato")
        assert gato.frequency == 3
        assert set(gato.variants) == {"gatos", "gato"}

    def test_derived_words_are_grouped(self, extractor):
        text = (
            "La economía crece. El crecimiento económico depende de la inversión. "
            "La economía regional mejora."
        )
        found = terms(extractor, text)
        assert "economía" in found
        assert "económico" not in found

    @pytest.mark.parametrize(
        ("first", "second"),
        [("economía", "económico"), ("política", "político"), ("energía", "energías")],
    )
    def test_variant_key(self, first, second):
        assert variant_key(first) == variant_key(second)

    def test_unrelated_words_have_different_keys(self):
        assert variant_key("mar") != variant_key("marea")

    def test_phrase_replaces_its_words(self, extractor):
        found = terms(extractor, CLIMATE_TEXT)
        assert "cambio climático" in found
        assert "cambio" not in found
        assert "climático" not in found

    def test_phrase_needs_repetition(self, extractor):
        found = terms(extractor, "El perro negro ladra. El gato duerme.")
        assert "perro negro" not in found


class TestConfiguration:
    @pytest.mark.parametrize("top_n", [1, 3, 5])
    def test_top_n(self, extractor, top_n):
        assert len(extractor.extract(CLIMATE_TEXT, top_n=top_n).keywords) == top_n

    def test_top_n_larger_than_candidates(self, extractor):
        result = extractor.extract("El sol brilla.", top_n=50)
        assert 1 <= len(result.keywords) < 50

    @pytest.mark.parametrize("top_n", [0, -1, 2.5, True, "3"])
    def test_invalid_top_n(self, extractor, top_n):
        with pytest.raises(ValueError):
            extractor.extract(CLIMATE_TEXT, top_n=top_n)


class TestEdgeCases:
    @pytest.mark.parametrize("text", ["", "   ", "!!! ???"])
    def test_empty_input(self, extractor, text):
        result = extractor.extract(text)
        assert isinstance(result, KeywordResult)
        assert result.is_empty

    def test_rejects_non_string(self, extractor):
        with pytest.raises(TypeError):
            extractor.extract(None)  # type: ignore[arg-type]

    def test_only_verbs_falls_back_to_content_words(self, extractor):
        assert not extractor.extract("Corrimos, saltamos y nadamos.").is_empty

    def test_reuses_analysis(self, extractor, analyzer):
        analysis = analyzer.analyze(CLIMATE_TEXT)
        assert extractor.extract_from_analysis(analysis) == extractor.extract(CLIMATE_TEXT)

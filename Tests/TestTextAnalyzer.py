"""Pruebas del analizador lingüístico."""

import pytest

from NLP.EmojiProcessor import EmojiProcessor
from NLP.TextAnalyzer import (
    MAX_TEXT_LENGTH,
    LanguageModelNotFoundError,
    TextAnalysis,
    TextAnalyzer,
    load_language_model,
)
from NLP.TextNormalizer import TextNormalizer


@pytest.fixture(scope="module")
def analyzer() -> TextAnalyzer:
    return TextAnalyzer()


def sentence_texts(analyzer: TextAnalyzer, text: str) -> list[str]:
    return analyzer.analyze(text).sentence_texts


class TestTokenization:
    def test_splits_words_and_punctuation(self, analyzer):
        tokens = analyzer.analyze("¡Hola, mundo!").tokens
        assert [token.text for token in tokens] == ["¡", "Hola", ",", "mundo", "!"]
        assert [token.is_punct for token in tokens] == [True, False, True, False, True]

    def test_token_offsets_point_to_text(self, analyzer):
        text = "Los gatos   corren."
        for token in analyzer.analyze(text).tokens:
            assert text[token.start:token.end] == token.text

    def test_word_count_ignores_punctuation_and_numbers(self, analyzer):
        analysis = analyzer.analyze("Compré 3 libros, ¿y tú?")
        assert analysis.word_count == 4  # compré, libros, y, tú

    def test_has_part_of_speech(self, analyzer):
        tokens = analyzer.analyze("El gato duerme").tokens
        assert [token.pos for token in tokens] == ["DET", "NOUN", "VERB"]


class TestSentenceSegmentation:
    def test_basic_punctuation(self, analyzer):
        text = "¿Qué haces? ¡No lo puedo creer! Estoy cansado... Mañana sigo."
        assert sentence_texts(analyzer, text) == [
            "¿Qué haces?",
            "¡No lo puedo creer!",
            "Estoy cansado...",
            "Mañana sigo.",
        ]

    def test_abbreviations_do_not_split(self, analyzer):
        text = "El Dr. Pérez llegó a las 3 p.m. y habló con la Sra. López. Después se fue."
        assert sentence_texts(analyzer, text) == [
            "El Dr. Pérez llegó a las 3 p.m. y habló con la Sra. López.",
            "Después se fue.",
        ]

    def test_abbreviation_before_new_sentence_still_splits(self, analyzer):
        text = "Llegó a las 5 p.m. Luego cenó."
        assert sentence_texts(analyzer, text) == ["Llegó a las 5 p.m.", "Luego cenó."]

    def test_ellipsis_followed_by_lowercase_continues(self, analyzer):
        text = "y bueno... no sé qué decir. Otra cosa."
        assert sentence_texts(analyzer, text) == ["y bueno... no sé qué decir.", "Otra cosa."]

    def test_opening_marks_start_the_next_sentence(self, analyzer):
        text = "Llegué. ¿Vienes? ¡Apúrate!"
        assert sentence_texts(analyzer, text) == ["Llegué.", "¿Vienes?", "¡Apúrate!"]

    def test_line_breaks_split_chat_messages(self, analyzer):
        text = "hola como estas\nyo bien gracias\n\nnos vemos mañana"
        assert sentence_texts(analyzer, text) == [
            "hola como estas",
            "yo bien gracias",
            "nos vemos mañana",
        ]

    def test_text_without_final_punctuation(self, analyzer):
        assert sentence_texts(analyzer, "una sola idea sin punto") == ["una sola idea sin punto"]

    def test_sentence_offsets_and_indexes(self, analyzer):
        text = "  Primera oración.   Segunda oración.  "
        analysis = analyzer.analyze(text)
        assert [s.index for s in analysis.sentences] == [0, 1]
        for sentence in analysis.sentences:
            assert text[sentence.start:sentence.end] == sentence.text
            assert sentence.text == sentence.text.strip()


class TestLemmatization:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Los gatos corrían por las calles", ["gato", "correr", "calle"]),
            ("Las niñas cantaron canciones", ["niña", "cantar", "canción"]),
        ],
    )
    def test_content_lemmas(self, analyzer, text, expected):
        assert analyzer.analyze(text).content_lemmas == expected

    def test_lemmas_are_lowercase(self, analyzer):
        lemmas = [token.lemma for token in analyzer.analyze("GATOS y Perros").tokens]
        assert all(lemma == lemma.lower() for lemma in lemmas)

    def test_abbreviation_period_is_not_part_of_lemma(self, analyzer):
        lemmas = [token.lemma for token in analyzer.analyze("Habló el Dr. Pérez").tokens]
        assert "dr." not in lemmas

    def test_lemma_frequencies_group_inflections(self, analyzer):
        analysis = analyzer.analyze("El gato come. Los gatos duermen. Un gato juega.")
        assert analysis.lemma_frequencies["gato"] == 3
        assert next(iter(analysis.lemma_frequencies)) == "gato"

    def test_lemmatized_text_for_algorithms(self, analyzer):
        sentence = analyzer.analyze("Los gatos corrían por las calles.").sentences[0]
        assert sentence.lemmatized_text == "gato correr calle"


class TestStopWords:
    def test_function_words_are_stop_words(self, analyzer):
        tokens = {t.text: t for t in analyzer.analyze("el gato de la casa y yo").tokens}
        for word in ["el", "de", "la", "y", "yo"]:
            assert tokens[word].is_stop
        assert not tokens["gato"].is_stop
        assert not tokens["casa"].is_stop

    def test_content_excludes_stop_words_and_punctuation(self, analyzer):
        assert analyzer.analyze("¡El sistema, de la empresa!").content_lemmas == [
            "sistema",
            "empresa",
        ]

    @pytest.mark.parametrize("text", ["jaja", "la Sra. López", "el Dr. Pérez"])
    def test_chat_noise_and_titles_are_not_content(self, analyzer, text):
        lemmas = analyzer.analyze(text).content_lemmas
        assert not {"jaja", "sra", "dr"} & set(lemmas)


class TestNegations:
    @pytest.mark.parametrize(
        "text",
        ["No me gusta", "Nunca vino", "ni idea", "Nadie llamó", "Tampoco fue"],
    )
    def test_negations_are_flagged(self, analyzer, text):
        assert analyzer.analyze(text).sentences[0].has_negation

    def test_negation_tokens_are_kept(self, analyzer):
        tokens = analyzer.analyze("No me gusta nada").tokens
        assert [t.text for t in tokens if t.is_negation] == ["No", "nada"]

    @pytest.mark.parametrize("text", ["Me gusta mucho", "Sin embargo, llegó a tiempo"])
    def test_affirmative_sentence(self, analyzer, text):
        assert not analyzer.analyze(text).sentences[0].has_negation


class TestPipelineCompatibility:
    def test_normalized_chat_text(self, analyzer):
        raw = "toy bn xfa 😂😂\nntp, mñn hablamos"
        normalized = TextNormalizer().normalize(raw).normalized_text
        processed = EmojiProcessor().process(normalized).processed_text
        analysis = analyzer.analyze(processed)
        assert analysis.sentence_texts == [
            "estoy bien por favor risa",
            "no te preocupes, mañana hablamos",
        ]
        assert "risa" in analysis.content_lemmas
        assert analysis.sentences[1].has_negation

    def test_original_text_is_not_modified(self, analyzer):
        text = "  Texto  original.\nCon dos líneas  "
        analysis = analyzer.analyze(text)
        assert analysis.text == text
        assert text == "  Texto  original.\nCon dos líneas  "


class TestEdgeCases:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
    def test_empty_text(self, analyzer, text):
        analysis = analyzer.analyze(text)
        assert isinstance(analysis, TextAnalysis)
        assert analysis.is_empty
        assert analysis.word_count == 0
        assert analysis.content_lemmas == []

    def test_only_punctuation(self, analyzer):
        analysis = analyzer.analyze("!!! ... ???")
        assert analysis.word_count == 0
        assert analysis.content_lemmas == []

    def test_rejects_non_string(self, analyzer):
        with pytest.raises(TypeError):
            analyzer.analyze(None)  # type: ignore[arg-type]

    def test_rejects_too_long_text(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze("a" * (MAX_TEXT_LENGTH + 1))

    def test_model_is_loaded_once(self):
        assert load_language_model() is load_language_model()

    def test_missing_model_has_clear_error(self):
        with pytest.raises(LanguageModelNotFoundError, match="spacy download"):
            TextAnalyzer("modelo_inexistente_xx")

    def test_result_is_immutable(self, analyzer):
        analysis = analyzer.analyze("Hola mundo.")
        with pytest.raises(AttributeError):
            analysis.text = "otro"  # type: ignore[misc]

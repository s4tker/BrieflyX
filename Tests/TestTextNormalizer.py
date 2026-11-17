"""Pruebas del normalizador de lenguaje informal."""

import pytest

from NLP.TextNormalizer import (
    CATEGORY_ABBREVIATION,
    CATEGORY_REPETITION,
    NormalizationResult,
    TextNormalizer,
)


@pytest.fixture(scope="module")
def normalizer() -> TextNormalizer:
    return TextNormalizer()


def normalized(normalizer: TextNormalizer, text: str) -> str:
    return normalizer.normalize(text).normalized_text


class TestAbbreviations:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("q", "que"),
            ("bn", "bien"),
            ("xfa", "por favor"),
            ("tmb", "también"),
            ("ntp", "no te preocupes"),
            ("toy", "estoy"),
            ("voy pa la casa", "voy para la casa"),
            ("nos vemos mñn", "nos vemos mañana"),
            ("salu2 a todos", "saludos a todos"),
        ],
    )
    def test_known_abbreviations(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected

    def test_combined_message(self, normalizer):
        text = "toy bn, tmb te aviso xfa"
        assert normalized(normalizer, text) == "estoy bien, también te aviso por favor"

    def test_preserves_initial_capital(self, normalizer):
        assert normalized(normalizer, "Tmb voy") == "También voy"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("¿xq no viniste?", "¿por qué no viniste?"),
            ("¿y eso xq?", "¿y eso por qué?"),
            ("no sé xq se fue", "no sé por qué se fue"),
            ("no vine xq estaba cansado", "no vine porque estaba cansado"),
            ("¿vienes xq quieres?", "¿vienes porque quieres?"),
        ],
    )
    def test_xq_depends_on_context(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("¿dnd estás?", "¿dónde estás?"),
            ("¿y tú dnd estás?", "¿y tú dónde estás?"),
            ("no sé dnd está", "no sé dónde está"),
            ("la casa dnd vivo", "la casa donde vivo"),
        ],
    )
    def test_dnd_depends_on_context(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("¿q haces?", "¿qué haces?"),
            ("¿a q hora llegas?", "¿a qué hora llegas?"),
            ("dijo q vendría", "dijo que vendría"),
        ],
    )
    def test_q_depends_on_context(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected

    def test_records_changes(self, normalizer):
        result = normalizer.normalize("toy bn")
        assert [(c.original, c.replacement, c.category) for c in result.changes] == [
            ("toy", "estoy", CATEGORY_ABBREVIATION),
            ("bn", "bien", CATEGORY_ABBREVIATION),
        ]


class TestInformalExpressions:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("ta bien", "está bien"),
            ("x fa", "por favor"),
            ("pa' que sepas", "para que sepas"),
            ("osea no", "o sea no"),
            ("enserio talvez", "en serio tal vez"),
            ("jajajajaja", "jaja"),
            ("jsjsjs q risa", "jaja que risa"),
        ],
    )
    def test_expressions(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected


class TestRepeatedLetters:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("holaaaa", "hola"),
            ("graaaacias", "gracias"),
            ("buenooooo", "bueno"),
            ("el carrrro", "el carro"),
            ("la calllle", "la calle"),
            ("xfaaaa", "por favor"),
        ],
    )
    def test_reduces_stretched_words(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected

    @pytest.mark.parametrize("text", ["leer", "creer", "perro", "llamar", "acción", "mmm"])
    def test_keeps_legitimate_doubles(self, normalizer, text):
        assert normalized(normalizer, text) == text

    def test_keeps_roman_numerals(self, normalizer):
        assert normalized(normalizer, "Capítulo III y siglo XXX") == "Capítulo III y siglo XXX"

    def test_records_repetition_change(self, normalizer):
        result = normalizer.normalize("holaaa")
        assert result.changes[0].category == CATEGORY_REPETITION


class TestCase:
    def test_lowercases_shouting(self, normalizer):
        assert normalized(normalizer, "NO PUEDO CREERLO") == "No puedo creerlo"

    def test_lowercases_stretched_caps_word(self, normalizer):
        assert normalized(normalizer, "HOLAAAA amigo") == "Hola amigo"

    def test_keeps_isolated_acronyms(self, normalizer):
        assert normalized(normalizer, "la ONU y la OEA") == "la ONU y la OEA"

    def test_keeps_known_acronyms_inside_shouting(self, normalizer):
        assert normalized(normalizer, "LO DIJO LA ONU") == "Lo dijo la ONU"


class TestSpacingAndPunctuation:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("hola    mundo", "hola mundo"),
            ("  hola  ", "hola"),
            ("hola , mundo", "hola, mundo"),
            ("hola,mundo", "hola, mundo"),
            ("¡¡¡ hola !!!", "¡hola!"),
            ("¿¿qué??", "¿qué?"),
            ("¿en serio?!?!", "¿en serio?!"),
            ("bueno......", "bueno..."),
            ("bueno...", "bueno..."),
            ("uno,, dos", "uno, dos"),
            ("a\n\n\n\nb", "a\n\nb"),
        ],
    )
    def test_cleans_format(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected

    def test_keeps_decimal_numbers(self, normalizer):
        assert normalized(normalizer, "mide 3,5 metros") == "mide 3,5 metros"


class TestNegations:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("nooooo", "no"),
            ("NO QUIERO", "No quiero"),
            ("ntp", "no te preocupes"),
            ("no toy bn", "no estoy bien"),
            ("nunca dije q no", "nunca dije que no"),
            ("ni idea xq", "ni idea por qué"),
        ],
    )
    def test_negations_survive(self, normalizer, text, expected):
        assert normalized(normalizer, text) == expected

    @pytest.mark.parametrize("word", ["no", "ni", "nunca", "jamás", "tampoco", "nadie", "nada"])
    def test_negation_words_are_never_removed(self, normalizer, word):
        assert word in normalized(normalizer, f"yo {word} lo hice").split()


class TestAmbiguousWords:
    @pytest.mark.parametrize(
        "text",
        [
            "mi pa llega mañana",  # pa puede ser papá
            "3 x 4 = 12",  # x como operador
            "vitamina k",  # k como letra
            "ns vemos",  # ns: no sé / nos
            "el punto d",  # d como letra
        ],
    )
    def test_ambiguous_words_are_kept(self, normalizer, text):
        assert normalized(normalizer, text) == text

    def test_without_question_context_xq_is_porque(self, normalizer):
        assert normalized(normalizer, "Xq lo dices") == "Porque lo dices"

    @pytest.mark.parametrize(
        "text",
        [
            "https://ejemplo.com/xq?q=1",
            "escribe a soporte@xfa.com",
            "sigue a @toy_bn",
            "#tqm",
        ],
    )
    def test_protected_tokens_are_untouched(self, normalizer, text):
        assert normalized(normalizer, text) == text


class TestOriginalPreservation:
    def test_original_text_is_kept_verbatim(self, normalizer):
        text = "  holaaa!!!   toy bn xfa  "
        result = normalizer.normalize(text)
        assert result.original_text == text
        assert result.normalized_text == "hola! estoy bien por favor"
        assert text == "  holaaa!!!   toy bn xfa  "

    def test_result_is_immutable(self, normalizer):
        result = normalizer.normalize("toy bn")
        with pytest.raises(AttributeError):
            result.normalized_text = "otro"  # type: ignore[misc]

    def test_clean_text_is_unchanged(self, normalizer):
        text = "El informe se publicó ayer. Incluye tres conclusiones principales."
        result = normalizer.normalize(text)
        assert isinstance(result, NormalizationResult)
        assert result.normalized_text == text
        assert not result.was_modified
        assert result.changes == ()

    def test_empty_text(self, normalizer):
        result = normalizer.normalize("")
        assert result.normalized_text == ""
        assert not result.was_modified

    def test_rejects_non_string(self, normalizer):
        with pytest.raises(TypeError):
            normalizer.normalize(None)  # type: ignore[arg-type]

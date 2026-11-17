"""Pruebas del procesador de emojis."""

import pytest

from NLP.EmojiLexicon import (
    AFFECTION,
    ANGER,
    APPROVAL,
    CRYING,
    ENTHUSIASM,
    JOY,
    LAUGHTER,
    SADNESS,
)
from NLP.EmojiProcessor import (
    UNKNOWN_EMOJI_DESCRIPTION,
    EmojiProcessingResult,
    EmojiProcessor,
    describe_emoji,
)
from NLP.TextNormalizer import TextNormalizer

# pictograma que la biblioteca emoji instalada no tiene en su catálogo
UNKNOWN_PICTOGRAPH = "\U0001FA7D"


@pytest.fixture(scope="module")
def processor() -> EmojiProcessor:
    return EmojiProcessor()


class TestDetection:
    def test_single_emoji(self, processor):
        result = processor.process("me encantó ❤️")
        assert result.has_emojis
        assert [item.emoji for item in result.occurrences] == ["❤️"]

    def test_several_emojis(self, processor):
        result = processor.process("😂 qué día 😡 y al final 👍")
        assert [item.emoji for item in result.occurrences] == ["😂", "😡", "👍"]
        assert result.total_emojis == 3

    def test_positions_point_to_original(self, processor):
        text = "hola 🔥 mundo"
        item = processor.process(text).occurrences[0]
        assert text[item.start:item.end] == "🔥"

    def test_skin_tone_and_zwj_sequences(self, processor):
        result = processor.process("👍🏽 y 👨‍👩‍👧")
        assert [item.emoji for item in result.occurrences] == ["👍🏽", "👨‍👩‍👧"]
        assert result.occurrences[0].category == APPROVAL


class TestMeaning:
    @pytest.mark.parametrize(
        ("symbol", "category"),
        [
            ("😂", LAUGHTER),
            ("😢", SADNESS),
            ("❤️", AFFECTION),
            ("😡", ANGER),
            ("👍", APPROVAL),
            ("😁", JOY),
            ("😭", CRYING),
            ("🔥", ENTHUSIASM),
        ],
    )
    def test_reference_emojis(self, symbol, category):
        _, detected_category, recognized = describe_emoji(symbol)
        assert recognized
        assert detected_category == category

    def test_description_comes_from_library_in_spanish(self):
        description, _, _ = describe_emoji("😂")
        assert description == "cara llorando de risa"

    @pytest.mark.parametrize(
        ("symbol", "category"),
        [
            ("🤣", LAUGHTER),  # fuera del ejemplo del brief
            ("💙", AFFECTION),
            ("😠", ANGER),
            ("❤️‍🔥", AFFECTION),  # clasificado por palabra clave
        ],
    )
    def test_other_emojis_are_classified(self, symbol, category):
        assert describe_emoji(symbol)[1] == category

    def test_emoji_without_emotion_keeps_description(self, processor):
        result = processor.process("voy en 🚗")
        item = result.occurrences[0]
        assert item.recognized
        assert item.category is None
        assert item.label == item.description == "coche"
        assert result.processed_text == "voy en coche"

    @pytest.mark.parametrize("symbol", ["🚒", "✡️"])
    def test_keywords_do_not_overreach(self, symbol):
        assert describe_emoji(symbol)[1] is None


class TestRepetition:
    def test_consecutive_repeats_are_grouped(self, processor):
        result = processor.process("qué rabia 😡😡😡")
        assert len(result.occurrences) == 1
        item = result.occurrences[0]
        assert item.count == 3
        assert item.is_repeated
        assert result.total_emojis == 3
        assert result.processed_text == "qué rabia enojo"

    def test_repeats_separated_by_spaces(self, processor):
        result = processor.process("jaja 😂 😂 😂")
        assert [(item.emoji, item.count) for item in result.occurrences] == [("😂", 3)]

    def test_skin_tones_join_the_same_run(self, processor):
        result = processor.process("👍👍🏽👍🏿")
        assert [(item.emoji, item.count) for item in result.occurrences] == [("👍", 3)]

    def test_repeats_separated_by_words_are_not_grouped(self, processor):
        result = processor.process("😂 hola 😂")
        assert [item.count for item in result.occurrences] == [1, 1]

    def test_counts_weigh_repetitions(self, processor):
        result = processor.process("😂😂😂 pero 😢")
        assert result.category_counts == {LAUGHTER: 3, SADNESS: 1}
        assert result.dominant_category == LAUGHTER


class TestPosition:
    @pytest.mark.parametrize(
        ("text", "processed", "without"),
        [
            ("😂 qué risa", "risa qué risa", "qué risa"),
            ("qué 😂 risa", "qué risa risa", "qué risa"),
            ("qué risa 😂", "qué risa risa", "qué risa"),
            ("genial🔥!", "genial entusiasmo!", "genial!"),
            ("¡😁 llegué!", "¡alegría llegué!", "¡llegué!"),
        ],
    )
    def test_start_middle_end(self, processor, text, processed, without):
        result = processor.process(text)
        assert result.processed_text == processed
        assert result.text_without_emojis == without

    def test_keeps_line_breaks(self, processor):
        result = processor.process("hola 👋\nadiós 😢")
        assert result.text_without_emojis == "hola\nadiós"


class TestWithoutEmojis:
    @pytest.mark.parametrize("text", ["Texto formal sin emojis.", "", "precio: 3,5 € (aprox.)"])
    def test_text_is_unchanged(self, processor, text):
        result = processor.process(text)
        assert not result.has_emojis
        assert result.occurrences == ()
        assert result.processed_text == text
        assert result.text_without_emojis == text
        assert result.category_counts == {}
        assert result.dominant_category is None

    @pytest.mark.parametrize("text", [":)", "✓ listo", "★ destacado"])
    def test_plain_symbols_are_not_emojis(self, processor, text):
        assert not processor.process(text).has_emojis


class TestUnknownEmojis:
    def test_unrecognized_pictograph(self, processor):
        result = processor.process(f"mira esto {UNKNOWN_PICTOGRAPH} raro")
        item = result.occurrences[0]
        assert not item.recognized
        assert item.category is None
        assert item.description == UNKNOWN_EMOJI_DESCRIPTION
        assert result.text_without_emojis == "mira esto raro"

    def test_unknown_does_not_break_known_ones(self, processor):
        result = processor.process(f"😂{UNKNOWN_PICTOGRAPH}")
        assert [item.recognized for item in result.occurrences] == [True, False]


class TestNormalizerCompatibility:
    def test_normalizer_leaves_emojis_untouched(self):
        text = "toy bn 😂😂 xfa ❤️"
        assert TextNormalizer().normalize(text).normalized_text == "estoy bien 😂😂 por favor ❤️"

    def test_normalize_then_process(self, processor):
        normalized = TextNormalizer().normalize("ntp 👍 toy bn xq ya llegué 😁").normalized_text
        result = processor.process(normalized)
        assert result.processed_text == (
            "no te preocupes aprobación estoy bien porque ya llegué alegría"
        )

    def test_order_does_not_matter(self, processor):
        text = "jajajaja 😂😂 tmb toy feliz 🔥"
        normalizer = TextNormalizer()
        first = processor.process(normalizer.normalize(text).normalized_text).processed_text
        second = normalizer.normalize(processor.process(text).processed_text).normalized_text
        assert first == second == "jaja risa también estoy feliz entusiasmo"


class TestOriginalPreservation:
    def test_original_is_kept_verbatim(self, processor):
        text = "  qué rabia 😡😡😡  !! "
        result = processor.process(text)
        assert isinstance(result, EmojiProcessingResult)
        assert result.original_text == text
        assert text == "  qué rabia 😡😡😡  !! "

    def test_result_is_immutable(self, processor):
        result = processor.process("hola 😂")
        with pytest.raises(AttributeError):
            result.processed_text = "otro"  # type: ignore[misc]

    def test_rejects_non_string(self, processor):
        with pytest.raises(TypeError):
            processor.process(None)  # type: ignore[arg-type]

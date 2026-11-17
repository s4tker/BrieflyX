"""Detección e interpretación de emojis.

El procesador nunca modifica el texto recibido: devuelve un
EmojiProcessingResult con el original intacto y dos versiones derivadas:

- ``processed_text``: cada emoji (o racha de emojis iguales) se reemplaza por
  su interpretación en palabras, para que los módulos de análisis conserven
  la información emocional ("genial 🔥🔥" → "genial entusiasmo").
- ``text_without_emojis``: el mismo texto sin emojis, útil para mostrar
  oraciones limpias.

No depende de TextNormalizer: ambos pueden aplicarse en cualquier orden,
porque el normalizador no altera emojis y este módulo no altera palabras.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import emoji

from NLP.EmojiLexicon import CATEGORY_KEYWORDS, EMOJI_CATEGORIES

LANGUAGE = "es"
UNKNOWN_EMOJI_DESCRIPTION = "emoji no reconocido"

VARIATION_SELECTOR = "️"
SKIN_TONE_PATTERN = re.compile("[\U0001F3FB-\U0001F3FF]")
# pictogramas que la versión instalada de la biblioteca todavía no conoce
UNKNOWN_PICTOGRAPH_PATTERN = re.compile("[\U0001F300-\U0001FAFF]")
RUN_SEPARATOR_PATTERN = re.compile(r"[ \t]*")


@dataclass(frozen=True)
class EmojiOccurrence:
    """Un emoji, o una racha del mismo emoji repetido, dentro del texto."""

    emoji: str
    description: str
    category: str | None
    count: int
    start: int
    end: int
    recognized: bool

    @property
    def label(self) -> str:
        """Texto que representa al emoji: su emoción o, si no tiene, su descripción."""
        return self.category or self.description

    @property
    def is_repeated(self) -> bool:
        return self.count > 1


@dataclass(frozen=True)
class EmojiProcessingResult:
    """Resultado inmutable del procesamiento de emojis."""

    original_text: str
    processed_text: str
    text_without_emojis: str
    occurrences: tuple[EmojiOccurrence, ...]

    @property
    def has_emojis(self) -> bool:
        return bool(self.occurrences)

    @property
    def total_emojis(self) -> int:
        """Cantidad total de emojis, contando cada repetición."""
        return sum(item.count for item in self.occurrences)

    @property
    def category_counts(self) -> dict[str, int]:
        """Emociones detectadas con su frecuencia, de mayor a menor."""
        counts: Counter[str] = Counter()
        for item in self.occurrences:
            if item.category:
                counts[item.category] += item.count
        return dict(counts.most_common())

    @property
    def dominant_category(self) -> str | None:
        """Emoción más frecuente, o None si ningún emoji tiene categoría."""
        return next(iter(self.category_counts), None)


class EmojiProcessor:
    """Identifica emojis y los traduce a información contextual en español."""

    def process(self, text: str) -> EmojiProcessingResult:
        """Analiza los emojis de ``text`` sin modificar el original.

        Raises:
            TypeError: si ``text`` no es una cadena.
        """
        if not isinstance(text, str):
            raise TypeError("el texto a procesar debe ser una cadena")

        occurrences = tuple(self._group_runs(text, self._find_emojis(text)))
        processed = _replace_spans(text, occurrences, lambda item: f" {item.label} ")
        without_emojis = _replace_spans(text, occurrences, lambda item: " ")
        return EmojiProcessingResult(
            original_text=text,
            processed_text=_tidy_spacing(processed),
            text_without_emojis=_tidy_spacing(without_emojis),
            occurrences=occurrences,
        )

    @staticmethod
    def _find_emojis(text: str) -> list[tuple[str, int, int]]:
        """Localiza emojis conocidos por la biblioteca y pictogramas desconocidos."""
        found = [
            (match["emoji"], match["match_start"], match["match_end"])
            for match in emoji.emoji_list(text)
        ]
        covered = {index for _, start, end in found for index in range(start, end)}
        found.extend(
            (match.group(), match.start(), match.end())
            for match in UNKNOWN_PICTOGRAPH_PATTERN.finditer(text)
            if match.start() not in covered
        )
        return sorted(found, key=lambda item: item[1])

    @staticmethod
    def _group_runs(
        text: str, found: list[tuple[str, int, int]]
    ) -> list[EmojiOccurrence]:
        """Agrupa emojis iguales y consecutivos ("😂😂😂" o "😂 😂") en una ocurrencia.

        Se comparan sin tono de piel, así "👍👍🏽" cuenta como una sola racha.
        """
        occurrences: list[EmojiOccurrence] = []
        for symbol, start, end in found:
            previous = occurrences[-1] if occurrences else None
            if (
                previous is not None
                and _base_form(previous.emoji) == _base_form(symbol)
                and RUN_SEPARATOR_PATTERN.fullmatch(text, previous.end, start)
            ):
                occurrences[-1] = EmojiOccurrence(
                    emoji=previous.emoji,
                    description=previous.description,
                    category=previous.category,
                    count=previous.count + 1,
                    start=previous.start,
                    end=end,
                    recognized=previous.recognized,
                )
                continue

            description, category, recognized = describe_emoji(symbol)
            occurrences.append(
                EmojiOccurrence(symbol, description, category, 1, start, end, recognized)
            )
        return occurrences


@lru_cache(maxsize=1024)
def describe_emoji(symbol: str) -> tuple[str, str | None, bool]:
    """Devuelve (descripción en español, categoría emocional, reconocido).

    La categoría es None cuando el emoji no expresa una emoción conocida
    (por ejemplo 🚗); en ese caso se usa solo su descripción.
    """
    data = emoji.EMOJI_DATA.get(symbol) or emoji.EMOJI_DATA.get(_base_form(symbol))
    if data is None:
        return UNKNOWN_EMOJI_DESCRIPTION, None, False

    description = _readable_name(emoji.demojize(symbol, language=LANGUAGE))
    if not description:
        description = _readable_name(data["en"])
    return description, _classify(symbol, data["en"]), True


def _classify(symbol: str, english_name: str) -> str | None:
    """Busca la categoría primero en el mapa revisado y luego por palabra clave."""
    base = _base_form(symbol)
    if base in EMOJI_CATEGORIES:
        return EMOJI_CATEGORIES[base]
    name = english_name.lower()
    for keyword, category in CATEGORY_KEYWORDS:
        if keyword in name:
            return category
    return None


def _base_form(symbol: str) -> str:
    """Quita selectores de variación y tonos de piel ("👍🏽" → "👍")."""
    return SKIN_TONE_PATTERN.sub("", symbol.replace(VARIATION_SELECTOR, ""))


def _readable_name(alias: str) -> str:
    """Convierte ":cara_llorando_de_risa:" en "cara llorando de risa".

    Devuelve cadena vacía si la biblioteca no tradujo el emoji.
    """
    if not (alias.startswith(":") and alias.endswith(":")):
        return ""
    return alias.strip(":").replace("_", " ")


def _replace_spans(
    text: str,
    occurrences: tuple[EmojiOccurrence, ...],
    render: Callable[[EmojiOccurrence], str],
) -> str:
    """Sustituye cada ocurrencia por el texto que devuelve ``render``."""
    pieces = []
    cursor = 0
    for item in occurrences:
        pieces.append(text[cursor:item.start])
        pieces.append(render(item))
        cursor = item.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _tidy_spacing(text: str) -> str:
    """Limpia los espacios que deja el reemplazo de emojis, sin tocar saltos de línea."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" +([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([¿¡(\[]) +", r"\1", text)
    return text.strip()

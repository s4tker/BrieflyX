"""Normalización conservadora de lenguaje informal en español.

El normalizador nunca modifica el texto recibido: devuelve un
NormalizationResult con el original intacto, una versión normalizada
independiente y el registro de cada cambio aplicado.

Principio general: ante la duda, no tocar. Solo se transforma lo que tiene
una única lectura razonable o lo que el contexto de la oración permite
resolver con seguridad.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from NLP.InformalLexicon import (
    ABBREVIATIONS,
    COMMON_ERRORS,
    DONDE_FORMS,
    INDIRECT_QUESTION_CUES,
    INFORMAL_PHRASES,
    INTERROGATIVE_LEAD_WORDS,
    KNOWN_ACRONYMS,
    PARA_FORMS,
    PORQUE_FORMS,
    POSSESSIVE_DETERMINERS,
    QUE_FORMS,
)

CATEGORY_ABBREVIATION = "abreviatura"
CATEGORY_EXPRESSION = "expresión informal"
CATEGORY_SPELLING = "error frecuente"
CATEGORY_REPETITION = "letras repetidas"
CATEGORY_LAUGHTER = "risa"
CATEGORY_CASE = "mayúsculas"
CATEGORY_PUNCTUATION = "puntuación"

STANDARD_LAUGHTER = "jaja"

# urls, correos, menciones y hashtags se copian sin cambios
PROTECTED_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+|[@#]\w+",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\w+")
LETTERS_ONLY_PATTERN = re.compile(r"[^\W\d_]+")
REPEATED_LETTER_PATTERN = re.compile(r"([^\W\d_])\1{2,}", re.IGNORECASE)
ROMAN_NUMERAL_PATTERN = re.compile(r"^[IVXLCDM]+$")
LAUGHTER_PATTERN = re.compile(
    r"\b(?:(?:ja){2,}j?|(?:je){2,}j?|(?:ji){2,}j?|(?:js){2,}j?|(?:ha){2,}h?)\b",
    re.IGNORECASE,
)
SENTENCE_BREAK_PATTERN = re.compile(r"[.!?\n]")

# letras que en español forman dígrafo o grupo doble legítimo (ll, rr, cc)
DOUBLE_LETTER_CONSONANTS = frozenset("lrc")


@dataclass(frozen=True)
class NormalizationChange:
    """Una transformación aplicada durante la normalización."""

    original: str
    replacement: str
    category: str


@dataclass(frozen=True)
class NormalizationResult:
    """Resultado inmutable: texto original, texto normalizado y cambios."""

    original_text: str
    normalized_text: str
    changes: tuple[NormalizationChange, ...]

    @property
    def was_modified(self) -> bool:
        return self.original_text != self.normalized_text


class TextNormalizer:
    """Normaliza abreviaturas, expresiones y formato del español informal."""

    def __init__(self) -> None:
        self._phrase_patterns = tuple(
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in INFORMAL_PHRASES
        )

    def normalize(self, text: str) -> NormalizationResult:
        """Devuelve una versión normalizada de ``text`` sin modificar el original.

        Raises:
            TypeError: si ``text`` no es una cadena.
        """
        if not isinstance(text, str):
            raise TypeError("el texto a normalizar debe ser una cadena")

        changes: list[NormalizationChange] = []
        pieces = []
        for chunk, is_protected in self._split_protected(text):
            pieces.append(chunk if is_protected else self._normalize_chunk(chunk, changes))

        normalized = self._normalize_whitespace("".join(pieces))
        return NormalizationResult(text, normalized, tuple(changes))

    # pipeline

    def _normalize_chunk(self, chunk: str, changes: list[NormalizationChange]) -> str:
        """Aplica los pasos de normalización a un fragmento no protegido."""
        chunk = self._normalize_shouting(chunk, changes)
        chunk = self._reduce_repeated_letters(chunk, changes)
        chunk = self._normalize_laughter(chunk, changes)
        chunk = self._expand_phrases(chunk, changes)
        chunk = self._expand_words(chunk, changes)
        return self._normalize_punctuation(chunk, changes)

    @staticmethod
    def _split_protected(text: str) -> list[tuple[str, bool]]:
        """Separa el texto en fragmentos normales y protegidos (urls, menciones...)."""
        parts: list[tuple[str, bool]] = []
        cursor = 0
        for match in PROTECTED_PATTERN.finditer(text):
            if match.start() > cursor:
                parts.append((text[cursor:match.start()], False))
            parts.append((match.group(), True))
            cursor = match.end()
        if cursor < len(text):
            parts.append((text[cursor:], False))
        return parts

    def _normalize_shouting(self, text: str, changes: list[NormalizationChange]) -> str:
        """Pasa a minúscula las palabras escritas en mayúsculas por énfasis.

        Solo actúa sobre secuencias de dos o más palabras en mayúsculas o sobre
        palabras sueltas estiradas ("HOLAAAA"). Una palabra suelta en mayúsculas
        puede ser una sigla, así que se respeta, igual que las siglas conocidas.
        """
        words = list(LETTERS_ONLY_PATTERN.finditer(text))
        is_caps = [w.group().isupper() and len(w.group()) >= 2 for w in words]

        result = []
        cursor = 0
        for index, match in enumerate(words):
            word = match.group()
            if (
                not is_caps[index]
                or word in KNOWN_ACRONYMS
                or ROMAN_NUMERAL_PATTERN.match(word)
            ):
                continue
            in_run = (index > 0 and is_caps[index - 1]) or (
                index + 1 < len(words) and is_caps[index + 1]
            )
            is_stretched = REPEATED_LETTER_PATTERN.search(word) is not None
            if not (in_run or is_stretched):
                continue

            lowered = word.lower()
            if self._starts_sentence(text, match.start()):
                lowered = lowered.capitalize()
            result.append(text[cursor:match.start()])
            result.append(lowered)
            cursor = match.end()
            changes.append(NormalizationChange(word, lowered, CATEGORY_CASE))

        result.append(text[cursor:])
        return "".join(result)

    @staticmethod
    def _reduce_repeated_letters(text: str, changes: list[NormalizationChange]) -> str:
        """Reduce letras repetidas tres o más veces ("holaaa" → "hola").

        Al final de palabra se deja una sola letra; dentro de la palabra, l, r y c
        se dejan dobles para respetar ll, rr y cc ("carrrro" → "carro").
        No toca números romanos ni palabras de una sola letra repetida ("mmm").
        """

        def fix_word(word_match: re.Match) -> str:
            word = word_match.group()
            if len(set(word.lower())) == 1 or ROMAN_NUMERAL_PATTERN.match(word):
                return word

            def collapse(run: re.Match) -> str:
                letter = run.group(1)
                at_word_end = run.end() == len(word)
                if not at_word_end and letter.lower() in DOUBLE_LETTER_CONSONANTS:
                    return letter * 2
                return letter

            fixed = REPEATED_LETTER_PATTERN.sub(collapse, word)
            if fixed != word:
                changes.append(NormalizationChange(word, fixed, CATEGORY_REPETITION))
            return fixed

        return LETTERS_ONLY_PATTERN.sub(fix_word, text)

    @staticmethod
    def _normalize_laughter(text: str, changes: list[NormalizationChange]) -> str:
        """Unifica risas escritas ("jajajaja", "jsjsjs") en una forma estándar."""

        def replace(match: re.Match) -> str:
            laughter = match.group()
            if laughter.lower() != STANDARD_LAUGHTER:
                changes.append(
                    NormalizationChange(laughter, STANDARD_LAUGHTER, CATEGORY_LAUGHTER)
                )
            return STANDARD_LAUGHTER

        return LAUGHTER_PATTERN.sub(replace, text)

    def _expand_phrases(self, text: str, changes: list[NormalizationChange]) -> str:
        """Expande expresiones de varias palabras o con apóstrofo ("ta bien", "pa'")."""
        for pattern, replacement in self._phrase_patterns:

            def replace(match: re.Match, target: str = replacement) -> str:
                expanded = _match_case(match.group(), target)
                changes.append(
                    NormalizationChange(match.group(), expanded, CATEGORY_EXPRESSION)
                )
                return expanded

            text = pattern.sub(replace, text)
        return text

    def _expand_words(self, text: str, changes: list[NormalizationChange]) -> str:
        """Expande abreviaturas y corrige errores frecuentes palabra por palabra."""

        def replace(match: re.Match) -> str:
            word = match.group()
            key = word.lower()
            if key in ABBREVIATIONS:
                target, category = ABBREVIATIONS[key], CATEGORY_ABBREVIATION
            elif key in COMMON_ERRORS:
                target, category = COMMON_ERRORS[key], CATEGORY_SPELLING
            else:
                target = self._resolve_contextual(text, match.start(), match.end(), key)
                category = CATEGORY_ABBREVIATION
            if target is None:
                return word
            expanded = _match_case(word, target)
            changes.append(NormalizationChange(word, expanded, category))
            return expanded

        return WORD_PATTERN.sub(replace, text)

    def _resolve_contextual(self, text: str, start: int, end: int, key: str) -> str | None:
        """Resuelve las abreviaturas que dependen del contexto.

        Devuelve None cuando la palabra no es contextual o cuando el contexto
        no basta para decidir con seguridad.
        """
        if key in QUE_FORMS:
            return "qué" if self._is_interrogative_slot(text, start, end) else "que"

        if key in PORQUE_FORMS:
            if self._is_interrogative_slot(text, start, end, allow_indirect=True):
                return "por qué"
            return "porque"

        if key in DONDE_FORMS:
            if self._is_interrogative_slot(text, start, end, allow_indirect=True):
                return "dónde"
            return "donde"

        if key in PARA_FORMS:
            # "mi pa" puede ser "mi papá": ambiguo, se conserva
            if _previous_word(text, start) in POSSESSIVE_DETERMINERS:
                return None
            return "para"

        return None

    @staticmethod
    def _is_interrogative_slot(
        text: str, start: int, end: int, allow_indirect: bool = False
    ) -> bool:
        """Indica si la palabra en [start, end) funciona como interrogativo.

        Es interrogativo cuando está en una pregunta y abre la cláusula
        ("¿xq no viniste?", "¿y tú dnd estás?") o la cierra ("¿y eso xq?").
        Con ``allow_indirect`` también cuenta la pregunta indirecta
        ("no sé xq se fue").
        """
        if allow_indirect and _previous_word(text, start) in INDIRECT_QUESTION_CUES:
            return True

        previous_breaks = list(SENTENCE_BREAK_PATTERN.finditer(text, 0, start))
        sentence_start = previous_breaks[-1].end() if previous_breaks else 0
        next_break = SENTENCE_BREAK_PATTERN.search(text, end)
        terminator = next_break.group() if next_break else ""
        sentence_end = next_break.start() if next_break else len(text)

        before = text[sentence_start:start]
        if "¿" in before:
            before = before[before.rindex("¿") + 1:]
        elif terminator != "?":
            return False

        lead_words = [w.lower() for w in WORD_PATTERN.findall(before)]
        opens_clause = all(w in INTERROGATIVE_LEAD_WORDS for w in lead_words)
        closes_question = terminator == "?" and not WORD_PATTERN.search(
            text, end, sentence_end
        )
        return opens_clause or closes_question

    @staticmethod
    def _normalize_punctuation(text: str, changes: list[NormalizationChange]) -> str:
        """Reduce signos repetidos sin alterar la intención de la frase."""
        rules = (
            (re.compile(r"[!?]*\?[!?]*!+[!?]*|[!?]*![!?]*\?+[!?]*"), "?!"),
            (re.compile(r"!{2,}"), "!"),
            (re.compile(r"\?{2,}"), "?"),
            (re.compile(r"¡{2,}"), "¡"),
            (re.compile(r"¿{2,}"), "¿"),
            (re.compile(r"\.{4,}"), "..."),
            (re.compile(r",{2,}"), ","),
            (re.compile(r";{2,}"), ";"),
        )
        for pattern, replacement in rules:

            def replace(match: re.Match, target: str = replacement) -> str:
                if match.group() != target:
                    changes.append(
                        NormalizationChange(match.group(), target, CATEGORY_PUNCTUATION)
                    )
                return target

            text = pattern.sub(replace, text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Limpia espacios duplicados y espacios mal ubicados junto a signos."""
        text = re.sub(r"[ \t\f\v ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +([,.;:!?)\]])", r"\1", text)
        text = re.sub(r"([¿¡(\[]) +", r"\1", text)
        # coma pegada entre letras: "hola,como" → "hola, como" (no toca "3,5")
        text = re.sub(r"(?<=[^\W\d_]),(?=[^\W\d_])", ", ", text)
        return text.strip()

    @staticmethod
    def _starts_sentence(text: str, position: int) -> bool:
        """Indica si ``position`` es el inicio de una oración."""
        preceding = text[:position].rstrip(" \t¿¡\"'(")
        return not preceding or preceding[-1] in ".!?\n"


def _previous_word(text: str, position: int) -> str:
    """Devuelve en minúscula la palabra inmediatamente anterior a ``position``."""
    words = WORD_PATTERN.findall(text, 0, position)
    return words[-1].lower() if words else ""


def _match_case(source: str, target: str) -> str:
    """Copia la mayúscula inicial de ``source`` sobre ``target``."""
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target

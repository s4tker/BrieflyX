"""Léxico de lenguaje informal en español usado por TextNormalizer.

Cada diccionario mapea la forma informal (en minúscula) a su forma estándar.
Solo se incluyen formas sin ambigüedad razonable. Las que dependen del
contexto (q, xq, dnd, pa) se resuelven en TextNormalizer, y las que no se
pueden resolver con seguridad (k, x, d, ns, ta) no aparecen aquí a propósito
para que el texto se conserve tal cual.
"""

# abreviaturas de chat con una sola lectura posible
ABBREVIATIONS = {
    "bn": "bien",
    "bno": "bueno",
    "xfa": "por favor",
    "porfa": "por favor",
    "porfis": "por favor",
    "plis": "por favor",
    "pls": "por favor",
    "tmb": "también",
    "tb": "también",
    "tbn": "también",
    "ntp": "no te preocupes",
    "toy": "estoy",
    "tamos": "estamos",
    "xo": "pero",
    "tqm": "te quiero mucho",
    "tkm": "te quiero mucho",
    "msj": "mensaje",
    "msjs": "mensajes",
    "dsp": "después",
    "dps": "después",
    "mñn": "mañana",
    "mñna": "mañana",
    "vdd": "verdad",
    "grax": "gracias",
    "nd": "nada",
    "salu2": "saludos",
    "100pre": "siempre",
    "tons": "entonces",
    "kiero": "quiero",
    "aki": "aquí",
    "finde": "fin de semana",
    "bss": "besos",
}

# errores de escritura frecuentes cuya corrección no admite otra lectura
COMMON_ERRORS = {
    "osea": "o sea",
    "enserio": "en serio",
    "porfavor": "por favor",
    "asique": "así que",
    "talvez": "tal vez",
    "aveces": "a veces",
    "almenos": "al menos",
    "apesar": "a pesar",
    "derrepente": "de repente",
    "sinembargo": "sin embargo",
    "haiga": "haya",
    "nadien": "nadie",
    "tambien": "también",
    "despues": "después",
    "ademas": "además",
    "aqui": "aquí",
    "asi": "así",
}

# expresiones de varias palabras o con apóstrofo: (patrón regex, reemplazo)
INFORMAL_PHRASES = (
    (r"\bta\s+bien\b", "está bien"),
    (r"\bx\s+fa(?:vor)?\b", "por favor"),
    (r"\bpa['’](?=\W|$)", "para"),
    (r"\bpal\b", "para el"),
)

# formas que se resuelven según el contexto de la oración
QUE_FORMS = frozenset({"q", "ke"})
PORQUE_FORMS = frozenset({"xq", "pq", "porq", "xk"})
DONDE_FORMS = frozenset({"dnd"})
PARA_FORMS = frozenset({"pa"})

# palabras que pueden ir antes del interrogativo sin cambiar su función:
# "¿y tú dnd estás?", "¿a q hora?", "¿por q?"
INTERROGATIVE_LEAD_WORDS = frozenset({
    "y", "e", "o", "pero", "entonces", "pues", "oye", "bueno", "ok",
    "tú", "tu", "vos", "usted", "ustedes", "él", "ella", "ellos", "ellas",
    "a", "de", "con", "para", "por", "desde", "hasta", "hacia", "en",
})

# verbos o expresiones que introducen una pregunta indirecta: "no sé xq"
INDIRECT_QUESTION_CUES = frozenset({
    "sé", "sabe", "sabes", "sabemos", "saben", "saber",
    "entiendo", "entiendes", "pregunto", "preguntó",
    "dime", "explica", "explícame", "explicame", "idea",
})

# tras un posesivo, "pa" puede ser "papá": se deja sin tocar
POSSESSIVE_DETERMINERS = frozenset({"mi", "tu", "su", "mis", "tus", "sus"})

# siglas que no se pasan a minúscula aunque estén dentro de un texto en mayúsculas
KNOWN_ACRONYMS = frozenset({
    "ONU", "OMS", "UE", "EEUU", "EE", "UU", "NASA", "UNESCO", "UNICEF",
    "FIFA", "PIB", "IA", "IVA", "DNI", "ONG", "OTAN", "URL", "PDF", "API",
    "USB", "PC", "TV", "CEO", "FMI", "BCE", "UNAM", "IPN",
})

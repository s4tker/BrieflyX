"""Categorías emocionales de emojis usadas por EmojiProcessor.

La clasificación se hace en dos niveles:

1. EMOJI_CATEGORIES: emojis frecuentes con una interpretación revisada a mano.
2. CATEGORY_KEYWORDS: palabras clave sobre el nombre en inglés que da la
   biblioteca ``emoji`` (estable entre versiones), para clasificar el resto.

Si ningún nivel aplica, el emoji queda sin categoría emocional pero conserva
su descripción en español; no se inventa una emoción.
"""

LAUGHTER = "risa"
JOY = "alegría"
SADNESS = "tristeza"
CRYING = "llanto"
AFFECTION = "afecto"
ANGER = "enojo"
APPROVAL = "aprobación"
DISAPPROVAL = "desaprobación"
ENTHUSIASM = "entusiasmo"
SURPRISE = "sorpresa"
FEAR = "miedo"
WORRY = "preocupación"
GRATITUDE = "agradecimiento"
DOUBT = "duda"
DISGUST = "asco"
CELEBRATION = "celebración"
PLAYFULNESS = "picardía"

# claves sin selector de variación ni tono de piel (se quitan antes de buscar)
EMOJI_CATEGORIES = {
    "😂": LAUGHTER, "🤣": LAUGHTER, "😹": LAUGHTER, "😆": LAUGHTER, "😅": LAUGHTER,
    "😁": JOY, "😀": JOY, "😃": JOY, "😄": JOY, "😊": JOY, "🙂": JOY, "☺": JOY,
    "😌": JOY, "🥳": CELEBRATION, "🎉": CELEBRATION, "🎊": CELEBRATION,
    "😢": SADNESS, "😞": SADNESS, "😔": SADNESS, "🙁": SADNESS, "☹": SADNESS,
    "😿": SADNESS, "💔": SADNESS, "🥺": SADNESS,
    "😭": CRYING,
    "❤": AFFECTION, "🧡": AFFECTION, "💛": AFFECTION, "💚": AFFECTION, "💙": AFFECTION,
    "💜": AFFECTION, "🖤": AFFECTION, "🤍": AFFECTION, "🤎": AFFECTION, "💕": AFFECTION,
    "💖": AFFECTION, "💗": AFFECTION, "💘": AFFECTION, "💞": AFFECTION, "😍": AFFECTION,
    "🥰": AFFECTION, "😘": AFFECTION, "🤗": AFFECTION, "😻": AFFECTION,
    "😡": ANGER, "😠": ANGER, "🤬": ANGER, "👿": ANGER, "💢": ANGER, "😤": ANGER,
    "👍": APPROVAL, "👌": APPROVAL, "✅": APPROVAL, "👏": APPROVAL, "💯": APPROVAL,
    "🙌": APPROVAL, "✔": APPROVAL,
    "👎": DISAPPROVAL, "❌": DISAPPROVAL, "🙄": DISAPPROVAL,
    "🔥": ENTHUSIASM, "🚀": ENTHUSIASM, "⭐": ENTHUSIASM, "🌟": ENTHUSIASM,
    "✨": ENTHUSIASM, "💪": ENTHUSIASM, "🤩": ENTHUSIASM,
    "😮": SURPRISE, "😯": SURPRISE, "😲": SURPRISE, "😳": SURPRISE, "🤯": SURPRISE,
    "😱": FEAR, "😨": FEAR, "😰": FEAR, "🫣": FEAR,
    "😟": WORRY, "😕": WORRY, "😬": WORRY, "😥": WORRY,
    "🙏": GRATITUDE,
    "🤔": DOUBT, "🤨": DOUBT,
    "🤢": DISGUST, "🤮": DISGUST,
    "😈": PLAYFULNESS, "😏": PLAYFULNESS, "😉": PLAYFULNESS, "😜": PLAYFULNESS,
    "😝": PLAYFULNESS, "😛": PLAYFULNESS,
}

# (fragmento del nombre en inglés, categoría); el orden importa: gana el primero
CATEGORY_KEYWORDS = (
    ("tears_of_joy", LAUGHTER),
    ("rolling_on_the_floor", LAUGHTER),
    ("laugh", LAUGHTER),
    ("broken_heart", SADNESS),
    ("heart", AFFECTION),
    ("kiss", AFFECTION),
    ("loudly_crying", CRYING),
    ("crying", SADNESS),
    ("sad", SADNESS),
    ("disappointed", SADNESS),
    ("frown", SADNESS),
    ("angry", ANGER),
    ("pouting", ANGER),
    ("enraged", ANGER),
    ("thumbs_up", APPROVAL),
    ("ok_hand", APPROVAL),
    ("clapping", APPROVAL),
    ("thumbs_down", DISAPPROVAL),
    ("fearful", FEAR),
    ("screaming", FEAR),
    ("anxious", WORRY),
    ("worried", WORRY),
    ("astonished", SURPRISE),
    ("open_mouth", SURPRISE),
    ("party", CELEBRATION),
    ("confetti", CELEBRATION),
    ("thinking", DOUBT),
    ("nauseated", DISGUST),
    ("vomiting", DISGUST),
    # "fire" y "star" no van aquí: atraparían 🚒 o ✡️
    ("grin", JOY),
    ("smil", JOY),
    ("beaming", JOY),
)

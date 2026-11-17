# Brieflyx

**Intelligent Text Processing** — *Understand more. Read less.*

Brieflyx es una aplicación web de Procesamiento de Lenguaje Natural (PLN) que analiza textos extensos, genera resúmenes extractivos, identifica ideas principales y extrae palabras clave. Está pensada también para lenguaje informal: abreviaturas, expresiones coloquiales y emojis propios de WhatsApp, Discord y redes sociales.

## Objetivo

Ayudar a estudiantes, profesionales y cualquier usuario a comprender información extensa rápidamente, conservando el significado original del contenido. Todo el procesamiento es local: no usa servicios de IA externos ni API de pago.

## Estado

MVP v1.0 en desarrollo.

- **Fase 1** — estructura y entorno.
- **Fase 2** — normalización del lenguaje informal (`NLP/TextNormalizer.py`).
- **Fase 3** — procesamiento de emojis (`NLP/EmojiProcessor.py`).

## Tecnologías

| Uso | Biblioteca |
|---|---|
| Interfaz | Streamlit |
| Análisis lingüístico | spaCy |
| Herramientas textuales complementarias | NLTK |
| Ponderación TF-IDF | scikit-learn |
| Grafo de oraciones y PageRank (TextRank) | NetworkX |
| Emojis | emoji |
| Pruebas | pytest |

## Estructura

```text
Brieflyx/
├── app.py          # punto de entrada de Streamlit
├── NLP/            # módulos de procesamiento de lenguaje
├── Utils/          # utilidades compartidas
├── Tests/          # pruebas unitarias
├── requirements.txt
└── README.md
```

## Módulos

Cada módulo devuelve un resultado inmutable que conserva el texto original sin cambios.

### Normalización (`NLP/TextNormalizer.py`)

```python
from NLP.TextNormalizer import TextNormalizer

result = TextNormalizer().normalize("toy bn xfa, ¿xq no viniste?")
result.normalized_text  # "estoy bien por favor, ¿por qué no viniste?"
result.changes          # cada cambio con su categoría
```

### Emojis (`NLP/EmojiProcessor.py`)

```python
from NLP.EmojiProcessor import EmojiProcessor

result = EmojiProcessor().process("qué rabia 😡😡😡 pero llegué 👍")
result.processed_text       # "qué rabia enojo pero llegué aprobación"
result.text_without_emojis  # "qué rabia pero llegué"
result.category_counts      # {"enojo": 3, "aprobación": 1}
result.occurrences[0]       # emoji, descripción, categoría, repeticiones, posición
```

Los emojis se clasifican con un mapa revisado de emojis frecuentes y, para el resto, con palabras clave sobre el nombre que da la biblioteca `emoji`. Si un emoji no expresa una emoción, se usa su descripción en español ("🚗" → "coche"). Las repeticiones seguidas cuentan como una sola ocurrencia con su cantidad. Ambos módulos son independientes y pueden aplicarse en cualquier orden.

## Instalación

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

La aplicación queda disponible en http://localhost:8501.

## Pruebas

```bash
pytest
```

Los archivos de prueba siguen la convención PascalCase (`Tests/Test*.py`); `pytest.ini` le indica a pytest cómo descubrirlos.

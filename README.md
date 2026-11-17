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
- **Fase 4** — análisis lingüístico con spaCy (`NLP/TextAnalyzer.py`).
- **Fase 5** — resumen extractivo con TextRank (`NLP/Summarizer.py`).
- **Fase 6** — palabras clave (`NLP/KeywordExtractor.py`) e ideas principales (`NLP/KeyPoints.py`).

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

### Análisis lingüístico (`NLP/TextAnalyzer.py`)

```python
from NLP.TextAnalyzer import TextAnalyzer

analysis = TextAnalyzer().analyze("Los gatos corrían. ¿Vienes?\nno te preocupes")
analysis.sentence_texts        # ["Los gatos corrían.", "¿Vienes?", "no te preocupes"]
sentence = analysis.sentences[0]
sentence.content_lemmas        # ["gato", "correr"]
sentence.lemmatized_text       # "gato correr" (entrada para TF-IDF)
analysis.sentences[2].has_negation  # True
analysis.lemma_frequencies     # lemas con contenido y su frecuencia
```

Usa el modelo `es_core_news_sm` de spaCy. La segmentación es por reglas (signos de cierre, `...`, saltos de línea, `¿` y `¡` como apertura) y corrige cortes falsos tras abreviaturas como `Sra.` o `p.m.`. Las palabras vacías se excluyen del contenido; las negaciones se marcan aparte para no perderlas. Un solo `TextAnalysis` está pensado para compartirse entre los algoritmos y no procesar el texto dos veces.

### Resumen TextRank (`NLP/Summarizer.py`)

```python
from NLP.Summarizer import Summarizer

result = Summarizer().summarize(texto, sentence_count=4)  # o ratio=0.3
result.summary_text          # oraciones elegidas, en orden original
result.ranking               # todas las oraciones con su puntuación PageRank
result.reduction_percentage  # porcentaje de palabras eliminadas
```

1. `TextAnalyzer` segmenta y lematiza; las palabras vacías ya quedan fuera.
2. Cada oración se vectoriza con TF-IDF sobre sus lemas (scikit-learn).
3. La similitud coseno entre oraciones define las aristas de un grafo (NetworkX).
4. PageRank (α = 0,85) puntúa la centralidad de cada oración.
5. Se eligen las mejor puntuadas, descartando las redundantes (coseno ≥ 0,8 o texto idéntico), y se devuelven en su orden original.

Es extractivo: cada oración es un fragmento literal del texto, recuperado por su posición. Con una sola oración no se construye grafo; si ninguna oración comparte términos, todas puntúan igual y se prefiere la de más contenido. `summarize_analysis()` acepta un `TextAnalysis` ya calculado para no procesar el texto dos veces.

### Palabras clave (`NLP/KeywordExtractor.py`)

```python
from NLP.KeywordExtractor import KeywordExtractor

result = KeywordExtractor().extract(texto, top_n=8)
result.terms                 # ["cambio climático", "temperatura", ...]
result.keywords[0].score     # relevancia normalizada (la primera vale 1.0)
result.keywords[0].variants  # formas en que aparece en el texto
```

Candidatos: sustantivos, nombres propios y adjetivos, más frases de dos palabras que se repiten ("cambio climático", "energía renovable"). Cada oración es un documento TF-IDF y la relevancia es la suma de pesos (frecuencia × idf), de modo que un término presente en todas las oraciones pierde peso frente a uno que distingue partes del texto. Las variantes ("economía", "económico") se agrupan con el stemmer Snowball de NLTK, y una frase sustituye a las palabras que contiene.

### Ideas principales (`NLP/KeyPoints.py`)

```python
from NLP.KeyPoints import KeyPointIdentifier

result = KeyPointIdentifier().identify(texto, count=3)
result.texts                # oraciones originales, en orden
result.points[0].keywords   # conceptos que contiene esa idea
```

Puntuación = 0,6 · TextRank + 0,4 · cobertura de palabras clave (ambas normalizadas). Reutiliza la similitud, PageRank y la selección sin redundancia de `Summarizer`, con un umbral más estricto (0,25) para que cada idea aporte un concepto distinto. Prefiere oraciones de al menos 4 palabras. `identify_from_analysis()` acepta el análisis, las palabras clave y el ranking ya calculados para no repetir trabajo.

## Instalación

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` incluye el modelo de spaCy para español (`es_core_news_sm`), que se descarga desde GitHub durante la instalación.

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

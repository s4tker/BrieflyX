# Brieflyx

**Intelligent Text Processing** — *Understand more. Read less.*

Brieflyx es una aplicación web de Procesamiento de Lenguaje Natural (PLN) que analiza textos extensos, genera resúmenes extractivos, identifica ideas principales y extrae palabras clave. Está pensada también para lenguaje informal: abreviaturas, expresiones coloquiales y emojis propios de WhatsApp, Discord y redes sociales.

## Objetivo

Ayudar a estudiantes, profesionales y cualquier usuario a comprender información extensa rápidamente, conservando el significado original del contenido. Todo el procesamiento es local: no usa servicios de IA externos ni API de pago.

## Estado

MVP v1.0 en desarrollo.

- **Fase 1** — estructura y entorno.
- **Fase 2** — normalización del lenguaje informal (`NLP/TextNormalizer.py`).

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

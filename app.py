"""Punto de entrada de Brieflyx.

placeholder de la fase 1: solo confirma que streamlit arranca.
la interfaz real se construye en la fase 4.
"""

import streamlit as st

APP_NAME = "Brieflyx"
APP_DESCRIPTION = "Intelligent Text Processing"
APP_SLOGAN = "Understand more. Read less."


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="centered")
    st.title(APP_NAME)
    st.caption(f"{APP_DESCRIPTION} · {APP_SLOGAN}")
    st.write("Entorno listo. La interfaz se construirá en la fase 4.")


if __name__ == "__main__":
    main()

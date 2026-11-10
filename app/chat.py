"""The course chat: the pipeline built in the slides, showing what happens behind it.

Run from the project root:  uv run streamlit run app/chat.py
"""

import streamlit as st
from chat_ui.pipeline import ask
from chat_ui.resources import indexes
from chat_ui.sidebar import sidebar
from chat_ui.views import show_turn

st.set_page_config(page_title="Chiedi ai bandi FESR", page_icon="💬", layout="wide")
st.title("Chiedi ai bandi FESR")
indexes()  # load models and indexes now, not at the first question
st.caption(
    "La pipeline costruita nel corso. Ogni risposta mostra da dove viene. "
    "Le domande sono indipendenti: la chat non ricorda le precedenti."
)

settings, example = sidebar()
turns = st.session_state.setdefault("turns", [])
for turn in turns:
    show_turn(turn)

question = st.chat_input("Fai una domanda sui bandi…") or example
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Cerco nei documenti…", expanded=True)
        turns.append(ask(question, settings, status))
    st.rerun()  # redraw from the history: citations become links, sources appear

# Ottimizzazione di una pipeline RAG

Materiale del corso pratico (11/11/2026, 4 ore): costruire una pipeline RAG completa,
valutarla e ottimizzarla su documenti reali della pubblica amministrazione.

## Setup

Requisiti: [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -m "not slow and not llm"
```

Slide (Quarto + reveal.js, codice eseguito):

```bash
uv run quarto preview slides   # anteprima live nel browser
uv run quarto render slides    # genera slides/_site/
```

## Dati

Corpus: bandi del PR FESR 2021-2027 Emilia-Romagna
([fesr.regione.emilia-romagna.it](https://fesr.regione.emilia-romagna.it)), scaricati
tramite la REST API del sito. I file finiscono in `data/raw/` (non versionata).

```bash
uv run python scripts/download_faq.py        # FAQ dei bandi → set di valutazione
uv run python scripts/download_documents.py  # PDF dei bandi + infografiche → corpus
uv run python scripts/extract_pages.py       # testo di ogni pagina → data/processed/
uv run python scripts/ocr_pages.py gemini    # OCR delle pagine-immagine (anche tesseract, qwen3-vl)
```

Le FAQ non entrano mai nel corpus: contengono le risposte attese.

## La chat

La pipeline del corso in una chat, con tutto quello che succede dietro le quinte:
pezzi trovati, riordino, prompt inviato al modello, tempi, token e pagine citate.
Le leve nella barra laterale (bando, OCR, ricerca, reranking, pezzi, prompt) cambiano
la risposta dal vivo; le domande di esempio sono quelle del set di valutazione.

```bash
uv run streamlit run app/chat.py
```

Usa gli indici in `.cache/qdrant/` costruiti dalle slide (se mancano li costruisce,
lentamente). Qdrant locale apre una cartella in un solo processo: non eseguire le slide
con la chat aperta.

## Struttura

- `src/ragcourse/` — collante per LangChain, Qdrant, ranx e DeepEval: corpus, modelli
  intercambiabili (anche il giudice), filtri, valutazione, grafici
- `tests/` — test automatici
- `slides/` — slide Quarto con codice eseguito e risultati
- `app/` — la chat dimostrativa (Streamlit)

# Ottimizzazione di una pipeline RAG

Materiale del corso pratico (11/11/2026, 4 ore): costruire una pipeline RAG completa,
valutarla e ottimizzarla su documenti reali della pubblica amministrazione.

## Setup

Requisiti: [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -m "not slow and not llm"
```

Slide (Quarto + reveal.js, codice eseguito): una sola presentazione, `slides/corso.qmd`,
che include un file per capitolo da `slides/_capitoli/`. Durata: 4 ore, pause comprese.

```bash
uv run quarto render slides             # dalle uscite salvate in slides/_freeze/: niente modelli
uv run quarto render slides/corso.qmd   # riesegue il codice (con la cache di Jupyter)
```

Le uscite eseguite sono salvate in `slides/_freeze/` (versionata): CI e chi non ha GPU o
chiavi API rigenerano le slide senza eseguire nulla. Con la cache di Jupyter, se cambia
solo il testo il codice non riparte; se cambia una cella, si riesegue tutto (circa un'ora).

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
chunk trovati, riordino, prompt inviato al modello, tempi, token e pagine citate.
Le leve nella barra laterale (bando, OCR, ricerca, reranking, chunk, prompt) cambiano
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
- `slides/` — la presentazione (Quarto, tema in stile Metropolis), codice eseguito e risultati
- `app/` — la chat dimostrativa (Streamlit)

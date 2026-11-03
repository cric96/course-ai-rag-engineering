# Ottimizzazione di una pipeline RAG

Materiale del corso pratico (11/11/2026, 4 ore): costruire una pipeline RAG completa,
valutarla e ottimizzarla su documenti reali della pubblica amministrazione.

## Setup

Requisiti: [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -m "not slow and not llm"
```

## Struttura

- `src/ragcourse/` — componenti della pipeline (chunking, embedding, retrieval, valutazione)
- `tests/` — test automatici
- `slides/` — slide Quarto con codice eseguito e risultati

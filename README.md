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
Online: <https://gianluca.aguzzi.dev/course-ai-rag-engineering/>

```bash
uv run python scripts/preview_slides.py   # anteprima live, anche quando cambia un capitolo
cd slides && uv run quarto publish gh-pages --no-prompt --no-browser   # pubblica il sito
```

Il codice gira in locale (GPU e chiavi API) e su GitHub Pages va il sito già pronto:
GitHub non esegue nulla. Con la cache di Jupyter (`slides/.jupyter_cache/`), se cambia
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

## Esperimento: dai chunk alle pagine

Si cercano e riordinano i chunk come nel capitolo sul reranking; solo dopo la
selezione si possono sostituire i 10 chunk con le loro pagine complete, senza
duplicati. L'indice e le pagine selezionate restano identici: non è il confronto
fra embedding di chunk e di pagine del capitolo sul chunking.

```bash
uv run python scripts/compare_page_context.py
```

Stesse 50 FAQ del corso (`random_state=0`), stesso prompt e giudice, indice senza
OCR. Il report JSON contiene risposte, pagine, punteggi e motivazioni; SVG e PNG
mostrano le medie. Le chiamate passano dalle cache del corso. I file finiscono in
`.cache/page-context.*`; `--sample 5 --output .cache/smoke.json` esegue una prova
su un sottoinsieme, salvandola separatamente.

Per separare l'effetto del reranker da quello del generatore locale:

```bash
# Reranker locale, generatore Gemini.
uv run python scripts/compare_page_context.py --reranker bge --context pagine \
  --output .cache/page-context-bge-gemini.json

# Retrieval, reranker e generatore locali; solo il giudice resta Gemini.
uv run python scripts/compare_page_context.py --reranker bge --context pagine \
  --model local:Qwen/Qwen3-4B-Instruct-2507 --load-in-4bit \
  --output .cache/page-context-local.json
```

Il generatore locale usa Transformers tramite LangChain, NF4 a 4 bit e al massimo
1024 nuovi token, con decoding greedy; le risposte che raggiungono il limite
vengono rigenerate con un massimo di 2048 token. Scarica circa 8 GB al primo avvio; la
quantizzazione riduce la memoria GPU. Il confronto misura la distanza dalla
pipeline Gemini sul nostro campione, non dallo stato dell'arte generale.
La valutazione resta via API per mantenere lo stesso metro fra le pipeline;
nessun OCR prodotto da un modello remoto entra in questo esperimento.

```bash
uv run python scripts/plot_pipeline_comparison.py
```

Questo comando verifica che i tre report usino le stesse domande e lo stesso
giudice, controlla che i due generatori con reranker bge ricevano le stesse
pagine, e salva il confronto complessivo in `.cache/pipeline-comparison.*`.

Misura del 6 novembre 2026 sulle 50 FAQ del corso (correttezza media GEval, 0–1):

| Reranker | Generatore | Contesto | Correttezza |
|---|---|---|---|
| Gemini Flash | Gemini Flash | 10 chunk | 0,510 |
| Gemini Flash | Gemini Flash | pagine dei 10 chunk | 0,538 |
| bge | Gemini Flash | pagine dei 10 chunk | 0,508 |
| bge | Qwen3-4B-Instruct-2507, NF4 | pagine dei 10 chunk | 0,368 |

L'espansione aumenta il contesto medio da 8.533 a 20.614 caratteri; IC bootstrap
appaiato al 95% del guadagno: −0,022 / +0,080, quindi ancora incerto. Il divario
della pipeline locale dal riferimento Gemini con pagine è −0,170 (IC −0,284 /
−0,050). Nessuna risposta locale finale raggiunge il limite di 2048 token.
Il riepilogo con i punteggi per domanda e il grafico è salvato anche in
`slides/assets/experiment-pipeline-comparison.json` e `.svg`; il capitolo finale
include entrambi i confronti.

## Struttura

- `src/ragcourse/` — collante per LangChain, Qdrant, ranx e DeepEval: corpus, modelli
  intercambiabili (anche il giudice), filtri, valutazione, grafici
- `tests/` — test automatici
- `slides/` — la presentazione (Quarto, tema in stile Metropolis), codice eseguito e risultati
- `app/` — la chat dimostrativa (Streamlit)

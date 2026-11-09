# Ottimizzazione di una pipeline RAG

Materiale del corso pratico per progettare, valutare e ottimizzare passo dopo passo una pipeline RAG (*Retrieval-Augmented Generation*) su documenti reali della pubblica amministrazione.

- **Slide online**: <https://gianluca.aguzzi.dev/course-ai-rag-engineering/>

## Cosa offre il repository

- **Slide con codice eseguito** (`slides/`): il percorso completo dalla baseline fino a filtri sui metadati, astensione, chunking, modelli di embedding, ricerca ibrida, reranking, OCR e Visual RAG, con tutti gli esperimenti e i confronti.
- **Libreria modulare** (`src/ragcourse/`): componenti riutilizzabili per caricare il corpus, scambiare modelli locali e via API, gestire gli indici vettoriali e valutare retrieval e generazione.
- **Chat dimostrativa** (`app/chat.py`): applicazione per provare la pipeline dal vivo, modificare le scelte progettuali dalla barra laterale e ispezionare dietro le quinte chunk recuperati, riordino, prompt, tempi, token e pagine citate.
- **Set di valutazione** (`data/eval/`): domande reali estratte dalle FAQ dei bandi e domande visive, associate alle pagine di riferimento (*gold*) per misurare retrieval, generazione e astensione.

## Struttura del progetto

- `slides/` — presentazione Quarto (`slides/corso.qmd` e capitoli in `slides/_capitoli/`)
- `src/ragcourse/` — libreria Python del corso (corpus, modelli, indici, filtri, reranking, valutazione, grafici e cache)
- `app/chat.py` — chat interattiva in Streamlit
- `data/eval/` — set di valutazione versionato (`faq_questions.jsonl`, `image_questions.jsonl`; dettagli in [`data/eval/README.md`](data/eval/README.md))
- `scripts/` — script per scaricare e preparare i dati e per riprodurre gli esperimenti delle slide
- `tests/` — test automatici (`pytest`)

## Dati

Il corpus è composto dai bandi del PR FESR 2021-2027 della Regione Emilia-Romagna ([fesr.regione.emilia-romagna.it](https://fesr.regione.emilia-romagna.it)), scaricati tramite la REST API del sito in `data/raw/` (cartella non versionata).

```bash
uv run python scripts/download_faq.py        # FAQ dei bandi → set di valutazione
uv run python scripts/download_documents.py  # PDF dei bandi e infografiche → corpus
uv run python scripts/extract_pages.py       # testo di ogni pagina → data/processed/
uv run python scripts/ocr_pages.py gemini    # OCR delle pagine-immagine (anche tesseract, qwen3-vl)
```

Le FAQ non entrano mai nel corpus indicizzato perché contengono le risposte attese usate per la valutazione (vedi [`data/eval/README.md`](data/eval/README.md)).

## Setup e utilizzo

Requisiti: Python 3.12, [uv](https://docs.astral.sh/uv/) e, per i modelli via API, una chiave `GEMINI_API_KEY` nel file `.env`.

```bash
uv sync                                          # installa le dipendenze
uv run pytest -m "not slow and not llm"          # esegue i test veloci offline
uv run streamlit run app/chat.py                 # avvia la chat dimostrativa
uv run python scripts/preview_slides.py --fast   # anteprima rapida delle slide (usa i risultati salvati)
uv run python scripts/preview_slides.py          # anteprima completa con esecuzione del codice
```

> **Nota**: Qdrant in modalità locale blocca la cartella dell'indice in `.cache/qdrant/` per singolo processo; non eseguire il rendering delle slide mentre la chat è aperta.

## Tecnologie utilizzate

- **Orchestrazione RAG**
  - [LangChain](https://python.langchain.com/docs/introduction/) — astrazioni standard per documenti, *text splitter*, modelli di embedding e di chat, catene LCEL
- **Database vettoriale e ricerca**
  - [Qdrant](https://qdrant.tech/documentation/) — database vettoriale per ricerca densa, ibrida e filtri sui metadati
  - [FastEmbed](https://qdrant.github.io/fastembed/) — vettori sparsi BM25 per la ricerca ibrida
- **Modelli (Embedding, Reranking, LLM e VLM)**
  - [Google Gemini API](https://ai.google.dev/gemini-api/docs) — generazione, embedding, reranking e lettura multimodale via API
  - [Sentence Transformers](https://sbert.net/) e [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — esecuzione locale di modelli di embedding ([Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-large), [BGE-M3](https://huggingface.co/BAAI/bge-m3), [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)), reranker ([BGE-Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3), [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)) e modelli visione-linguaggio ([Qwen3-VL](https://huggingface.co/Qwen))
- **Valutazione**
  - [ranx](https://amenra.github.io/ranx/) — metriche di *Information Retrieval* (Hit@k, Recall@k, MRR), fusione RRF e test statistici
  - [DeepEval](https://deepeval.com/docs/getting-started) — valutazione della generazione tramite *LLM-as-a-Judge* (correttezza GEval, fedeltà, pertinenza, richiamo del contesto)
- **Lettura dei documenti e OCR**
  - [PyMuPDF](https://pymupdf.readthedocs.io/) — estrazione del testo e rendering delle pagine PDF
  - [Tesseract OCR](https://tesseract-ocr.github.io/) — riconoscimento ottico dei caratteri sulle pagine-immagine
- **Presentazione, interfaccia e tooling**
  - [Quarto](https://quarto.org/docs/presentations/revealjs/) — creazione delle slide reveal.js con codice Python eseguito
  - [Streamlit](https://docs.streamlit.io/) — interfaccia web per la chat dimostrativa
  - [uv](https://docs.astral.sh/uv/) — gestione dell'ambiente Python e delle dipendenze

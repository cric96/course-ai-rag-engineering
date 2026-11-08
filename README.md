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
uv run python scripts/preview_slides.py --fast   # testo attuale e risultati salvati, senza calcoli
cd slides && uv run quarto publish gh-pages --no-prompt --no-browser   # pubblica il sito
```

Il codice gira in locale (GPU e chiavi API) e su GitHub Pages va il sito già pronto:
GitHub non esegue nulla. Con la cache di Jupyter (`slides/.jupyter_cache/`), se cambia
solo il testo il codice non riparte; se cambia una cella, si riesegue tutto (circa un'ora).

Per lavorare sul testo, `--fast` evita del tutto l'avvio del kernel: riusa gli
output delle celle corrispondenti a un'esecuzione salvata e aggiorna i capitoli
dal vivo. Le celle nuove o modificate restano senza output e vengono segnalate
nel terminale. I risultati sono quelli salvati, anche se cambiano dati o moduli
Python: per aggiornarli, avviare la preview senza `--fast`.
Ogni avvio della preview rapida usa un progetto temporaneo distinto dentro
`.cache/slides-preview/`, con tema e immagini propri: due terminali non condividono
i file generati. La cartella della sessione viene rimossa alla chiusura.

La cache delle ricerche in `.cache/searches.sqlite` è trasferibile insieme alla
cartella del progetto: gli indici sono identificati senza il percorso assoluto.
Le vecchie chiavi vengono convertite automaticamente all'apertura della cache,
conservando i risultati. Dopo un aggiornamento del codice, riavviare la preview.

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

## Esperimento: associare le FAQ una pagina alla volta

Per confrontare le etichette originali con richieste che contengono una sola pagina
e tutte le FAQ del bando startup:

```bash
uv run python scripts/compare_pagewise_labels.py --limit 6  # prova iniziale
uv run python scripts/compare_pagewise_labels.py          # tutte le pagine
```

Il modello resta Argon. Ogni associazione contiene una citazione e distingue
supporto diretto e parziale. Le richieste sono salvate separatamente in
`data/interim/labels-pagewise/`: riprendere l'esecuzione riusa quelle completate.
Il report `.cache/pagewise-label-comparison.json` confronta le associazioni,
conservando consumi e costi stimati. Il set `data/eval/faq_questions.jsonl` rimane
invariato. Più associazioni non dimostrano maggiore accuratezza: il metodo originale
ne seleziona al massimo cinque per FAQ, mentre il confronto per pagina le raccoglie
tutte, anche quando offrono soltanto una parte della risposta.

Prova dell'8 novembre 2026: 239 pagine e 29 FAQ, stesso Argon. Le FAQ con almeno
un riferimento proposto passano da 22 a 26; tutte e quattro le aggiunte hanno
soltanto supporti parziali. Sono confermate 50 delle 54 associazioni precedenti,
mentre la nuova procedura ne propone 129 in totale. La scansione costa circa
2,41 USD al listino standard e usa 1,06 milioni di token di input complessivi,
contro circa 169 mila token di contenuto della richiesta originale. I risultati
e i casi controllati sono in `slides/assets/experiment-pagewise-labels.json`.
Il controllo testuale rileva anche un'associazione sospetta: per il software viene
citata una nota sul costo storico collegata alla voce impianti e macchinari.

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

## Esperimento: confrontare i generatori

Per confrontare due generatori con la configurazione iniziale del capitolo sulla
generazione (stesse 50 FAQ, e5-small, filtro per bando e 5 chunk):

```bash
uv run python scripts/compare_generators.py --model google_genai:gemini-4-argon \
  --output .cache/generator-comparison-argon.json
```

Il giudice resta Gemini Flash. Il report conserva contesti identici per i due
generatori, risposte, quattro metriche, motivazioni e intervallo bootstrap appaiato
della differenza di correttezza. `--sample 5` esegue una prova sullo stesso campione.
JSON, SVG e PNG vengono salvati senza dover rieseguire l'intera presentazione.
Il grafico confronta fedeltà e pertinenza sulle domande a cui rispondono entrambi;
correttezza e richiamo del contesto includono tutte le 50 FAQ.

Misura del 7 novembre 2026: correttezza media Flash **0,252**, Argon **0,356**;
astensioni **21 → 5**. Guadagno +0,104 (IC bootstrap appaiato al 95% +0,044 / +0,170).
Sulle 29 risposte comuni: fedeltà 0,744 / 0,759, pertinenza 0,980 / 0,877.
I risultati riassunti e il grafico sono in `slides/assets/experiment-generator-comparison.*`.

Per aggiungere un modello a pesi aperti allo stesso confronto:

```bash
uv run python scripts/compare_generators.py --model google_genai:gemma-4-31b-it \
  --output .cache/generator-comparison-gemma.json
uv run python scripts/plot_generator_comparison.py \
  --output slides/assets/experiment-generator-comparison-open
```

[Gemma 4 31B](https://ai.google.dev/gemma/docs/core/model_card_4) è eseguito tramite
API Google. Il grafico a tre generatori richiede
domande, contesti, prompt, giudice e baseline identici nei due report; fedeltà e
pertinenza vengono confrontate solo sulle domande a cui rispondono tutti e tre.

Gemma ottiene correttezza **0,328**, con 12 astensioni; guadagno rispetto a Flash
+0,076 (IC bootstrap appaiato al 95% +0,028 / +0,128). Sulle 29 risposte comuni:
fedeltà 0,731 e pertinenza 0,961. Il richiamo resta 0,106 per tutti, con 40 FAQ
su 50 a zero secondo il giudice. Il grafico a tre generatori è nella slide
«Stessi documenti, tre generatori»; i risultati riassunti sono in
`slides/assets/experiment-generator-comparison-open.json`.

Per rivalutare senza cache i 25 casi con richiamo zero ma pagina gold recuperata:

```bash
uv run python scripts/audit_context_recall.py
uv run python scripts/audit_context_recall.py --context gold-chunks \
  --output .cache/context-recall-audit-single-chunks.json
```

Il test usa gli stessi chunk e la stessa FAQ salvati nell'esperimento originale;
controlla che il contesto coincida prima di effettuare nuove chiamate al giudice
Flash. Il primo comando ripete la misura sui cinque chunk, il secondo valuta
separatamente i chunk della pagina gold già presenti tra quei cinque.
Nell'audit del 7 novembre, 24 casi restano a zero e uno passa a 0,5. Isolando i
46 chunk gold, due FAQ ottengono 1 su un singolo chunk ma 0 sul contesto completo:
la misura è sensibile al contesto e richiede verifica sui casi. Il riepilogo con
esempi è in `slides/assets/experiment-context-recall-audit.json`; i report completi
sono in `.cache/context-recall-audit*.json`. L'audit mirato resta separato dalle
medie originali del confronto a 50 FAQ.

Per misurare il richiamo sui contesti effettivi del capitolo «Quanti chunk al modello?»:

```bash
uv run python scripts/compare_context_recall_k.py
```

Usa le ricerche già salvate per 5, 10 e 20 chunk sulle stesse 50 FAQ con risposta
attesa, senza generare nuove risposte. Giudice e definizione della metrica restano
Flash e ContextualRecallMetric standard. Medie del 7 novembre 2026: **0,100 / 0,160 /
0,184**, con **41 / 36 / 36** casi a zero. Il riferimento usa la cache del capitolo
(`memory:e5-small-1000`), anziché l'indice su disco del confronto tra generatori.
Il riepilogo è in `slides/assets/experiment-context-recall-k.json`; contesti e
motivazioni per caso sono in `.cache/context-recall-k.json`.

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

Per aggiungere Argon anche alla pipeline finale, mantenendo reranker Flash e
contesti identici per entrambi i generatori:

```bash
uv run python scripts/compare_page_context.py --model google_genai:gemini-4-argon \
  --output .cache/page-context-argon.json
uv run python scripts/plot_pipeline_comparison.py --argon-report .cache/page-context-argon.json \
  --output .cache/pipeline-comparison-argon
```

Questo comando verifica che i tre report usino le stesse domande e lo stesso
giudice, controlla che i due generatori con reranker bge ricevano le stesse
pagine, e salva il confronto complessivo in `.cache/pipeline-comparison.*`.

Misure sulle 50 FAQ del corso (correttezza media GEval, 0–1): Flash e Qwen il
6 novembre 2026, Argon il 7 novembre. Stesso prompt e giudice Flash; contesti
identici tra Flash e Argon a parità di retrieval e reranker.

| Reranker | Generatore | Contesto | Correttezza |
|---|---|---|---|
| Gemini Flash | Gemini Flash | 10 chunk | 0,510 |
| Gemini Flash | Gemini Flash | pagine dei 10 chunk | 0,538 |
| Gemini Flash | Gemini 4 Argon | 10 chunk | 0,584 |
| Gemini Flash | Gemini 4 Argon | pagine dei 10 chunk | 0,592 |
| bge | Gemini Flash | pagine dei 10 chunk | 0,508 |
| bge | Qwen3-4B-Instruct-2507, NF4 | pagine dei 10 chunk | 0,368 |

L'espansione aumenta il contesto medio da 8.533 a 20.614 caratteri; IC bootstrap
appaiato al 95% del guadagno: −0,022 / +0,080, quindi ancora incerto. Il divario
della pipeline locale dal riferimento Gemini con pagine è −0,170 (IC −0,284 /
−0,050). Nessuna risposta locale finale raggiunge il limite di 2048 token.
Il riepilogo con i punteggi per domanda e il grafico è salvato anche in
`slides/assets/experiment-pipeline-comparison.json` e `.svg`; il capitolo finale
include entrambi i confronti.

Il confronto completo con Argon è in `slides/assets/experiment-pipeline-comparison-argon.*`.
Il guadagno rispetto a Flash è +0,074 sui chunk (IC 95% +0,002 / +0,152) e +0,054
sulle pagine (IC −0,006 / +0,122). Sulle pagine il vantaggio è ancora incerto.
Argon non si astiene su nessuna delle 50 FAQ in entrambe le varianti; Flash si
astiene 9 volte sui chunk e 7 sulle pagine. Il giudice resta un modello, non una
verifica umana indipendente.

## Struttura

- `src/ragcourse/` — collante per LangChain, Qdrant, ranx e DeepEval: corpus, modelli
  intercambiabili (anche il giudice), filtri, valutazione, grafici
- `tests/` — test automatici
- `slides/` — la presentazione (Quarto, tema in stile Metropolis), codice eseguito e risultati
- `app/` — la chat dimostrativa (Streamlit)

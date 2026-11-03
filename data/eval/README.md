# Set di valutazione

`faq_questions.jsonl`: una riga per FAQ dei bandi PR FESR 2021-2027 Emilia-Romagna
(fonte: [fesr.regione.emilia-romagna.it](https://fesr.regione.emilia-romagna.it), CC BY 4.0).

| campo | significato |
|-------|-------------|
| `id` | identificativo stabile (`faq-001`, ...) |
| `question`, `answer` | domanda e risposta ufficiale |
| `call`, `url` | bando e pagina FAQ di origine |
| `gold` | pagine del corpus che sostengono la risposta: `[{url, page}]` |
| `evidence` | dove sta la risposta: `text`, `table`, `image`, `none` (non presente nei documenti) |
| `source` | `faq` |

Le pagine `gold` sono state proposte da `gemini-4-argon` leggendo tutte le pagine del bando
(le pagine con poco testo come immagini): `scripts/label_faq.py`. Vanno validate a campione.

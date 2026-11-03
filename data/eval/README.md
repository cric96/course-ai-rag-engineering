# Set di valutazione

Due file con lo stesso formato:

- `faq_questions.jsonl`: una riga per FAQ dei bandi PR FESR 2021-2027 Emilia-Romagna
  (fonte: [fesr.regione.emilia-romagna.it](https://fesr.regione.emilia-romagna.it), CC BY 4.0).
- `image_questions.jsonl`: domande la cui risposta sta solo nell'immagine di una pagina
  (slide, infografiche, pagine senza strato di testo), generate da `gemini-4-argon`
  con `scripts/generate_image_questions.py`; la pagina di origine è la pagina `gold`.

| campo | significato |
|-------|-------------|
| `id` | identificativo stabile (`faq-001`, ...) |
| `question`, `answer` | domanda e risposta ufficiale |
| `call`, `url` | bando e pagina FAQ di origine |
| `gold` | pagine del corpus che sostengono la risposta: `[{url, page}]` |
| `evidence` | dove sta la risposta: `text`, `table`, `image`, `none` (non presente nei documenti) |
| `source` | `faq` oppure `image` |

## Come sono state costruite

Le pagine `gold` sono state proposte da `gemini-4-argon` leggendo tutte le pagine del bando
(le pagine con poco testo come immagini): `scripts/label_faq.py`.

> **Scorciatoia didattica.** In un progetto reale la verità di riferimento va costruita,
> o almeno validata, da persone che conoscono i bandi. Etichette e domande prodotte da un
> modello:
>
> - possono sbagliare (pagine mancanti, `none` errati, risposte lette male);
> - favoriscono le domande "facili" per un modello;
> - rendono la valutazione circolare se un modello simile fa anche da giudice.
>
> Per questo un campione va validato a mano, e il tasso di errore va dichiarato.

## Come si usano

- **Metriche di retrieval**: solo le domande con `gold` non vuoto.
- **Valutazione della generazione**: un campione di circa 50 domande con `gold`.
- **FAQ con `evidence: none`**: la risposta ufficiale non sta nei documenti (indicazioni
  operative, casi specifici, moduli e norme fuori dal corpus). Restano fuori dalle metriche e servono a un solo test:
  il sistema sa dire "non lo trovo nei documenti" invece di inventare?

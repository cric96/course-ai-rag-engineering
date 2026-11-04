"""OCR: read the pages that are only pictures, so that text search can find them."""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image

from ragcourse.evalset import page_image
from ragcourse.parsing import render_page
from ragcourse.paths import DATA

OCR_PROMPT = (
    "Trascrivi tutto il testo di questa pagina, nell'ordine di lettura. "
    "Le tabelle riga per riga, con le celle separate da ' | '. "
    "Non aggiungere commenti né descrizioni."
)
TESSDATA = DATA / "raw/tessdata"  # Tesseract's language models, downloaded once
TESSDATA_URL = (
    "https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata"
)

# A reader turns one page record (path, page, kind) into its text.
Reader = Callable[[dict], str]


def page_picture(page: dict, dpi: int = 150) -> Image.Image:
    """The page as a picture: image files as they are, PDF pages rendered."""
    if page["kind"] == "image":
        return Image.open(page["path"]).convert("RGB")
    png = render_page(Path(page["path"]), page["page"], dpi=dpi)
    return Image.open(BytesIO(png)).convert("RGB")


def tesseract(page: dict, lang: str = "ita") -> str:
    """Classic OCR: finds letters in the picture, knows nothing about the content."""
    import pytesseract  # needs the tesseract program: models group only

    model = TESSDATA / f"{lang}.traineddata"
    if not model.exists():
        model.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(TESSDATA_URL.format(lang=lang), follow_redirects=True)
        model.write_bytes(response.raise_for_status().content)
    picture = page_picture(page, dpi=300)  # Tesseract wants about 300 dpi
    return pytesseract.image_to_string(
        picture, lang=lang, config=f"--tessdata-dir {TESSDATA}"
    ).strip()


def ocr_message(page: dict) -> HumanMessage:
    """The transcription request: instructions, then the page image."""
    return HumanMessage(
        content=[{"type": "text", "text": OCR_PROMPT}, page_image(page)]
    )


def vlm_reader(chat: BaseChatModel) -> Reader:
    """Any LangChain chat model that sees images becomes an OCR engine (e.g. Gemini)."""

    def read(page: dict) -> str:
        return chat.invoke([ocr_message(page)]).text.strip()

    return read


def local_vlm_reader(name: str = "Qwen/Qwen3-VL-2B-Instruct") -> Reader:
    """A small vision-language model on the local GPU, asked the same question."""
    import torch  # heavy: models group only
    from transformers import pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm = pipeline(
        "image-text-to-text", model=name, dtype=torch.bfloat16, device=device
    )

    def read(page: dict) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_picture(page, dpi=100)},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        out = vlm(text=messages, max_new_tokens=1024, return_full_text=False)
        return out[0]["generated_text"].strip()

    return read

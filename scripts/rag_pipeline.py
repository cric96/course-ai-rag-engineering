"""The course's best RAG pipeline as one LangChain chain, in a single script.

Hybrid search (bge-m3 + BM25) on text and OCR pages, optional call filter,
reranking, then a cited answer from Gemini.

    uv run python scripts/rag_pipeline.py "Quali spese sono ammissibili?"
    uv run python scripts/rag_pipeline.py "..." --call <call-slug> --reranker bge
"""

import argparse

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragcourse.chat import cited
from ragcourse.corpus import load_pages
from ragcourse.filters import metadata_filter
from ragcourse.generation import PROMPT, format_context
from ragcourse.index import extended_store, vector_store
from ragcourse.models import chat_model, embedding_model, sparse_model
from ragcourse.paths import DATA
from ragcourse.rerank import RerankRetriever, cross_encoder, llm_listwise

CANDIDATES = 30  # what the reranker reads
K = 10  # what the generator reads


def rag_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    """Question in; the chunks found and the cited answer out."""
    answer = (
        {
            "context": lambda x: format_context(x["chunks"]),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    # Retrieve once, keep the chunks next to the answer to show the sources.
    return RunnableParallel(chunks=retriever, question=RunnablePassthrough()).assign(
        answer=answer
    )


def build_store():
    """The hybrid index of decks 09-11: text chunks plus the pages read by OCR."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings, bm25 = embedding_model("bge-m3"), sparse_model("bm25")
    vector_store(  # built only if missing
        splitter.split_documents(load_pages()),
        embeddings,
        "bge-m3-1000-bm25",
        sparse=bm25,
    )
    ocr = splitter.split_documents(load_pages(DATA / "processed/ocr/gemini.jsonl"))
    return extended_store(
        "bge-m3-1000-bm25", ocr, embeddings, "bge-m3-1000-bm25-ocr-gemini", sparse=bm25
    )


def build_retriever(call: str | None, reranker: str) -> BaseRetriever:
    """Search (in one call, if given), then keep the K best chunks."""
    search = {"k": K if reranker == "nessuno" else CANDIDATES}
    if call:
        search["filter"] = metadata_filter(call=call)
    retriever = build_store().as_retriever(search_kwargs=search)
    if reranker == "nessuno":
        return retriever
    if reranker == "bge":
        scorer = cross_encoder("bge-reranker-v2-m3")
    else:
        scorer = llm_listwise(chat_model(thinking_level="low", timeout=60))
    return RerankRetriever(base=retriever, scorer=scorer, top_n=K)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--call", help="search only in this call (slug)")
    parser.add_argument(
        "--reranker", choices=["gemini", "bge", "nessuno"], default="gemini"
    )
    args = parser.parse_args()

    chain = rag_chain(build_retriever(args.call, args.reranker), chat_model())
    result = chain.invoke(args.question)

    print(result["answer"], "\n\nFonti:")
    for n in cited(result["answer"], result["chunks"]):
        meta = result["chunks"][n - 1].metadata
        print(f"[{n}] {meta['title']}, p. {meta['page']}")


if __name__ == "__main__":
    main()

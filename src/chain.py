"""Retrieval + generation logic."""
from typing import List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .config import settings

NOT_FOUND = "I couldn't find this in the uploaded documents."

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a document assistant. Answer ONLY from the context below. "
        f'If the answer is not in the context, reply exactly: "{NOT_FOUND}" '
        "Be concise and mention page numbers when helpful.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's latest question as a standalone question, using the "
        "chat history for context. Return only the rewritten question.",
    ),
    ("human", "History:\n{history}\n\nLatest question: {question}"),
])


def get_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. See .env.example.")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0,
    )


def format_context(docs: list) -> str:
    """Join retrieved chunks, labelled with file name and page number."""
    parts = []
    for d in docs:
        page = int(d.metadata.get("page", 0)) + 1
        parts.append(f"[{d.metadata.get('source')} | p.{page}]\n{d.page_content}")
    return "\n\n".join(parts)


def condense_question(llm, question: str, history: List[dict]) -> str:
    """Turn a follow-up question into a standalone one using recent history."""
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"history": recent, "question": question})


def ask(store, question: str, history: Optional[List[dict]] = None) -> Tuple[str, list]:
    """Answer a question from the indexed documents. Returns (answer, sources)."""
    llm = get_llm()
    standalone = condense_question(llm, question, history) if history else question
    docs = store.max_marginal_relevance_search(
        standalone, k=settings.top_k, fetch_k=settings.fetch_k
    )
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(docs), "question": question})
    return answer, docs

"""Answers a natural-language question about a codebase, grounded in retrieved chunks.

Retrieval and generation are kept as separate steps (`retrieve` then
`generate_answer`) rather than one combined function, so each is
independently testable: retrieval quality can be checked without paying for
an LLM call, and prompt changes can be tested without re-embedding anything.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from codesage.models import Answer, RetrievedChunk
from codesage.vector_store import VectorStore

_SYSTEM_PROMPT = (
    "You are a codebase assistant. Answer the user's question using only the "
    "provided code excerpts. If the excerpts don't contain the answer, say so "
    "plainly instead of guessing."
)


def _build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks and the question into a single user turn."""
    excerpts = "\n\n".join(
        f"# {r.chunk.path} ({r.chunk.name}, lines {r.chunk.start_line}-{r.chunk.end_line})\n"
        f"{r.chunk.content}"
        for r in retrieved
    )
    return f"Code excerpts:\n\n{excerpts}\n\nQuestion: {question}"


async def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    *,
    client: AsyncAnthropic,
    model: str,
) -> Answer:
    """Calls the Anthropic API to answer `question`, grounded in `retrieved` chunks."""
    if not retrieved:
        return Answer(text="No relevant code was found in the index for this question.", sources=[])

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(question, retrieved)}],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return Answer(text="\n".join(text_blocks), sources=[r.chunk for r in retrieved])


def retrieve(store: VectorStore, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
    """Thin wrapper kept so call sites depend on `qa`, not on `VectorStore` directly."""
    return store.search(query_vector, top_k=top_k)

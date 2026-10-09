"""Domain models for CodeSage.

Kept in one module because these types are shared across every stage of the
pipeline (chunking -> embedding -> retrieval -> answering) and importing them
from a single source avoids circular imports between stage modules.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkKind(StrEnum):
    """What a chunk of source code represents."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"


class CodeChunk(BaseModel):
    """A single retrievable unit of source code.

    Chunks are function- or class-scoped rather than fixed-length windows:
    a fixed-length window can split a function mid-body, which both wastes
    embedding budget on a fragment and gives the LLM an incomplete unit of
    logic to reason about. Splitting on `ast` boundaries costs a bit more
    up-front parsing complexity but keeps each chunk semantically whole.
    """

    path: str
    kind: ChunkKind
    name: str
    start_line: int
    end_line: int
    content: str

    @property
    def chunk_id(self) -> str:
        """Stable identifier used as the vector store key."""
        return f"{self.path}:{self.start_line}-{self.end_line}"


class EmbeddedChunk(BaseModel):
    """A chunk paired with its embedding vector."""

    chunk: CodeChunk
    vector: list[float]

    model_config = {"arbitrary_types_allowed": True}


INDEX_FORMAT_VERSION = 2


class IndexMeta(BaseModel):
    """What the index was built from, so `index` can tell what is stale without re-embedding."""

    format_version: int = INDEX_FORMAT_VERSION
    embedding_model: str = ""
    chunker_version: int = 0
    # rel path -> SHA-256 of the raw file bytes
    files: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned from a similarity search, with its score."""

    chunk: CodeChunk
    score: float = Field(ge=-1.0, le=1.0)


class Answer(BaseModel):
    """The final response returned to the user."""

    text: str
    sources: list[CodeChunk]

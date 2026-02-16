"""Retrieval configuration schema."""

from pydantic import BaseModel


class RetrievalConfig(BaseModel):
    table_retrieval_size: int
    table_column_retrieval_size: int
    table_retrieval_score_threshold: float | None

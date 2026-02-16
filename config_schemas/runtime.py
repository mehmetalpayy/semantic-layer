"""Runtime configuration schema."""

from pydantic import BaseModel


class RuntimeConfig(BaseModel):
    project_id: str
    retrieval_query: str
    enable_column_pruning: bool
    unique_values_concurrency: int

"""Summary configuration schemas."""

from pydantic import BaseModel


class SummaryLongTermConfig(BaseModel):
    embedding_dim: int
    model: str
    embeddings_azure_endpoint: str
    embeddings_deployment_name: str
    embeddings_api_version: str


class SummaryConfig(BaseModel):
    long_term: SummaryLongTermConfig

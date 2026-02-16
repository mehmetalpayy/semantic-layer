"""Top-level application configuration schema."""

from pydantic import BaseModel

from .database import DatabaseConfig
from .infra import InfraConfig
from .retrieval import RetrievalConfig
from .runtime import RuntimeConfig
from .summary import SummaryConfig


class AppConfig(BaseModel):
    database: DatabaseConfig
    infra: InfraConfig
    retrieval: RetrievalConfig
    runtime: RuntimeConfig
    summary: SummaryConfig

"""Database configuration schemas."""

from pydantic import BaseModel


class PostgresConnectionConfig(BaseModel):
    host: str
    port: int
    user: str
    db: str
    min_pool_size: int
    max_pool_size: int
    timeout: float


class DatabaseConfig(BaseModel):
    local_db: PostgresConnectionConfig
    remote_db: PostgresConnectionConfig

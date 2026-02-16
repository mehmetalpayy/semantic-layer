"""Environment-based secret configuration using Pydantic settings."""

from pydantic_settings import BaseSettings


class SecretSettings(BaseSettings):
    """Loads secret values from the .env file."""

    AZURE_OPENAI_API_KEY: str
    EMBEDDINGS_AZURE_OPENAI_API_KEY: str
    LOCAL_POSTGRES_PASSWORD: str
    REMOTE_POSTGRES_PASSWORD: str

    class Config:
        """Pydantic settings configuration for reading from the .env file."""

        env_file = ".env"
        env_file_encoding = "utf-8"


secrets = SecretSettings()

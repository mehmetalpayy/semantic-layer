"""Infrastructure configuration schemas."""

from pydantic import BaseModel


class AzureConfig(BaseModel):
    deployment_name: str
    endpoint: str
    api_version: str
    context_window_size: int


class InfraConfig(BaseModel):
    azure: AzureConfig

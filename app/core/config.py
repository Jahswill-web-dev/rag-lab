from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Lab API"
    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    database_url: str
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1)
    answer_model: str = "gpt-4.1-mini"
    answer_max_output_tokens: int = Field(default=400, ge=1, le=4_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()

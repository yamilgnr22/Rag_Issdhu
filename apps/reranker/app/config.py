from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    reranker_host: str = Field(default="0.0.0.0")
    reranker_port: int = Field(default=7997)
    reranker_default_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_device: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    reranker_max_batch_size: int = Field(default=64)
    reranker_trust_remote_code: bool = Field(default=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()

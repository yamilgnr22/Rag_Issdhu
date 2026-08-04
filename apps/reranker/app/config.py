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
    # bge-reranker-v2-m3 admite 8192 tokens y sentence-transformers usa ese
    # limite si no se fija otro: la atencion crece con el cuadrado de la longitud
    # y la VRAM se llena. Medido con 32 pasajes largos: 3,24s sin limite frente a
    # 0,95s con 512, con scores equivalentes. 512 es ademas lo que suelen usar
    # los cross-encoders de reranking.
    reranker_max_length: int = Field(default=512)
    reranker_trust_remote_code: bool = Field(default=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()

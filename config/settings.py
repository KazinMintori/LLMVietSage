"""Model inference settings. Run from the project root to load its .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:4b"
    ollama_keep_alive: str = "1h"
    ollama_warmup_enabled: bool = True

    # Context capacity for this model; actual VRAM savings require measurement.
    ollama_num_ctx: int = 4096
    # Short factual turns do not need the full KV cache. Long requests keep
    # ``ollama_num_ctx`` so itinerary answers retain enough room.
    ollama_short_num_ctx: int = 2048
    ollama_temperature: float = 0.1
    ollama_num_predict: int = 1024
    # Long-answer requests may use this larger budget only when the estimated
    # input still leaves room inside the configured context window.
    ollama_long_num_predict: int = 1536
    ollama_context_reserve_tokens: int = 128
    ollama_think: bool = False

    # Shared HTTP pool for repeated inference requests.
    api_timeout_seconds: float = 300.0
    api_connect_timeout_seconds: float = 10.0
    api_max_connections: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

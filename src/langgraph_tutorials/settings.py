from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# base classes
class LangFuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    langfuse_secret_key: SecretStr
    langfuse_public_key: SecretStr
    langfuse_base_url: str


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="ollama_")
    api_key: SecretStr = SecretStr(secret_value="ollama")
    base_url: str = "http://localhost:11434/v1"
    model_name: str = "deepseek-r1:1.5b-qwen-distill-q8_0"
    reasoning_effort: Literal["high"] = "high"


class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="groq_")
    api_key: SecretStr
    base_url: str = "https://api.groq.com/openai/v1"
    model_name: str = "openai/gpt-oss-120b"
    reasoning_effort: Literal["minimal"] = "minimal"


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="openai_")
    api_key: SecretStr
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-5.4-mini"
    reasoning_effort: Literal["none"] = "none"


class LLMProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="llm_")
    provider: Literal["openai", "groq", "ollama"] = "openai"
    concurrency: int | None = None


class WebSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    tavily_api_key: SecretStr


class LLMSettings:
    api_key: SecretStr
    base_url: str
    model_name: str
    reasoning_effort: str

    def __init__(self, settings: OllamaSettings | GroqSettings | OpenAISettings) -> None:
        self.api_key = settings.api_key
        self.base_url = settings.base_url
        self.model_name = settings.model_name
        self.reasoning_effort = settings.reasoning_effort

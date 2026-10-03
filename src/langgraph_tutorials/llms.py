import threading
from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .settings import GroqSettings, LLMProviderSettings, LLMSettings, OllamaSettings, OpenAISettings

llm_provider_settings = LLMProviderSettings()
provider_settings: OllamaSettings | GroqSettings | OpenAISettings
if llm_provider_settings.provider == "groq":
    provider_settings = GroqSettings()
elif llm_provider_settings.provider == "ollama":
    provider_settings = OllamaSettings()
else:
    provider_settings = OpenAISettings()

llm_settings = LLMSettings(settings=provider_settings)

default_llm = ChatOpenAI(
    api_key=llm_settings.api_key,
    base_url=llm_settings.base_url,
    model=llm_settings.model_name,
    reasoning_effort=llm_settings.reasoning_effort,
    use_responses_api=True,
)

# Caps how many LLM calls (across parallel graph branches, e.g. concurrent interviews) may be
# in flight at once. Local providers like Ollama can otherwise get hit with more concurrent
# requests than they can serve, which triggers idle-unload races between requests.
_llm_semaphore = threading.Semaphore(llm_provider_settings.concurrency) if llm_provider_settings.concurrency else None


def llm_invoke(runnable: Runnable, input: Any, **kwargs: Any) -> Any:
    if _llm_semaphore is None:
        return runnable.invoke(input, **kwargs)
    with _llm_semaphore:
        return runnable.invoke(input, **kwargs)

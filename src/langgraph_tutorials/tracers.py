from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from .settings import LangFuseSettings

langfuse_settings = LangFuseSettings()
langfuse = Langfuse(
    public_key=langfuse_settings.langfuse_public_key.get_secret_value(),
    secret_key=langfuse_settings.langfuse_secret_key.get_secret_value(),
    base_url=langfuse_settings.langfuse_base_url,
)

langfuse_handler = CallbackHandler()

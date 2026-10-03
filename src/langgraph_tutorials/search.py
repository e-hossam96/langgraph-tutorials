from tavily import TavilyClient

from .settings import WebSearchSettings

web_search_settings = WebSearchSettings()
web_search_client = TavilyClient(api_key=web_search_settings.tavily_api_key.get_secret_value())

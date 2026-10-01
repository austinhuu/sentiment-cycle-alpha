"""Public-safe API configuration. Set environment variables; never hardcode secrets."""
import os

OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENAI_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

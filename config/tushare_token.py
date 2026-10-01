"""Public-safe Tushare configuration. Set TUSHARE_TOKEN as an environment variable."""
import os

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

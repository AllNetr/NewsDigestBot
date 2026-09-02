"""
Конфигурация проекта.

Секретные значения (токены, ключи API) берутся из переменных окружения
через python-dotenv — файл .env НЕ должен попадать в git.
Скопируйте .env.example в .env и заполните своими данными.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# RSS-ленты для сбора новостей
# ---------------------------------------------------------------------------
# Можно добавлять/убирать источники по своему усмотрению.
RSS_FEEDS = [
    "https://tengrinews.kz/rss/",              # Tengrinews (Казахстан)
    "https://www.zakon.kz/rss/",                # Zakon.kz (Казахстан)
    "https://24.kz/ru/rss",                     # 24.kz (Казахстан)
    "http://feeds.bbci.co.uk/news/world/rss.xml",  # BBC World News
    "https://feeds.reuters.com/reuters/topNews",   # Reuters Top News
]

# Сколько новостей брать с каждой ленты за один запуск
MAX_ITEMS_PER_FEED = 5

# ---------------------------------------------------------------------------
# Настройки ИИ (DeepSeek или OpenRouter — совместимы с OpenAI Chat API)
# ---------------------------------------------------------------------------
# Провайдер выбирается через переменную окружения AI_PROVIDER: "deepseek" или "openrouter"
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek").lower()

if AI_PROVIDER == "openrouter":
    AI_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    AI_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    AI_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
else:  # deepseek по умолчанию
    AI_API_URL = "https://api.deepseek.com/chat/completions"
    AI_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    AI_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

REQUEST_TIMEOUT = 30  # секунд, для запросов к ИИ и Telegram

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Расписание (24-часовой формат, локальное время сервера)
# ---------------------------------------------------------------------------
MORNING_TIME = os.getenv("MORNING_TIME", "08:00")
EVENING_TIME = os.getenv("EVENING_TIME", "20:00")

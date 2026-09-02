"""
Сборщик новостей с ИИ-пересказом и отправкой в Telegram.

Логика работы:
1. collect_news()   — скачивает и парсит RSS-ленты (feedparser).
2. summarize_news() — отправляет текст новости в LLM (DeepSeek / OpenRouter)
                       и получает краткий пересказ в 2 предложения.
3. send_digest()    — формирует дайджест и отправляет его в Telegram.
4. run_job()        — объединяет шаги 1-3, отлавливая любые ошибки,
                       чтобы падение одного источника не остановило программу.
5. main()           — планирует запуск run_job() дважды в день (schedule).

Настройки (токены, список RSS-лент и т.д.) вынесены в config.py и .env,
чтобы не хранить секреты в коде.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import feedparser
import requests
import schedule

import config

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("news_digest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    title: str
    link: str
    summary_raw: str          # исходное описание из RSS
    source: str
    ai_summary: Optional[str] = None   # пересказ от ИИ (заполняется позже)


# ---------------------------------------------------------------------------
# 1. Сбор новостей
# ---------------------------------------------------------------------------
def collect_news() -> List[NewsItem]:
    """
    Проходит по всем RSS-лентам из config.RSS_FEEDS и собирает последние
    новости (не больше config.MAX_ITEMS_PER_FEED с каждой ленты).
    Ошибка одной ленты не прерывает сбор остальных.
    """
    items: List[NewsItem] = []

    for feed_url in config.RSS_FEEDS:
        try:
            logger.info("Парсинг ленты: %s", feed_url)

            # feedparser.parse() сам по себе не поддерживает timeout и может
            # зависнуть, если сайт не отвечает. Поэтому сначала скачиваем
            # содержимое через requests (с таймаутом), а затем отдаём
            # готовые байты в feedparser — так зависание исключено.
            response = requests.get(
                feed_url,
                timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0 (NewsDigestBot/1.0)"},
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            if feed.bozo and not feed.entries:
                # bozo=1 значит ошибка парсинга; если при этом нет записей — пропускаем
                raise ValueError(f"Не удалось разобрать ленту: {feed.bozo_exception}")

            source_name = feed.feed.get("title", feed_url)

            for entry in feed.entries[: config.MAX_ITEMS_PER_FEED]:
                title = entry.get("title", "Без заголовка")
                link = entry.get("link", "")
                raw_summary = entry.get("summary", entry.get("description", ""))

                items.append(
                    NewsItem(
                        title=title,
                        link=link,
                        summary_raw=raw_summary,
                        source=source_name,
                    )
                )

        except requests.exceptions.RequestException as exc:
            logger.error("Не удалось скачать ленту %s: %s", feed_url, exc)
            continue  # переходим к следующей ленте, не прерывая работу
        except Exception as exc:  # noqa: BLE001 — намеренно широкий перехват
            logger.error("Ошибка при обработке ленты %s: %s", feed_url, exc)
            continue  # переходим к следующей ленте, не прерывая работу

    logger.info("Всего собрано новостей: %d", len(items))
    return items


# ---------------------------------------------------------------------------
# 2. ИИ-пересказ
# ---------------------------------------------------------------------------
def _strip_html(text: str) -> str:
    """Простая очистка текста от HTML-тегов без внешних зависимостей."""
    import re

    clean = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", clean).strip()


def summarize_text(title: str, raw_text: str) -> Optional[str]:
    """
    Отправляет запрос в LLM (DeepSeek или OpenRouter — выбирается в config.py)
    и возвращает краткий пересказ в 2 предложения.
    В случае любой ошибки (сеть, лимиты, пустой ответ) возвращает None,
    и в дайджест уйдёт оригинальное описание новости.
    """
    clean_text = _strip_html(raw_text)[:3000]  # ограничиваем длину на всякий случай

    prompt = (
        f"Заголовок новости: {title}\n\n"
        f"Текст новости: {clean_text}\n\n"
        "Кратко перескажи эту новость в 2 предложениях на русском языке. "
        "Пиши только сам пересказ, без вступлений."
    )

    headers = {
        "Authorization": f"Bearer {config.AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
    }

    try:
        response = requests.post(
            config.AI_API_URL,
            headers=headers,
            json=payload,
            timeout=config.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        summary = data["choices"][0]["message"]["content"].strip()
        return summary if summary else None

    except requests.exceptions.RequestException as exc:
        logger.warning("Сетевая ошибка при обращении к ИИ: %s", exc)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Некорректный ответ от ИИ: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Неожиданная ошибка при пересказе: %s", exc)

    return None


def summarize_news(items: List[NewsItem]) -> List[NewsItem]:
    """Проставляет ai_summary для каждой новости, не падая при ошибках ИИ."""
    for item in items:
        summary = summarize_text(item.title, item.summary_raw)
        if summary:
            item.ai_summary = summary
        else:
            # fallback — используем очищенное исходное описание
            fallback = _strip_html(item.summary_raw)
            item.ai_summary = (fallback[:300] + "…") if len(fallback) > 300 else fallback
            logger.info("Используется fallback-пересказ для: %s", item.title)
    return items


# ---------------------------------------------------------------------------
# 3. Отправка в Telegram
# ---------------------------------------------------------------------------
def _format_digest(items: List[NewsItem]) -> List[str]:
    """
    Формирует список текстовых блоков для отправки.
    Отправляется обычным текстом (без Markdown-разметки Telegram), потому что
    заголовки и описания новостей могут содержать символы вроде *, _, [, ], (, )
    которые Telegram трактует как разметку — при их некорректном сочетании
    Telegram отклоняет всё сообщение целиком с ошибкой 400 Bad Request.
    Обычный текст гарантированно доставляется независимо от содержимого новости.

    Telegram ограничивает сообщение 4096 символами, поэтому дайджест
    разбивается на несколько сообщений при необходимости.
    """
    messages = []
    current = "🗞 Новостной дайджест\n\n"

    for item in items:
        block = (
            f"{item.title}\n"
            f"Источник: {item.source}\n"
            f"{item.ai_summary}\n"
            f"Читать полностью: {item.link}\n\n"
        )
        if len(current) + len(block) > 4000:
            messages.append(current)
            current = ""
        current += block

    if current.strip():
        messages.append(current)

    return messages


def send_to_telegram(text: str) -> bool:
    """Отправляет одно текстовое сообщение в Telegram через Bot API."""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        # Без parse_mode: символы вроде *, _, [, ] в новостях иначе могут
        # сломать Markdown-разметку и Telegram отклонит всё сообщение (400).
        "disable_web_page_preview": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError as exc:
        # Логируем тело ответа Telegram — там обычно есть понятное описание причины (description)
        try:
            details = response.json()
        except ValueError:
            details = response.text
        logger.error("Ошибка при отправке в Telegram: %s | Ответ сервера: %s", exc, details)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Ошибка при отправке в Telegram: %s", exc)
        return False


def send_digest(items: List[NewsItem]) -> None:
    """Форматирует и отправляет дайджест по частям."""
    if not items:
        logger.info("Нет новостей для отправки — дайджест пропущен.")
        return

    messages = _format_digest(items)
    for i, msg in enumerate(messages, 1):
        ok = send_to_telegram(msg)
        status = "успешно" if ok else "с ошибкой"
        logger.info("Отправка части %d/%d %s.", i, len(messages), status)
        time.sleep(1)  # небольшая пауза, чтобы не упереться в лимиты Telegram


# ---------------------------------------------------------------------------
# 4. Основная задача (объединяет все шаги)
# ---------------------------------------------------------------------------
def run_job() -> None:
    """Полный цикл: собрать → переслать в ИИ → отправить дайджест."""
    logger.info("=== Запуск задачи сбора новостей ===")
    try:
        news = collect_news()
        news = summarize_news(news)
        send_digest(news)
    except Exception as exc:  # noqa: BLE001 — последний рубеж защиты
        logger.critical("Необработанная ошибка в run_job: %s", exc, exc_info=True)
    logger.info("=== Задача завершена ===")


# ---------------------------------------------------------------------------
# 5. Точка входа
# ---------------------------------------------------------------------------
# Есть два режима запуска — выберите один в зависимости от того, где хостите проект.
#
# РЕЖИМ A: локально / на своём сервере, процесс работает постоянно (schedule).
#          Используйте main_local(), если запускаете `python3 news_digest.py`
#          на компьютере или VPS, который не выключается.
#
# РЕЖИМ B: GitHub Actions / внешний cron.
#          Планировщик — GitHub, а скрипт просто выполняет run_job() один раз
#          за запуск и завершается. Это режим по умолчанию (см. блок ниже).


def main_local() -> None:
    """Режим для постоянно работающего процесса (ПК/VPS) со встроенным schedule."""
    logger.info("Планировщик запущен. Дайджест будет приходить в %s и %s.",
                config.MORNING_TIME, config.EVENING_TIME)

    schedule.every().day.at(config.MORNING_TIME).do(run_job)
    schedule.every().day.at(config.EVENING_TIME).do(run_job)

    while True:
        try:
            schedule.run_pending()
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка в цикле планировщика: %s", exc)
        time.sleep(30)


if __name__ == "__main__":
    # По умолчанию — режим одиночного запуска (для GitHub Actions/cron).
    # Расписание в этом случае задаётся снаружи (workflow-файл), а не внутри скрипта.
    #
    # Если хотите вернуть режим с бесконечным циклом (локальный ПК/VPS) —
    # закомментируйте run_job() ниже и раскомментируйте main_local().
    run_job()
    # main_local()

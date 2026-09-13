"""Общие утилиты проекта: настройка логирования и мелкие помощники.

Модуль намеренно не зависит от pandas/torch, чтобы его можно было использовать
на любом этапе пайплайна (EDA, обучение, оценка, API).
"""

from __future__ import annotations

import logging
import sys

#: Единый формат логов для всех скриптов проекта.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: Стандартное зерно воспроизводимости (используется на этапах обучения).
DEFAULT_SEED = 42


def setup_logging(level: str = "INFO") -> None:
    """Настроить корневой логгер проекта.

    Args:
        level: имя уровня логирования (``DEBUG``, ``INFO``, ``WARNING``, ...).
    """
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)


def force_utf8_stdout() -> None:
    # Переключить stdout/stderr в UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                continue

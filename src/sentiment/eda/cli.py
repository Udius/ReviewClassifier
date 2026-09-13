"""Командная строка EDA: аргументы, логирование, запуск конвейера.

Отдельный модуль нужен, чтобы код аргументов можно было протестировать без
запуска разведки, а ``python -m sentiment.eda`` оставалось тонкой обёрткой.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sentiment.dataset import DEFAULT_DATA_PATH, DatasetError
from sentiment.eda import run_eda
from sentiment.utils import force_utf8_stdout, setup_logging

logger = logging.getLogger(__name__)

#: Директория для артефактов EDA по умолчанию.
DEFAULT_OUTPUT_DIR = Path("reports/eda")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки скрипта EDA."""
    parser = argparse.ArgumentParser(description="Разведочный анализ датасета отзывов")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help=f"Путь к CSV (по умолчанию {DEFAULT_DATA_PATH})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Каталог артефактов (по умолчанию {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--log-level", default="INFO", help="Уровень логирования (INFO, DEBUG, ...)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Запустить EDA; возвращает код возврата процесса."""
    args = parse_args(argv)
    force_utf8_stdout()
    setup_logging(args.log_level)
    try:
        run_eda(args.data, args.out)
    except DatasetError as error:
        logger.error("Ошибка датасета: %s", error)
        return 1
    return 0

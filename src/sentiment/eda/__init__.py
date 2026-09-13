"""Разведочный анализ (EDA) датасета отзывов.

Пакет закрывает этап «загрузка, разведка, отчёт»: считает баланс классов,
длины текстов, технический шум, строит графики и пишет артефакты в
``reports/eda/``. Доступ к самому CSV живёт в :mod:`sentiment.dataset`.

Запуск из корня репозитория:
    python -m sentiment.eda
    python -m sentiment.eda --data data/sentiment_dataset.csv --out reports/eda
"""

from __future__ import annotations

import logging
from pathlib import Path

from sentiment.dataset import LABEL_TO_NAME, load_and_validate
from sentiment.eda.lengths import TruncationAnalyzer, TruncationStats
from sentiment.eda.noise import NoiseAnalyzer, NoiseStats
from sentiment.eda.statistics import DatasetStatistics, DatasetStatisticsCalculator

logger = logging.getLogger(__name__)

__all__ = [
    "NoiseAnalyzer",
    "NoiseStats",
    "DatasetStatistics",
    "DatasetStatisticsCalculator",
    "TruncationAnalyzer",
    "TruncationStats",
    "run_eda",
]


def run_eda(data_path: Path, output_dir: Path) -> None:
    """Точка входа этапа EDA.

    Конвейер: загрузка и валидация -> статистики -> шум -> (далее) truncation,
    графики и отчёт.
    """
    frame, report = load_and_validate(data_path)

    logger.info("Маппинг классов: %s", LABEL_TO_NAME)
    statistics = DatasetStatisticsCalculator().compute(frame)
    logger.debug("Баланс классов:\n%s", statistics.balance_table().to_string(index=False))
    logger.debug("Разбивка по источникам:\n%s", statistics.source_table().to_string(index=False))

    noise = NoiseAnalyzer().compute(frame)
    logger.debug("Паттерны шума:\n%s", noise.pattern_table().to_string(index=False))

    truncation = TruncationAnalyzer().compute(frame)
    logger.debug("Обрезка по длине:\n%s", truncation.table().to_string(index=False))

    logger.debug("Статистики: %s", statistics.as_dict())
    logger.debug("Шум: %s", noise.as_dict())
    logger.debug("Длины: %s", truncation.as_dict())

    # Отчёт о валидации пока только считается: он попадёт в eda_stats.json,
    # когда будет готов модуль отчётов.
    logger.debug("Валидация: %s", report.as_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Артефакты EDA пишутся в: %s", output_dir)

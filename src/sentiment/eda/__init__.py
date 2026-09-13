"""Разведочный анализ (EDA) датасета отзывов.

Пакет закрывает этап «загрузка, разведка, отчёт»: считает баланс классов,
длины текстов, технический шум, обрезку по max_len, строит графики и пишет
артефакты в ``reports/eda/``. Доступ к самому CSV живёт в
:mod:`sentiment.dataset`.

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
from sentiment.eda.plots import plot_all
from sentiment.eda.report import EdaResults, write_report, write_stats
from sentiment.eda.statistics import DatasetStatistics, DatasetStatisticsCalculator

logger = logging.getLogger(__name__)

__all__ = [
    "NoiseAnalyzer",
    "NoiseStats",
    "DatasetStatistics",
    "DatasetStatisticsCalculator",
    "TruncationAnalyzer",
    "TruncationStats",
    "EdaResults",
    "run_eda",
]


def run_eda(data_path: Path, output_dir: Path) -> None:
    """Точка входа этапа EDA.

    Конвейер: загрузка и валидация -> статистики -> шум -> обрезка по длине
    -> графики -> JSON и markdown-отчёт.
    """
    frame, report = load_and_validate(data_path)

    logger.info("Маппинг классов: %s", LABEL_TO_NAME)
    statistics = DatasetStatisticsCalculator().compute(frame)
    noise = NoiseAnalyzer().compute(frame)
    truncation = TruncationAnalyzer().compute(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_all(frame, statistics, truncation, output_dir)

    results = EdaResults(validation=report, statistics=statistics, noise=noise, truncation=truncation)
    write_stats(results, output_dir)
    write_report(results, output_dir)
    logger.info("EDA завершён, артефакты в: %s", output_dir)

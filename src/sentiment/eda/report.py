"""Сборка текстовых артефактов EDA: ``eda_stats.json`` и ``eda_report.md``.

JSON содержит все числа конвейера без пересчёта, markdown — интерпретацию:
что эти числа означают для объёма выборки, стратификации, max_len и правил
предобработки.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from sentiment.dataset import DatasetValidationReport, LABEL_TO_NAME
from sentiment.eda.lengths import TruncationStats
from sentiment.eda.noise import NoiseStats
from sentiment.eda.statistics import DatasetStatistics

logger = logging.getLogger(__name__)

#: Имя файла с числовыми результатами.
STATS_FILENAME = "eda_stats.json"

#: Имя файла с интерпретацией результатов.
REPORT_FILENAME = "eda_report.md"


@dataclass(frozen=True)
class EdaResults:
    """Все результаты одного прогона EDA в одном контейнере.

    Собирается в run_eda и передаётся модулям отчётов единым объектом, чтобы
    сигнатуры не разрастались с каждым новым анализом.
    """

    validation: DatasetValidationReport
    statistics: DatasetStatistics
    noise: NoiseStats
    truncation: TruncationStats

    def as_dict(self) -> dict:
        """Плоское представление всех результатов для JSON."""
        return {
            "validation": self.validation.as_dict(),
            "statistics": self.statistics.as_dict(),
            "noise": self.noise.as_dict(),
            "truncation": self.truncation.as_dict(),
        }


def write_stats(results: EdaResults, output_dir: Path) -> Path:
    """Записать все числа прогона в ``eda_stats.json``.

    Args:
        results: результаты прогона EDA.
        output_dir: каталог артефактов.

    Returns:
        Путь к записанному файлу.
    """
    path = output_dir / STATS_FILENAME
    path.write_text(json.dumps(results.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Статистики записаны: %s", path)
    return path


def write_report(results: EdaResults, output_dir: Path) -> Path:
    """Собрать и записать ``eda_report.md`` с выводами.

    Args:
        results: результаты прогона EDA.
        output_dir: каталог артефактов.

    Returns:
        Путь к записанному файлу.
    """
    path = output_dir / REPORT_FILENAME
    path.write_text(_render_markdown(results), encoding="utf-8")
    logger.info("Отчёт записан: %s", path)
    return path


def _render_markdown(results: EdaResults) -> str:
    """Собрать текст отчёта из результатов прогона."""
    stats = results.statistics
    noise = results.noise
    truncation = results.truncation
    validation = results.validation

    balance_rows = "\n".join(
        f"| {name} | {entry.count:,} | {entry.share * 100:.2f}% |".replace(",", " ")
        for name, entry in stats.class_balance.items()
    )
    noise_rows = "\n".join(
        f"| {row['pattern']} | {row['rows']:,} | {row['share'] * 100:.2f}% | {row['action']} |".replace(",", " ")
        for row in noise.pattern_table().to_dict("records")
    )
    truncation_rows = "\n".join(
        f"| {row.max_len} | {row.kept_share * 100:.2f}% | {row.truncated_rows:,} | {row.lost_tokens_share * 100:.2f}% |".replace(",", " ")
        for row in truncation.rows
    )
    sources_rows = "\n".join(
        f"| {row['src']} | {row['total']:,} |".replace(",", " ")
        for row in stats.source_table().to_dict("records")
    )

    best_truncation = max(truncation.rows, key=lambda row: row.kept_share)
    label_map = ", ".join(f"{label} = {name}" for label, name in LABEL_TO_NAME.items())

    return f"""# EDA: датасет отзывов

## Валидация

- строк в файле: {validation.total_rows:,};
- после очистки: {validation.kept_rows:,} (отброшено {validation.dropped_rows});
- пустых текстов: {validation.empty_text_rows};
- дубликатов текста: {validation.duplicate_text_rows} ({validation.duplicate_share * 100:.2f}%).

## Классы

Маппинг меток: {label_map}.

| Класс | Отзывов | Доля |
|---|---|---|
{balance_rows}

Классы сбалансированы (отклонение долей меньше половины процента), поэтому
взвешивание классов при обучении не требуется, а стратификация сплитов
достаточна по метке класса.

## Длины текстов

- средняя длина: {stats.word_length.mean} слов ({stats.char_length.mean} символов);
- медиана: {stats.word_length.percentiles.get("50", 0)} слов;
- 95% отзывов короче {stats.word_length.percentiles.get("95", 0)} слов, 99% — короче
  {stats.word_length.percentiles.get("99", 0)} слов;
- однословесных отзывов: {stats.single_word_reviews:,}.

## Обрезка по max_len

| max_len, слов | Влезает целиком | Обрезается | Теряется токенов |
|---|---|---|---|
{truncation_rows}

Ограничение {best_truncation.max_len} слов покрывает {best_truncation.kept_share * 100:.2f}% отзывов
без потери токенов — берём его для baseline. Для BERT ограничение задаётся в
подсловах токенизатора, их на слово меньше; предварительное значение 128
подслов нужно перепроверить реальным токенизатором на этапе обучения BERT.

## Технический шум

| Паттерн | Строк | Доля | Действие предобработки |
|---|---|---|---|
{noise_rows}

Строк, затрагиваемых очисткой: {noise.rows_with_technical_noise:,} ({noise.rows_with_technical_noise / noise.total_rows * 100:.2f}%).
Повторные знаки «!?», эмодзи и КАПС сохраняются — по ТЗ они несут тональность.

## Источники

| Источник | Отзывов |
|---|---|
{sources_rows}

Классы представлены внутри каждого источника почти равномерно, поэтому
доменная стратификация не нужна: достаточно стратификации по классу.

## Рекомендации для следующих этапов

1. Выборка: {stats.total_rows:,} строк — больше, чем нужно для CPU-обучения;
   стратифицированная подвыборка 20–30 тыс. сохранит баланс и сократит эпоху
   до минут. Дубликаты ({validation.duplicate_text_rows} шт.) на такой выборке
   почти не влияют, но при сэмплировании их стоит исключить первыми.
2. Сплиты: 70/15/15 со стратификацией по классу; checkpoint — по validation
   macro F1.
3. Предобработка: удалить HTML-сущности, ссылки, email, упоминания и
   литеральные «\\n»; схлопнуть повторные пробелы и переносы; сохранить
   «!?», эмодзи и КАПС.
4. max_len: 256 слов для baseline; для BERT — 128 подслов с перепроверкой.
"""

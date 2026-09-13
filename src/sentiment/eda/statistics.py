"""Статистики датасета: баланс классов, длины текстов, словарь, источники.

Числа отсюда — вход для решения о размере обучающей выборки, стратификации
сплитов и значении ``max_len``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import pandas as pd

from sentiment.dataset import LABEL_TO_NAME

logger = logging.getLogger(__name__)

#: Токены вида «буквы/цифры с подчёркиванием», регистр не учитывается.
#: Совпадает с word-level токенизацией baseline-модели.
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

#: Перцентили, которые выводятся по распределениям длин.
LENGTH_PERCENTILES: tuple[int, ...] = (50, 75, 90, 95, 99)


@dataclass
class LengthStats:
    """Одно числовое распределение: среднее, разброс и ключевые перцентили."""

    mean: float = 0.0
    std: float = 0.0
    minimum: int = 0
    maximum: int = 0
    percentiles: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_series(cls, values: pd.Series) -> LengthStats:
        """Посчитать статистику по числовой серии (пустая серия даёт нули)."""
        if values.empty:
            return cls()
        quantiles = values.quantile([p / 100 for p in LENGTH_PERCENTILES])
        return cls(
            mean=round(float(values.mean()), 2),
            std=round(float(values.std()), 2),
            minimum=int(values.min()),
            maximum=int(values.max()),
            percentiles={str(p): int(round(float(quantiles[p / 100]))) for p in LENGTH_PERCENTILES},
        )

    def as_dict(self) -> dict:
        """Сериализуемое представление для JSON."""
        return asdict(self)


@dataclass
class ClassBalanceEntry:
    """Доля одного класса в датасете."""

    count: int = 0
    share: float = 0.0


@dataclass
class DatasetStatistics:
    """Сводные статистики датасета, получаемые на этапе EDA.

    Всё, что здесь лежит, — вход для выбора объёма выборки, стратификации и
    значения ``max_len``; в отчёт попадает без пересчёта.
    """

    total_rows: int = 0
    class_balance: dict[str, ClassBalanceEntry] = field(default_factory=dict)
    char_length: LengthStats = field(default_factory=LengthStats)
    word_length: LengthStats = field(default_factory=LengthStats)
    word_length_by_class: dict[str, LengthStats] = field(default_factory=dict)
    vocabulary_size: int = 0
    unique_sources: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    source_label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    single_word_reviews: int = 0

    def balance_table(self) -> pd.DataFrame:
        """Баланс классов в виде таблицы ``class | count | share``."""
        rows = [{"class": name, "count": entry.count, "share": entry.share} for name, entry in self.class_balance.items()]
        return pd.DataFrame(rows)

    def source_table(self) -> pd.DataFrame:
        """Разбивка по источникам: всего строк и по каждому классу."""
        table = pd.DataFrame([{"src": src, "total": total} for src, total in self.source_counts.items()])
        for label_name in self.class_balance:
            table[label_name] = table["src"].map(lambda src, name=label_name: self.source_label_counts.get(src, {}).get(name, 0))
        return table.sort_values("total", ascending=False).reset_index(drop=True)

    def as_dict(self) -> dict:
        """Сериализуемое представление для JSON."""
        return {
            "total_rows": self.total_rows,
            "class_balance": {name: asdict(entry) for name, entry in self.class_balance.items()},
            "char_length": self.char_length.as_dict(),
            "word_length": self.word_length.as_dict(),
            "word_length_by_class": {name: stats.as_dict() for name, stats in self.word_length_by_class.items()},
            "vocabulary_size": self.vocabulary_size,
            "unique_sources": self.unique_sources,
            "source_counts": self.source_counts,
            "source_label_counts": self.source_label_counts,
            "single_word_reviews": self.single_word_reviews,
        }


class DatasetStatisticsCalculator:
    """Считает сводные статистики по очищенному датасету.

    Ответственность — только числа. Графики и текстовые выводы находятся в
    других компонентах (SRP).
    """

    def compute(self, frame: pd.DataFrame) -> DatasetStatistics:
        """Посчитать все статистики.

        Args:
            frame: очищенный DataFrame с колонками ``text``, ``label``, ``src``.

        Returns:
            Заполненный :class:`DatasetStatistics`.
        """
        stats = DatasetStatistics(total_rows=len(frame))
        stats.class_balance = self._class_balance(frame["label"])
        stats.char_length = LengthStats.from_series(frame["text"].str.len())

        word_counts = self._word_counts(frame["text"])
        stats.word_length = LengthStats.from_series(word_counts)
        stats.word_length_by_class = {
            LABEL_TO_NAME[label]: LengthStats.from_series(word_counts[frame["label"] == label])
            for label in sorted(frame["label"].unique())
        }
        stats.single_word_reviews = int((word_counts <= 1).sum())
        stats.vocabulary_size = self._vocabulary_size(frame["text"])
        stats.unique_sources = int(frame["src"].nunique())
        stats.source_counts = {str(src): int(count) for src, count in frame["src"].value_counts().items()}
        stats.source_label_counts = self._source_label_counts(frame)

        self._log_summary(stats)
        return stats

    def _class_balance(self, labels: pd.Series) -> dict[str, ClassBalanceEntry]:
        """Считать количество и долю каждого класса."""
        total = len(labels)
        counts = labels.value_counts().sort_index()
        return {
            LABEL_TO_NAME[int(label)]: ClassBalanceEntry(count=int(count), share=round(int(count) / total, 4))
            for label, count in counts.items()
        }

    def _word_counts(self, texts: pd.Series) -> pd.Series:
        """Число токенов word-level токенизации в каждом тексте."""
        return texts.str.findall(WORD_PATTERN).str.len()

    def _vocabulary_size(self, texts: pd.Series) -> int:
        """Размер словаря уникальных токенов (в нижнем регистре)."""
        tokens: Counter[str] = Counter()
        for text in texts:
            tokens.update(WORD_PATTERN.findall(text))
        return len(tokens)

    def _source_label_counts(self, frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Матрица «источник x класс» в виде вложенного словаря."""
        crosstab = pd.crosstab(frame["src"], frame["label"])
        result: dict[str, dict[str, int]] = {}
        for src in crosstab.index:
            result[str(src)] = {LABEL_TO_NAME[int(label)]: int(crosstab.loc[src, label]) for label in crosstab.columns}
        return result

    def _log_summary(self, stats: DatasetStatistics) -> None:
        """Вывести ключевые числа в лог, чтобы видеть результат без чтения JSON."""
        for name, entry in stats.class_balance.items():
            logger.info("Класс %s: %d строк (%.2f%%)", name, entry.count, entry.share * 100)
        logger.info(
            "Длины в символах: среднее %.1f, p50 %d, p95 %d, максимум %d",
            stats.char_length.mean,
            stats.char_length.percentiles.get("50", 0),
            stats.char_length.percentiles.get("95", 0),
            stats.char_length.maximum,
        )
        logger.info(
            "Длины в словах: среднее %.1f, p50 %d, p95 %d, максимум %d",
            stats.word_length.mean,
            stats.word_length.percentiles.get("50", 0),
            stats.word_length.percentiles.get("95", 0),
            stats.word_length.maximum,
        )
        logger.info("Словарь уникальных токенов: %d", stats.vocabulary_size)
        logger.info("Однословесных отзывов: %d", stats.single_word_reviews)
        logger.info("Источников текста: %d", stats.unique_sources)

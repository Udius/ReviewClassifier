"""Анализ длин текстов относительно кандидатов на ``max_len``.

Верхний отсечение нужен в двух местах: baseline обрезает отзыв по числу слов
своего словаря, BERT — по числу подслов токенизатора. Оценку делаем по словам,
потому что она даёт верхнюю границу для числа токенов: токенизатор никогда не
даёт их больше, чем слов (в русском языке он почти всегда даёт меньше).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import pandas as pd

from sentiment.eda.statistics import WORD_PATTERN

logger = logging.getLogger(__name__)

#: Кандидаты на max_len: компактный, стандартный для BERT и с запасом по p99.
MAX_LENGTH_CANDIDATES: tuple[int, ...] = (64, 128, 256)


@dataclass
class TruncationRow:
    """Последствия одного значения max_len.

    Attributes:
        max_len: рассматриваемое ограничение в словах.
        kept_share: доля отзывов, которые влезают целиком (0..1).
        truncated_rows: сколько отзывов будет обрезано.
        lost_tokens_share: доля всех токенов корпуса, которая при этом теряется.
    """

    max_len: int = 0
    kept_share: float = 0.0
    truncated_rows: int = 0
    lost_tokens_share: float = 0.0

    def as_dict(self) -> dict:
        """Сериализуемое представление строки для JSON."""
        return asdict(self)


@dataclass
class TruncationStats:
    """Результат сравнения кандидатов max_len."""

    rows: list[TruncationRow] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        """Таблица кандидатов: сколько теряется при каждом ограничении."""
        return pd.DataFrame([row.as_dict() for row in self.rows])

    def as_dict(self) -> dict:
        """Сериализуемое представление для JSON."""
        return {"candidates": [row.as_dict() for row in self.rows]}


class TruncationAnalyzer:
    """Считает, какая часть данных теряется при разных ``max_len``."""

    def compute(self, frame: pd.DataFrame, candidates: tuple[int, ...] = MAX_LENGTH_CANDIDATES) -> TruncationStats:
        """Посчитать долю обрезанных отзывов и потерянных токенов.

        Args:
            frame: очищенный DataFrame с колонкой ``text``.
            candidates: значения max_len в словах для сравнения.

        Returns:
            Заполненный :class:`TruncationStats`.
        """
        word_counts = frame["text"].str.findall(WORD_PATTERN).str.len()
        total_tokens = int(word_counts.sum())

        stats = TruncationStats()
        for max_len in candidates:
            truncated = word_counts > max_len
            lost_tokens = (word_counts[truncated] - max_len).sum()
            stats.rows.append(
                TruncationRow(
                    max_len=max_len,
                    kept_share=round(float((~truncated).mean()), 4),
                    truncated_rows=int(truncated.sum()),
                    lost_tokens_share=round(float(lost_tokens) / total_tokens, 4) if total_tokens else 0.0,
                )
            )

        for row in stats.rows:
            logger.info(
                "max_len %d: целиком влезает %.2f%% отзывов, обрезается %d, теряется %.2f%% токенов",
                row.max_len,
                row.kept_share * 100,
                row.truncated_rows,
                row.lost_tokens_share * 100,
            )
        return stats

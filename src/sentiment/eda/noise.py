"""Замер технического шума в текстах отзывов.

Результат нужен одному решению: что именно убирает предобработка. Содержательная
разметка классов здесь не перепроверяется — датасет считается качественным.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Паттерны технического шума. Набор собран по реальным находкам в датасете,
#: а не по списку из теории: HTML здесь почти нет, зато много литеральных
#: последовательностей «\n» и двойных пробелов.
NOISE_PATTERNS: dict[str, re.Pattern[str]] = {
    "html_tag": re.compile(r"<[^>\s]{1,30}>"),
    "html_entity": re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{2,5});"),
    "url": re.compile(r"(?:https?://|www\.)\S+"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "mention": re.compile(r"@\w+"),
    # Литеральная последовательность из двух символов: обратный слэш и буква n.
    # Hex-escape вместо \x5c\x5c — иначе экранирование теряется при правке файла.
    "literal_newline": re.compile(r"\x5cn"),
    "multiple_spaces": re.compile(r"\S {2,}\S"),
    "real_newline": re.compile(r"\n"),
    "repeated_punctuation": re.compile(r"[!?]{3,}"),
    "emoji": re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"),
    "all_caps_word": re.compile(r"\b[А-ЯЁ]{4,}\b"),
}

#: Паттерны, которые предобработка обязана убирать (технический мусор).
NOISE_TO_REMOVE: tuple[str, ...] = ("html_tag", "html_entity", "url", "email", "mention", "literal_newline")

#: Паттерны, которые предобработка нормализует, но не удаляет совсем.
NOISE_TO_NORMALIZE: tuple[str, ...] = ("multiple_spaces", "real_newline")

#: Паттерны, которые по ТЗ сохраняются как носители тональности.
NOISE_TO_KEEP: tuple[str, ...] = ("repeated_punctuation", "emoji", "all_caps_word")


@dataclass
class NoiseStats:
    """Оценка технического шума в текстах отзывов.

    Числа нужны, чтобы правила предобработки опирались на факты: если паттерн
    встречается в долях ниже процента, усложнять ради него очистку бессмысленно.
    """

    total_rows: int = 0
    pattern_counts: dict[str, int] = field(default_factory=dict)
    pattern_shares: dict[str, float] = field(default_factory=dict)
    rows_with_technical_noise: int = 0

    def pattern_table(self) -> pd.DataFrame:
        """Таблица «паттерн | строк | доля | что с ним делать»."""
        def action(name: str) -> str:
            if name in NOISE_TO_REMOVE:
                return "удалять"
            if name in NOISE_TO_NORMALIZE:
                return "нормализовать"
            return "сохранить"

        rows = [
            {"pattern": name, "rows": self.pattern_counts[name], "share": self.pattern_shares[name], "action": action(name)}
            for name in NOISE_PATTERNS
        ]
        return pd.DataFrame(rows)

    def as_dict(self) -> dict:
        """Сериализуемое представление для JSON."""
        return asdict(self)


class NoiseAnalyzer:
    """Считает долю текстов с техническим шумом.

    Сюда входит только то, что правит предобработка: HTML, ссылки, «\n» как
    текст, лишние пробелы. Содержательная разметка классов считается
    достоверной и здесь не перепроверяется.
    """

    def compute(self, frame: pd.DataFrame) -> NoiseStats:
        """Посчитать шумовые статистики.

        Args:
            frame: очищенный DataFrame с колонками ``text`` и ``label``.

        Returns:
            Заполненный :class:`NoiseStats`.
        """
        stats = NoiseStats(total_rows=len(frame))
        texts = frame["text"]

        masks: dict[str, pd.Series] = {}
        for name, pattern in NOISE_PATTERNS.items():
            mask = texts.str.contains(pattern, regex=True, na=False)
            masks[name] = mask
            stats.pattern_counts[name] = int(mask.sum())
            stats.pattern_shares[name] = round(float(mask.mean()), 4)

        technical = masks[NOISE_TO_REMOVE[0]]
        for name in NOISE_TO_REMOVE[1:]:
            technical = technical | masks[name]
        stats.rows_with_technical_noise = int(technical.sum())

        self._log_summary(stats)
        return stats

    def _log_summary(self, stats: NoiseStats) -> None:
        """Вывести доли технического шума в лог."""
        for name, share in stats.pattern_shares.items():
            logger.info("Шум %s: %d строк (%.2f%%)", name, stats.pattern_counts[name], share * 100)
        logger.info(
            "Строк с техническим шумом (удаляется предобработкой): %d (%.2f%%)",
            stats.rows_with_technical_noise,
            stats.rows_with_technical_noise / stats.total_rows * 100 if stats.total_rows else 0.0,
        )

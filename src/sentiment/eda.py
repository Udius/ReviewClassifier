"""Разведочный анализ (EDA) датасета отзывов.

Модуль закрывает: загрузку данных, базовую валидацию,
статистики, оценку уровня шума и длины текстов, построение графиков и запись
отчётов в ``reports/eda/``.

Запуск из корня репозитория:
    python -m sentiment.eda
    python -m sentiment.eda --data data/sentiment_dataset.csv --out reports/eda
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from sentiment.utils import force_utf8_stdout, setup_logging

logger = logging.getLogger(__name__)

#: Путь к датасету по умолчанию (файл не коммитится, см. .gitignore).
DEFAULT_DATA_PATH = Path("data/sentiment_dataset.csv")

#: Директория для артефактов EDA по умолчанию.
DEFAULT_OUTPUT_DIR = Path("reports/eda")

#: Обязательные колонки входного CSV.
REQUIRED_COLUMNS: tuple[str, ...] = ("text", "label", "src")

#: Маппинг числовых меток в имена классов (зафиксирован в ТЗ, п. 2 и п. 11).
LABEL_TO_NAME: dict[int, str] = {0: "neutral", 1: "positive", 2: "negative"}

#: Обратный маппинг: имя класса -> числовая метка (нужен для Dataset/метрик).
NAME_TO_LABEL: dict[str, int] = {name: label for label, name in LABEL_TO_NAME.items()}

#: Число классов задачи.
NUM_CLASSES: int = len(LABEL_TO_NAME)

#: Ожидаемые значения колонки src (для диагностических логов, не для жёсткой проверки).
KNOWN_SOURCES: tuple[str, ...] = (
    "rureviews",
    "geo",
    "perekrestok",
    "anime",
    "kinopoisk",
    "rusentiment",
    "linis",
    "ru-reviews-classification",
    "sber",
    "news",
    "bank",
)


class DatasetError(RuntimeError):
    """Ошибка загрузки или схемы датасета: дальше считать статистики нельзя."""


@dataclass
class DatasetValidationReport:
    """Результат базовой валидации датасета.

    Поля содержат только счётчики и доли — их же пишет отчёт EDA, чтобы цифры
    в коде, JSON и markdown не разъезжались.
    """

    total_rows: int = 0
    missing_by_column: dict[str, int] = field(default_factory=dict)
    empty_text_rows: int = 0
    duplicate_text_rows: int = 0
    unexpected_label_rows: int = 0
    kept_rows: int = 0

    @property
    def dropped_rows(self) -> int:
        """Сколько строк отброшено при очистке (пропуски и пустые тексты)."""
        return self.total_rows - self.kept_rows

    @property
    def duplicate_share(self) -> float:
        """Доля строк с текстом, дублирующим другой текст (0..1)."""
        if self.total_rows == 0:
            return 0.0
        return self.duplicate_text_rows / self.total_rows

    def as_dict(self) -> dict:
        """Сериализуемое представление отчёта для JSON."""
        data = asdict(self)
        data["dropped_rows"] = self.dropped_rows
        data["duplicate_share"] = round(self.duplicate_share, 6)
        return data


@dataclass(frozen=True)
class DatasetSpec:
    """Описание ожидаемой схемы датасета.

    Вынесено в отдельный неизменяемый класс, чтобы правила схемы можно было
    переиспользовать (обучение, инференс) без чтения модуля EDA.
    """

    required_columns: tuple[str, ...] = REQUIRED_COLUMNS
    valid_labels: tuple[int, ...] = tuple(sorted(LABEL_TO_NAME))

    def missing_columns(self, columns) -> tuple[str, ...]:
        """Вернуть обязательные колонки, отсутствующие в датасете."""
        present = set(columns)
        return tuple(col for col in self.required_columns if col not in present)


class ReviewsDatasetLoader:
    """Загружает CSV с отзывами и приводит его к внутренней схеме.

    Ответственность — только чтение файла и проверка схемы. Статистикой и
    графиками занимаются другие компоненты модуля (SRP).
    """

    def __init__(self, spec: DatasetSpec | None = None) -> None:
        """Args:
            spec: ожидаемая схема; по умолчанию используется :class:`DatasetSpec`.
        """
        self._spec = spec or DatasetSpec()

    def load(self, path: Path) -> pd.DataFrame:
        """Прочитать CSV и вернуть DataFrame с нормализованными колонками.

        Args:
            path: путь к CSV-файлу датасета.

        Returns:
            DataFrame с колонками ``text`` (str), ``label`` (int64), ``src`` (str).

        Raises:
            DatasetError: файла нет, CSV пуст, нарушена схема или типы колонок.
        """
        if not path.exists():
            raise DatasetError(
                f"Датасет не найден: {path}. Скачай Russian Sentiment Dataset и положи файл в data/."
            )

        logger.info("Чтение датасета: %s", path)
        frame = pd.read_csv(path)

        missing = self._spec.missing_columns(frame.columns)
        if missing:
            raise DatasetError(f"В датасете нет обязательных колонок {missing}. Колонки файла: {list(frame.columns)}")

        if frame.empty:
            raise DatasetError(f"Датасет пуст: {path}")

        frame = self._coerce_dtypes(frame)
        logger.info("Загружено строк: %d, колонок: %d", len(frame), frame.shape[1])
        return frame

    def _coerce_dtypes(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Привести колонки к ожидаемым типам и отбросить строки с битой меткой.

        Метка обязана быть целым числом из ``spec.valid_labels``: строки вне
        этого диапазона делать «нейтральными» нельзя, поэтому они отбрасываются,
        но количество фиксируется в явном виде (см. :class:`DatasetValidator`).
        """
        frame = frame.loc[:, list(self._spec.required_columns)].copy()
        frame["text"] = frame["text"].astype("string")
        frame["src"] = frame["src"].astype("string")

        labels = pd.to_numeric(frame["label"], errors="coerce")
        frame["label"] = labels.astype("Int64")
        return frame


class DatasetValidator:
    """Проверяет содержательную целостность датасета и чистит его для EDA.

    Правила:
    - колонки не должны быть пустыми целиком;
    - метка должна входить в ожидаемый набор;
    - пустой/пробельный текст неинформативен — строка отбрасывается;
    - дубликаты текста считаются, но НЕ удаляются: их доля — часть вывода EDA,
      решение об удалении принимается на этапе подготовки выборки.
    """

    def __init__(self, spec: DatasetSpec | None = None) -> None:
        """Args:
            spec: ожидаемая схема датасета.
        """
        self._spec = spec or DatasetSpec()

    def validate(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, DatasetValidationReport]:
        """Проверить датасет и вернуть очищенный DataFrame + отчёт.

        Args:
            frame: DataFrame, загруженный :class:`ReviewsDatasetLoader`.

        Returns:
            Кортеж ``(очищенный DataFrame, отчёт о валидации)``.

        Raises:
            DatasetError: метка целиком нечисловая или после очистки не осталось строк.
        """
        report = DatasetValidationReport(total_rows=len(frame))
        report.missing_by_column = {col: int(frame[col].isna().sum()) for col in self._spec.required_columns}
        logger.info("Пропуски по колонкам: %s", report.missing_by_column)

        if frame["label"].isna().all():
            raise DatasetError("Колонка label не содержит корректных чисел — проверь формат датасета.")

        text_stripped = frame["text"].str.strip()
        is_empty_text = text_stripped.isna() | (text_stripped == "")
        report.empty_text_rows = int(is_empty_text.sum())

        label_is_valid = frame["label"].isin(self._spec.valid_labels)
        report.unexpected_label_rows = int((frame["label"].notna() & ~label_is_valid).sum())
        if report.unexpected_label_rows:
            logger.warning("Строк с меткой вне ожидаемого набора %s: %d", list(self._spec.valid_labels), report.unexpected_label_rows)

        # Дубликаты считаем только по информативным текстам: пустые строки уже отброшены.
        report.duplicate_text_rows = int(text_stripped[~is_empty_text].duplicated(keep="first").sum())

        cleaned = frame.loc[~is_empty_text & label_is_valid].copy()
        cleaned["text"] = cleaned["text"].astype(str).str.strip()
        cleaned["label"] = cleaned["label"].astype("int64")
        cleaned["src"] = cleaned["src"].astype(str).str.strip()

        if cleaned.empty:
            raise DatasetError("После очистки не осталось ни одной строки — датасет непригоден.")

        report.kept_rows = len(cleaned)
        logger.info(
            "Валидация: строк %d -> %d (отброшено %d), пустых текстов %d, дубликатов текста %d",
            report.total_rows,
            report.kept_rows,
            report.dropped_rows,
            report.empty_text_rows,
            report.duplicate_text_rows,
        )
        return cleaned, report


def label_names(labels: pd.Series) -> pd.Series:
    """Преобразовать числовые метки в человекочитаемые имена классов."""
    return labels.map(LABEL_TO_NAME)


def load_and_validate(path: Path) -> tuple[pd.DataFrame, DatasetValidationReport]:
    """Полный цикл «прочитать и проверить» для удобного вызова из других модулей.

    Args:
        path: путь к CSV датасета.

    Returns:
        Кортеж ``(очищенный DataFrame, отчёт о валидации)``.
    """
    frame = ReviewsDatasetLoader().load(path)
    return DatasetValidator().validate(frame)


# --- Статистики датасета -------------------------------------------------------

#: Токены вида «буквы/цифры с подчёркиванием», регистр не учитывается.
#: Совпадает с word-level токенизацией baseline-модели.
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

#: Перцентили, которые выводятся по распределениям длин.
LENGTH_PERCENTILES: tuple[int, ...] = (50, 75, 90, 95, 99)


# --- Шум в текстах -------------------------------------------------------------

#: Паттерны технического шума. Набор собран по реальным находкам в датасете,
#: а не по списку из теории: HTML здесь почти нет, зато много литеральных
#: последовательностей «\n» и двойных пробелов.
NOISE_PATTERNS: dict[str, re.Pattern[str]] = {
    "html_tag": re.compile(r"<[^>\s]{1,30}>"),
    "html_entity": re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{2,5});"),
    "url": re.compile(r"(?:https?://|www\.)\S+"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "mention": re.compile(r"@\w+"),
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
    других компонентах модуля (SRP).
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


def run_eda(data_path: Path, output_dir: Path) -> None:
    """Точка входа этапа EDA.

    Конвейер: загрузка и валидация -> статистики -> (далее) шум, truncation,
    графики и отчёт.
    """
    frame, report = load_and_validate(data_path)

    logger.info("Маппинг классов: %s", LABEL_TO_NAME)
    statistics = DatasetStatisticsCalculator().compute(frame)
    logger.debug("Баланс классов:\n%s", statistics.balance_table().to_string(index=False))
    logger.debug("Разбивка по источникам:\n%s", statistics.source_table().to_string(index=False))

    noise = NoiseAnalyzer().compute(frame)
    logger.debug("Паттерны шума:\n%s", noise.pattern_table().to_string(index=False))
    logger.debug("Статистики: %s", statistics.as_dict())
    logger.debug("Шум: %s", noise.as_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Артефакты EDA пишутся в: %s", output_dir)


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


if __name__ == "__main__":
    raise SystemExit(main())

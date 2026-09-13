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


def run_eda(data_path: Path, output_dir: Path) -> None:
    """Точка входа этапа EDA.

    Сейчас реализованы загрузка и валидация (шаг 4 плана). Дальнейшие блоки —
    статистики, шум, truncation, графики и отчёт — добавляются в этот же конвейер.
    """
    frame, report = load_and_validate(data_path)

    logger.info("Маппинг классов: %s", LABEL_TO_NAME)
    balance = frame["label"].value_counts().sort_index()
    for label, count in balance.items():
        logger.info(
            "Класс %d (%s): %d строк (%.2f%%)",
            label,
            LABEL_TO_NAME[label],
            count,
            100.0 * count / len(frame),
        )

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

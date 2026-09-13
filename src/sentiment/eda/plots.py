"""Построение графиков EDA и сохранение их в ``reports/eda/``.

Модуль отвечает только за отрисовку: все числа уже посчитаны в statistics,
noise и lengths. Распределения берутся из исходного DataFrame, агрегаты —
из готовых статистик.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

# Неинтерактивный бэкенд: запуск без дисплея (CI, контейнер, ssh) не падает.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sentiment.eda.lengths import TruncationStats
from sentiment.eda.statistics import WORD_PATTERN, DatasetStatistics

logger = logging.getLogger(__name__)

#: Стиль графиков для всех фигур отчёта.
SEABORN_THEME = "whitegrid"

#: Верхняя граница гистограмм длин в словах: выше p99 хвост только ломает масштаб.
HISTOGRAM_CLIP_WORDS = 300

#: Разрешение сохраняемых фигур.
FIGURE_DPI = 150

#: Подписи классов для легенд (число -> имя).
CLASS_LEGEND_LABELS = {0: "neutral", 1: "positive", 2: "negative"}


def _save(figure: plt.Figure, output_dir: Path, name: str) -> Path:
    """Сохранить фигуру и закрыть её, чтобы не копить объекты в памяти."""
    path = output_dir / name
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    logger.info("График сохранён: %s", path)
    return path


def plot_class_balance(stats: DatasetStatistics, output_dir: Path) -> Path:
    """Столбчатая диаграмма баланса классов с подписями значений."""
    table = stats.balance_table()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(table["class"], table["count"], color=sns.color_palette("deep", len(table)))
    for bar, count in zip(bars, table["count"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}".replace(",", " "), ha="center", va="bottom")
    ax.set_title("Баланс классов")
    ax.set_ylabel("Число отзывов")
    fig.tight_layout()
    return _save(fig, output_dir, "class_balance.png")


def plot_word_lengths_by_class(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Гистограммы распределения длин в словах для каждого класса."""
    word_counts = frame["text"].str.findall(WORD_PATTERN).str.len().clip(upper=HISTOGRAM_CLIP_WORDS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = range(0, HISTOGRAM_CLIP_WORDS + 10, 10)
    for label, group in word_counts.groupby(frame["label"]):
        ax.hist(group, bins=bins, alpha=0.55, label=CLASS_LEGEND_LABELS.get(int(label), str(label)), density=True)
    ax.set_title("Распределение длины отзыва в словах (обрезано на 300)")
    ax.set_xlabel("Слов в отзыве")
    ax.set_ylabel("Плотность")
    ax.legend(title="Класс")
    fig.tight_layout()
    return _save(fig, output_dir, "word_lengths_by_class.png")


def plot_word_lengths_by_source(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Boxplot длин в словах по источникам: видно, какие домены многословнее."""
    lengths = frame["text"].str.findall(WORD_PATTERN).str.len().clip(upper=HISTOGRAM_CLIP_WORDS)
    plot_frame = pd.DataFrame({"src": frame["src"], "words": lengths})
    order = plot_frame.groupby("src")["words"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=plot_frame, x="src", y="words", order=order, ax=ax)
    ax.set_title("Длина отзыва в словах по источникам")
    ax.set_xlabel("")
    ax.set_ylabel("Слов в отзыве")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save(fig, output_dir, "word_lengths_by_source.png")


def plot_source_label_heatmap(stats: DatasetStatistics, output_dir: Path) -> Path:
    """Тепловая карта «источник x класс» в долях внутри источника.

    Абсолютные числа различаются в десятки раз, поэтому нормируем по строке:
    интересна структура классов внутри домена, а не размер домена.
    """
    table = stats.source_table().set_index("src")
    classes = [name for name in table.columns if name != "total"]
    shares = table[classes].div(table[classes].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(shares, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_title("Доли классов внутри каждого источника")
    ax.set_xlabel("Класс")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, output_dir, "source_label_heatmap.png")


def plot_truncation(truncation: TruncationStats, output_dir: Path) -> Path:
    """Доля отзывов, влезающих целиком, в зависимости от max_len."""
    table = truncation.table()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["max_len"], table["kept_share"] * 100, marker="o")
    for _, row in table.iterrows():
        ax.annotate(
            f"{row['kept_share'] * 100:.1f}%",
            (row["max_len"], row["kept_share"] * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    ax.set_title("Доля отзывов, влезающих целиком, от max_len")
    ax.set_xlabel("max_len, слов")
    ax.set_ylabel("Влезает, %")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return _save(fig, output_dir, "truncation.png")


def plot_all(frame: pd.DataFrame, stats: DatasetStatistics, truncation: TruncationStats, output_dir: Path) -> list[Path]:
    """Построить весь стандартный набор графиков EDA."""
    sns.set_theme(style=SEABORN_THEME)
    return [
        plot_class_balance(stats, output_dir),
        plot_word_lengths_by_class(frame, output_dir),
        plot_word_lengths_by_source(frame, output_dir),
        plot_source_label_heatmap(stats, output_dir),
        plot_truncation(truncation, output_dir),
    ]

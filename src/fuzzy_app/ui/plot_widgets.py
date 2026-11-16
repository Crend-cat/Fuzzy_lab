"""Виджеты Matplotlib для плотностей, α-сечений и кластеризации."""

from __future__ import annotations

import math

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fuzzy_core.gaussian import density_for_number, maximum_density
from fuzzy_core.level_sets import level_interval
from fuzzy_core.models import GaussianFuzzyNumber
from fuzzy_clustering.results import ClusterResult


class DistributionPlotWidget(QWidget):
    """Показывает выбранную плотность, все плотности и зависимость N(α).

    На вкладке «Плотности» два графика расположены рядом:
    слева — выбранная величина, справа — все величины исходного множества.
    Прошедшие α-порог рисуются синим цветом, не прошедшие — серым пунктиром.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        # Первая вкладка: два согласованных графика плотности рядом.
        density_page = QWidget()
        density_layout = QVBoxLayout(density_page)
        density_layout.setContentsMargins(0, 8, 0, 0)
        density_layout.setSpacing(2)

        counter_row = QHBoxLayout()
        counter_row.setContentsMargins(0, 0, 0, 0)
        counter_row.setSpacing(6)

        self.alpha_summary = QLabel("Уровень α: 0.700")
        self.alpha_summary.setStyleSheet("font-weight: 600;")
        self.count_summary = QLabel("Подходят: 0 из 0; исключено: 0")
        self.count_summary.setStyleSheet("font-weight: 600;")

        counter_row.addWidget(self.alpha_summary)
        counter_row.addStretch(1)
        counter_row.addWidget(self.count_summary)
        density_layout.addLayout(counter_row)

        charts_row = QHBoxLayout()
        charts_row.setContentsMargins(0, 0, 0, 0)
        charts_row.setSpacing(10)

        # Левый график — выбранная нечёткая величина.
        selected_panel = QWidget()
        selected_layout = QVBoxLayout(selected_panel)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.setSpacing(2)
        selected_layout.addWidget(QLabel("Выбранная нечёткая величина"))

        self.selected_figure = Figure(figsize=(6.0, 4.8), tight_layout=True)
        self.selected_canvas = FigureCanvas(self.selected_figure)
        selected_layout.addWidget(self.selected_canvas, 1)

        # Правый график — все плотности исходного множества.
        all_panel = QWidget()
        all_layout = QVBoxLayout(all_panel)
        all_layout.setContentsMargins(0, 0, 0, 0)
        all_layout.setSpacing(2)
        all_layout.addWidget(QLabel("Все плотности исходного множества"))

        self.collection_figure = Figure(figsize=(6.0, 4.8), tight_layout=True)
        self.collection_canvas = FigureCanvas(self.collection_figure)
        all_layout.addWidget(self.collection_canvas, 1)

        charts_row.addWidget(selected_panel, 1)
        charts_row.addWidget(all_panel, 1)
        density_layout.addLayout(charts_row, 1)
        self.tabs.addTab(density_page, "Плотности")

        # Вторая вкладка: зависимость количества подходящих величин N(α).
        count_page = QWidget()
        count_layout = QVBoxLayout(count_page)
        self.count_figure = Figure(figsize=(9.0, 5.8), tight_layout=True)
        self.count_canvas = FigureCanvas(self.count_figure)
        count_layout.addWidget(self.count_canvas)
        self.tabs.addTab(count_page, "Количество от α")

    @staticmethod
    def _count_at_level(items: list[GaussianFuzzyNumber], alpha: float) -> int:
        """Количество объектов с непустым α-сечением."""
        return sum(level_interval(item, alpha) is not None for item in items)

    def show_number(
        self,
        item: GaussianFuzzyNumber,
        all_items: list[GaussianFuzzyNumber],
        alpha: float,
    ) -> None:
        """Обновить графики и счётчик после выбора объекта или изменения α."""
        items = list(all_items)
        included_count = self._count_at_level(items, alpha)
        excluded_count = len(items) - included_count

        self.alpha_summary.setText(f"Уровень α: {alpha:.6g}")
        self.count_summary.setText(
            f"Подходят: {included_count} из {len(items)}; исключено: {excluded_count}"
        )

        self._draw_selected_density(item, alpha)
        self._draw_all_densities(items, item, alpha, included_count)
        self._draw_count_curve(items, alpha)

    def _draw_selected_density(self, item: GaussianFuzzyNumber, alpha: float) -> None:
        self.selected_figure.clear()
        ax = self.selected_figure.add_subplot(1, 1, 1)

        # ±4.5σ — только область построения, а не границы распределения.
        x_values = np.linspace(
            item.x0 - 4.5 * item.sigma,
            item.x0 + 4.5 * item.sigma,
            700,
        )
        y_values = density_for_number(x_values, item)
        peak = maximum_density(item.sigma)

        ax.plot(
            x_values,
            y_values,
            linewidth=2.2,
            color="tab:blue",
            label=f"Плотность A{item.id}",
        )
        ax.axhline(
            alpha,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"Уровень α = {alpha:g}",
        )
        ax.axvline(
            item.x0,
            color="tab:green",
            linestyle=":",
            linewidth=1.3,
            label=f"x₀ = {item.x0:g}",
        )

        interval = level_interval(item, alpha)
        if interval is not None:
            ax.axvline(
                interval.left,
                color="tab:purple",
                linestyle="-.",
                linewidth=1.1,
                label=f"xL = {interval.left:.5g}",
            )
            if not math.isclose(
                interval.left,
                interval.right,
                rel_tol=1e-12,
                abs_tol=1e-14,
            ):
                ax.axvline(
                    interval.right,
                    color="tab:purple",
                    linestyle="-.",
                    linewidth=1.1,
                    label=f"xR = {interval.right:.5g}",
                )
            ax.fill_between(
                x_values,
                y_values,
                alpha,
                where=(y_values >= alpha),
                color="tab:blue",
                alpha=0.16,
                interpolate=True,
            )
        else:
            ax.text(
                0.02,
                0.96,
                "Максимум плотности ниже α:\nα-сечение пустое",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

        ax.set_title(f"A{item.id}: x₀={item.x0:.5g}, σ={item.sigma:.5g}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_ylim(bottom=0.0, top=max(peak, alpha) * 1.12)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        # Нижнюю границу области построения поднимаем, не меняя диапазоны осей.
        self.selected_figure.subplots_adjust(
            left=0.12,
            right=0.96,
            bottom=0.13,
            top=0.88,
        )
        self.selected_canvas.draw_idle()

    def _draw_all_densities(
        self,
        all_items: list[GaussianFuzzyNumber],
        selected_item: GaussianFuzzyNumber,
        alpha: float,
        included_count: int,
    ) -> None:
        """Нарисовать все плотности исходного множества."""
        self.collection_figure.clear()
        ax = self.collection_figure.add_subplot(1, 1, 1)

        ax.axhline(
            alpha,
            color="tab:red",
            linestyle="--",
            linewidth=1.6,
            label=f"Уровень α = {alpha:g}",
            zorder=4,
        )

        if not all_items:
            ax.text(
                0.5,
                0.5,
                "Исходное множество пусто",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(0.0, max(alpha * 1.12, 0.1))
        else:
            all_left = min(
                candidate.x0 - 4.5 * candidate.sigma for candidate in all_items
            )
            all_right = max(
                candidate.x0 + 4.5 * candidate.sigma for candidate in all_items
            )
            if math.isclose(all_left, all_right):
                all_left -= 1.0
                all_right += 1.0

            # Каждая функция строится на собственном интервале, чтобы узкие
            # кривые не терялись на общей широкой оси x.
            for candidate in all_items:
                candidate_x = np.linspace(
                    candidate.x0 - 4.5 * candidate.sigma,
                    candidate.x0 + 4.5 * candidate.sigma,
                    260,
                )
                candidate_y = density_for_number(candidate_x, candidate)
                included = level_interval(candidate, alpha) is not None
                is_selected = candidate.id == selected_item.id

                if is_selected:
                    ax.plot(
                        candidate_x,
                        candidate_y,
                        color="black",
                        linewidth=2.0,
                        alpha=0.95,
                        zorder=3,
                    )
                elif included:
                    ax.plot(
                        candidate_x,
                        candidate_y,
                        color="tab:blue",
                        linewidth=0.9,
                        alpha=0.32,
                        zorder=2,
                    )
                else:
                    ax.plot(
                        candidate_x,
                        candidate_y,
                        color="0.65",
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.27,
                        zorder=1,
                    )

            handles = [
                Line2D(
                    [0], [0], color="black", linewidth=2.0,
                    label=f"Выбранная A{selected_item.id}",
                ),
                Line2D(
                    [0], [0], color="tab:blue", linewidth=1.5,
                    label="Прошла α-уровень",
                ),
                Line2D(
                    [0], [0], color="0.65", linewidth=1.2,
                    linestyle="--", label="Не прошла α-уровень",
                ),
                Line2D(
                    [0], [0], color="tab:red", linewidth=1.5,
                    linestyle="--", label="Горизонтальный уровень α",
                ),
            ]
            ax.legend(handles=handles, fontsize=8, loc="best")

            largest_peak = max(
                maximum_density(candidate.sigma) for candidate in all_items
            )
            ax.set_xlim(all_left, all_right)
            ax.set_ylim(bottom=0.0, top=max(largest_peak, alpha) * 1.12)

        ax.set_title(
            f"Все плотности: {included_count} из {len(all_items)} прошли α-уровень"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.22)

        self.collection_figure.subplots_adjust(
            left=0.12,
            right=0.96,
            bottom=0.13,
            top=0.88,
        )
        self.collection_canvas.draw_idle()

    def _draw_count_curve(self, items: list[GaussianFuzzyNumber], alpha: float) -> None:
        """Построить ступенчатую зависимость N(α) для исходного множества."""
        self.count_figure.clear()
        ax = self.count_figure.add_subplot(1, 1, 1)
        total_count = len(items)
        current_count = self._count_at_level(items, alpha)

        ax.axvline(
            alpha,
            color="tab:red",
            linestyle="--",
            linewidth=1.6,
            label=f"Текущий α = {alpha:g}",
        )

        if total_count == 0:
            ax.text(
                0.5,
                0.5,
                "Исходное множество пусто",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(0.0, max(alpha * 1.1, 1.0))
            ax.set_ylim(0.0, 1.0)
        else:
            peak_values = sorted({maximum_density(item.sigma) for item in items})
            peak_counts = [self._count_at_level(items, value) for value in peak_values]
            right_limit = max(peak_values[-1] * 1.12, alpha * 1.12, 1e-6)

            # В точке α=f_max объект ещё включён (условие f_max >= α).
            step_x = [0.0, *peak_values, right_limit]
            step_y = [total_count, *peak_counts, 0]
            ax.step(
                step_x,
                step_y,
                where="pre",
                linewidth=2.0,
                color="tab:blue",
                label="N(α) — подходящие величины",
            )
            ax.scatter(
                peak_values,
                peak_counts,
                s=24,
                color="tab:blue",
                zorder=3,
                label="Пороговые значения f_max",
            )
            ax.scatter(
                [alpha],
                [current_count],
                s=105,
                color="tab:red",
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
                label=f"Сейчас: {current_count} из {total_count}",
            )
            ax.annotate(
                f"α = {alpha:g}\nN(α) = {current_count}",
                xy=(alpha, current_count),
                xytext=(10, 12),
                textcoords="offset points",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88},
            )
            ax.set_xlim(0.0, right_limit)
            ax.set_ylim(-max(1, total_count * 0.04), total_count * 1.12)

        ax.set_title("Зависимость количества подходящих величин от уровня α")
        ax.set_xlabel("α — порог плотности")
        ax.set_ylabel("Количество величин N(α)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        self.count_canvas.draw_idle()

    def clear(self) -> None:
        """Очистить графики и счётчики, если исходный набор пуст."""
        self.alpha_summary.setText("Уровень α: —")
        self.count_summary.setText("Подходят: 0 из 0; исключено: 0")
        for figure, canvas in (
            (self.selected_figure, self.selected_canvas),
            (self.collection_figure, self.collection_canvas),
            (self.count_figure, self.count_canvas),
        ):
            figure.clear()
            canvas.draw_idle()


class ClusterPlotsWidget(QWidget):
    """Дендрограмма с текущим K-срезом и параметрическая плоскость кластеров.

    Иерархическое дерево слияний определяется матрицей расстояний и Average
    Linkage, поэтому его топология сама по себе не меняется при изменении K.
    При изменении K пересчитывается разбиение, на дендрограмме перемещается
    горизонтальная линия среза и перекрашиваются ветви согласно новому составу
    кластеров. Параметрическая плоскость также перестраивается для нового K.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    @staticmethod
    def _cluster_palette(cluster_ids: list[int]) -> dict[int, str]:
        """Создать различимые цвета для текущих кластеров."""
        if not cluster_ids:
            return {}
        cmap = colormaps.get_cmap("turbo").resampled(max(len(cluster_ids), 2))
        denominator = max(len(cluster_ids) - 1, 1)
        return {
            cluster_id: to_hex(cmap(index / denominator))
            for index, cluster_id in enumerate(cluster_ids)
        }

    @staticmethod
    def _dendrogram_node_colors(
        linkage_matrix: np.ndarray,
        cluster_labels: list[int],
        cluster_colors: dict[int, str],
    ) -> dict[int, str]:
        """Окрасить ветви по кластеру, которому принадлежат все их листья.

        Поддерево, объединяющее листья из разных кластеров, окрашивается серым.
        Поэтому при изменении K цветовая структура дендрограммы обновляется.
        """
        object_count = len(cluster_labels)
        node_leaves: dict[int, set[int]] = {
            leaf_id: {leaf_id} for leaf_id in range(object_count)
        }
        node_colors: dict[int, str] = {}

        for merge_index, merge in enumerate(np.asarray(linkage_matrix)):
            left_node = int(merge[0])
            right_node = int(merge[1])
            leaves = node_leaves[left_node] | node_leaves[right_node]
            node_id = object_count + merge_index
            node_leaves[node_id] = leaves

            labels_in_node = {cluster_labels[leaf_id] for leaf_id in leaves}
            if len(labels_in_node) == 1:
                cluster_id = next(iter(labels_in_node))
                node_colors[node_id] = cluster_colors[cluster_id]
            else:
                node_colors[node_id] = "#858585"

        return node_colors

    @staticmethod
    def _cut_height_for_k(
        linkage_matrix: np.ndarray,
        object_count: int,
        k: int,
    ) -> tuple[float | None, bool]:
        """Найти высоту среза дерева для K кластеров.

        При равных высотах нескольких слияний одна горизонтальная линия может
        не соответствовать ровно K группам. В таком случае точное разбиение
        остаётся заданным результатом cut_tree и цветами листовых поддеревьев.
        """
        if object_count < 2:
            return None, False

        heights = np.asarray(linkage_matrix, dtype=float)[:, 2]
        if k >= object_count:
            return 0.0, False
        if k <= 1:
            maximum = float(heights[-1])
            margin = max(abs(maximum) * 0.05, 1e-9)
            return maximum + margin, False

        # После object_count - k слияний остаётся ровно k компонентов.
        last_included_merge = object_count - k - 1
        first_excluded_merge = object_count - k
        lower_height = float(heights[last_included_merge])
        upper_height = float(heights[first_excluded_merge])
        has_ties = math.isclose(
            lower_height,
            upper_height,
            rel_tol=1e-12,
            abs_tol=1e-14,
        )
        return (lower_height + upper_height) / 2.0, has_ties

    def show_result(self, result: ClusterResult) -> None:
        """Перерисовать графики по актуальному результату кластеризации."""
        self.figure.clear()
        dendro_ax = self.figure.add_subplot(1, 2, 1)
        scatter_ax = self.figure.add_subplot(1, 2, 2)

        items = list(result.included_items)
        object_count = len(items)
        selected_k = result.selected_k
        cluster_ids = sorted(result.clusters)
        cluster_colors = self._cluster_palette(cluster_ids)

        if result.linkage_matrix is not None and object_count >= 2:
            # Порядок labels соответствует порядку объектов в матрице расстояний
            # и, следовательно, индексам листьев linkage_matrix.
            labels_by_leaf = [result.labels[item.id] for item in items]
            node_colors = self._dendrogram_node_colors(
                result.linkage_matrix,
                labels_by_leaf,
                cluster_colors,
            )

            dendrogram(
                result.linkage_matrix,
                ax=dendro_ax,
                no_labels=True,
                link_color_func=lambda node_id: node_colors.get(
                    int(node_id), "#858585"
                ),
                above_threshold_color="#858585",
            )

            cut_height, has_ties = self._cut_height_for_k(
                result.linkage_matrix,
                object_count,
                selected_k,
            )
            if cut_height is not None:
                dendro_ax.axhline(
                    cut_height,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Срез для K={selected_k}",
                    zorder=5,
                )
                current_top = dendro_ax.get_ylim()[1]
                dendro_ax.set_ylim(
                    top=max(current_top, cut_height * 1.08, 1e-8)
                )
                dendro_ax.legend(fontsize=8, loc="best")

            dendro_ax.set_title(f"Дендрограмма: K={selected_k} кластеров")
            dendro_ax.set_xlabel("Объекты")
            dendro_ax.set_ylabel("Расстояние")

            if has_ties:
                dendro_ax.text(
                    0.02,
                    0.98,
                    "Есть слияния одинаковой высоты.\n"
                    "Цвет ветвей показывает точное разбиение на K.",
                    transform=dendro_ax.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.85,
                        "edgecolor": "none",
                    },
                )
            elif selected_k == object_count:
                dendro_ax.text(
                    0.02,
                    0.98,
                    "Каждый объект образует отдельный кластер.",
                    transform=dendro_ax.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.85,
                        "edgecolor": "none",
                    },
                )
        else:
            message = (
                "В рабочем множестве один объект.\n"
                "Дендрограмма не требуется."
                if object_count == 1
                else "Нет объектов для построения дендрограммы."
            )
            dendro_ax.text(0.5, 0.5, message, ha="center", va="center")
            dendro_ax.set_axis_off()

        by_id = {item.id: item for item in items}
        for cluster_id, item_ids in result.clusters.items():
            cluster_items = [by_id[item_id] for item_id in item_ids]
            color = cluster_colors[cluster_id]
            scatter_ax.scatter(
                [item.x0 for item in cluster_items],
                [item.sigma for item in cluster_items],
                color=color,
                label=f"Кластер {cluster_id} ({len(cluster_items)})",
                s=34,
                alpha=0.82,
            )

            medoid_id = result.medoids[cluster_id]
            medoid = by_id[medoid_id]
            scatter_ax.scatter(
                [medoid.x0],
                [medoid.sigma],
                color=color,
                edgecolors="black",
                marker="X",
                s=105,
                linewidths=1.0,
                zorder=4,
            )

        scatter_ax.set_title(
            f"Параметрическая плоскость (x₀, σ), K={selected_k}"
        )
        scatter_ax.set_xlabel("x₀")
        scatter_ax.set_ylabel("σ")
        scatter_ax.grid(True, alpha=0.25)
        if result.clusters:
            scatter_ax.legend(fontsize=8, loc="best")

        self.figure.suptitle(
            f"Кластеризация при α={result.h:g}: "
            f"{object_count} объектов, {selected_k} кластеров",
            fontsize=11,
        )
        self.figure.tight_layout(rect=(0, 0, 1, 0.95))
        self.canvas.draw_idle()

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

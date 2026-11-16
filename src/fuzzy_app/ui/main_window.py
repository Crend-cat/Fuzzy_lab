"""Главное окно Fuzzy Lab на PySide6."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fuzzy_core.level_sets import level_interval
from fuzzy_core.gaussian import maximum_density
from fuzzy_app.controller import FuzzyLabController
from .generation_dialog import GenerationDialog
from .plot_widgets import ClusterPlotsWidget, DistributionPlotWidget


class NumberEditDialog(QDialog):
    def __init__(self, x0: float = 0.0, sigma: float = 0.3, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Нечёткая величина")
        self.x0_field = QDoubleSpinBox()
        self.x0_field.setRange(-1_000_000_000, 1_000_000_000)
        self.x0_field.setDecimals(10)
        self.x0_field.setValue(x0)
        self.sigma_field = QDoubleSpinBox()
        self.sigma_field.setRange(0.000000001, 1_000_000_000)
        self.sigma_field.setDecimals(10)
        self.sigma_field.setValue(sigma)

        form = QFormLayout(self)
        form.addRow("x₀ (положение максимума)", self.x0_field)
        form.addRow("σ (ширина, σ > 0)", self.sigma_field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[float, float]:
        return self.x0_field.value(), self.sigma_field.value()


class MainWindow(QMainWindow):
    def __init__(self, root_directory: str | Path = ".") -> None:
        super().__init__()
        self.controller = FuzzyLabController(root_directory)
        self.setWindowTitle("Fuzzy Lab — гауссовские нечёткие величины")
        self.resize(1280, 850)
        self._last_selected_id: int | None = None
        self._last_cluster_result = None
        # Флаг остаётся True даже тогда, когда после изменения α для прежнего
        # режима временно не хватает объектов. При последующем изменении α
        # приложение снова попробует выполнить кластеризацию.
        self._clustering_requested = False

        central = QWidget()
        outer = QVBoxLayout(central)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Уровень α (порог плотности):"))
        self.alpha_field = QDoubleSpinBox()
        self.alpha_field.setRange(0.000001, 1_000_000)
        self.alpha_field.setDecimals(6)
        self.alpha_field.setSingleStep(0.05)
        self.alpha_field.setValue(0.7)
        self.alpha_field.setToolTip("Абсолютный порог плотности f(x) ≥ α. Начальное значение — 0.7.")
        self.alpha_field.valueChanged.connect(self._on_alpha_changed)
        top_row.addWidget(self.alpha_field)
        top_row.addStretch(1)
        self.dataset_count_label = QLabel("Элементов: 0")
        top_row.addWidget(self.dataset_count_label)
        outer.addLayout(top_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dataset_tab(), "Множество Ω")
        self.tabs.addTab(self._build_density_tab(), "Плотность")
        self.tabs.addTab(self._build_clustering_tab(), "Кластеризация")
        outer.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Готово. Создайте или загрузите набор данных.")
        self._refresh_all()

    def _build_dataset_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        buttons = QHBoxLayout()
        actions = [
            ("Сгенерировать", self._generate),
            ("Добавить", self._add),
            ("Редактировать", self._edit_selected),
            ("Удалить", self._delete_selected),
            ("Загрузить TSV", self._load),
            ("Сохранить TSV", self._save),
            ("Сохранить след генерации", self._save_generation_trace),
        ]
        for title, callback in actions:
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.dataset_table = QTableWidget(0, 7)
        self.dataset_table.setHorizontalHeaderLabels(["ID", "x₀", "σ", "f_max", "x_left", "x_right", "В Ω_α"])
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dataset_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.dataset_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.dataset_table.horizontalHeader().setStretchLastSection(True)
        self.dataset_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.dataset_table.cellDoubleClicked.connect(lambda _row, _column: self._edit_selected())
        layout.addWidget(self.dataset_table)

        self.dataset_note = QLabel(
            "В исходном наборе хранятся все записи. При изменении α программа пересчитывает "
            "границы и состав рабочего подмножества Ω_α; исключённые элементы не удаляются."
        )
        self.dataset_note.setWordWrap(True)
        layout.addWidget(self.dataset_note)
        return widget

    def _build_density_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.density_plot = DistributionPlotWidget()
        layout.addWidget(self.density_plot, 3)

        self.density_parameters = QLabel("Выберите элемент во вкладке «Множество Ω».")
        self.density_parameters.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.density_parameters.setMinimumWidth(260)
        self.density_parameters.setWordWrap(True)
        self.density_parameters.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.density_parameters, 1)
        return widget

    def _build_clustering_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls = QHBoxLayout()
        self.cluster_mode = QComboBox()
        self.cluster_mode.addItem("Заданное количество K", "fixed")
        self.cluster_mode.addItem("Автоматически по силуэту", "auto")
        self.cluster_mode.currentIndexChanged.connect(self._on_cluster_mode_changed)
        self.cluster_k = QSpinBox()
        self.cluster_k.setRange(1, 300)
        self.cluster_k.setValue(4)
        self.cluster_k.valueChanged.connect(
            lambda value: self._on_cluster_parameters_changed("k", value)
        )
        self.cluster_max_k = QSpinBox()
        self.cluster_max_k.setRange(2, 300)
        self.cluster_max_k.setValue(10)
        self.cluster_max_k.valueChanged.connect(
            lambda value: self._on_cluster_parameters_changed("max_k", value)
        )
        controls.addWidget(QLabel("Режим:"))
        controls.addWidget(self.cluster_mode)
        controls.addWidget(QLabel("K:"))
        controls.addWidget(self.cluster_k)
        controls.addWidget(QLabel("Максимум K:"))
        controls.addWidget(self.cluster_max_k)
        run_button = QPushButton("Запустить кластеризацию")
        run_button.clicked.connect(self._run_clustering)
        save_button = QPushButton("Сохранить результат")
        save_button.clicked.connect(self._save_cluster_result)
        controls.addWidget(run_button)
        controls.addWidget(save_button)
        layout.addLayout(controls)

        self.cluster_summary = QLabel("Кластеризация ещё не запускалась.")
        self.cluster_summary.setWordWrap(True)
        layout.addWidget(self.cluster_summary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.cluster_tree = QTreeWidget()
        self.cluster_tree.setHeaderLabels(["Структура результата", "ID / количество"])
        splitter.addWidget(self.cluster_tree)
        self.cluster_plots = ClusterPlotsWidget()
        splitter.addWidget(self.cluster_plots)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)
        self._update_cluster_controls()
        return widget

    def _update_cluster_controls(self) -> None:
        fixed = self.cluster_mode.currentData() == "fixed"
        self.cluster_k.setEnabled(fixed)
        self.cluster_max_k.setEnabled(not fixed)

    def _on_cluster_mode_changed(self, _index: int = -1) -> None:
        """Обновить доступность параметров и пересчитать уже запущенную кластеризацию."""
        self._update_cluster_controls()
        self._rerun_clustering_if_requested()

    def _on_cluster_parameters_changed(self, parameter: str, value: int) -> None:
        """Зафиксировать изменение K/max K и обновить существующий результат."""
        self.controller.set_cluster_parameter(parameter, value)
        self._rerun_clustering_if_requested()

    def _rerun_clustering_if_requested(self) -> None:
        # До первого нажатия кнопки «Запустить кластеризацию» настройки можно
        # менять свободно: вычисления не запускаются неожиданно. После первого
        # запуска изменение K, max K или режима сразу перестраивает результат.
        if self._clustering_requested:
            self._run_clustering(show_errors=False)

    def _on_alpha_changed(self, value: float) -> None:
        """Обновить Ω_α, графики и уже построенную кластеризацию."""
        try:
            # Запоминаем, запускал ли пользователь кластеризацию ранее.
            # После изменения α старое разбиение теряет актуальность.
            was_clustered = self._clustering_requested
            self.controller.set_level(value)
            self._last_cluster_result = None
            self.cluster_tree.clear()
            self.cluster_plots.clear()
            self._refresh_table()
            self._refresh_density()

            if was_clustered:
                # Повторный запуск выполняется автоматически и без всплывающего
                # окна при каждом изменении α. Если объектов стало недостаточно,
                # причину показываем в сводке вкладки кластеризации.
                self._run_clustering(show_errors=False)
            else:
                self.cluster_summary.setText(
                    "α-уровень изменён. Состав Ω_α и график N(α) обновлены. "
                    "Запустите кластеризацию, чтобы построить разбиение для нового уровня."
                )
            self.statusBar().showMessage(f"α-уровень установлен: {value:g}")
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка α-уровня", str(exc))

    def _refresh_all(self) -> None:
        self._refresh_table()
        self._refresh_density()

    def _invalidate_cluster_view(self) -> None:
        """Очистить результат, если изменился сам исходный набор."""
        self._last_cluster_result = None
        self._clustering_requested = False
        self.cluster_tree.clear()
        self.cluster_plots.clear()
        self.cluster_summary.setText("Исходный набор изменён. Выполните кластеризацию заново.")

    def _refresh_table(self) -> None:
        items = self.controller.dataset.items
        self.dataset_table.setRowCount(len(items))
        selected_row = None
        included_count = 0

        for row, item in enumerate(items):
            interval = level_interval(item, self.controller.h)
            if interval is not None:
                included_count += 1
            values = [
                str(item.id),
                f"{item.x0:.8g}",
                f"{item.sigma:.8g}",
                f"{maximum_density(item.sigma):.8g}",
                f"{interval.left:.8g}" if interval else "—",
                f"{interval.right:.8g}" if interval else "—",
                "Да" if interval else "Нет",
            ]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.dataset_table.setItem(row, column, cell)
            if item.id == self._last_selected_id:
                selected_row = row

        self.dataset_table.resizeColumnsToContents()
        self.dataset_count_label.setText(
            f"Исходно: {len(items)}   |   В Ω_α: {included_count}   |   Исключено: {len(items) - included_count}"
        )
        if selected_row is not None:
            self.dataset_table.selectRow(selected_row)
        elif items and self._last_selected_id is None:
            self.dataset_table.selectRow(0)
            self._last_selected_id = items[0].id

    def _refresh_density(self) -> None:
        if not hasattr(self, "density_plot"):
            return

        items = self.controller.dataset.items
        alpha = self.controller.h
        if not items:
            self.density_plot.clear()
            self.density_parameters.setText("Набор пуст. Создайте или загрузите данные.")
            return

        if self._last_selected_id is None:
            self._last_selected_id = items[0].id

        try:
            item = self.controller.dataset.get(self._last_selected_id)
        except KeyError:
            self._last_selected_id = items[0].id
            item = items[0]

        # Обновляются графики выбранной функции, всех функций исходного
        # множества и ступенчатая зависимость количества N(α).
        self.density_plot.show_number(item, items, alpha)
        interval = level_interval(item, alpha)
        included_count = sum(level_interval(candidate, alpha) is not None for candidate in items)
        text = (
            f"<b>Нечёткая величина A{item.id}</b><br><br>"
            f"x₀ = {item.x0:.10g}<br>"
            f"σ = {item.sigma:.10g}<br>"
            f"Дисперсия σ² = {item.sigma**2:.10g}<br>"
            f"f_max = {maximum_density(item.sigma):.10g}<br>"
            f"Текущий уровень α = {alpha:.10g}<br>"
            f"В исходном множестве: {len(items)}<br>"
            f"В Ω_α: {included_count}<br><br>"
        )
        if interval is None:
            text += "<b>Пересечений нет.</b><br>Максимум плотности ниже текущего α."
        else:
            text += (
                f"x_left = {interval.left:.10g}<br>"
                f"x_right = {interval.right:.10g}<br>"
                f"Ширина интервала = {interval.right - interval.left:.10g}<br>"
                "<br>Элемент входит в Ω_α."
            )
        self.density_parameters.setText(text)

    def _selected_id(self) -> int | None:
        row = self.dataset_table.currentRow()
        if row < 0:
            return None
        cell = self.dataset_table.item(row, 0)
        return int(cell.text()) if cell else None

    def _on_selection_changed(self) -> None:
        selected = self._selected_id()
        if selected is not None:
            self._last_selected_id = selected
            self._refresh_density()

    def _generate(self) -> None:
        dialog = GenerationDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            config = dialog.generation_config()
            self.controller.generate(config)
            self._last_selected_id = self.controller.dataset.items[0].id
            saved = self.controller.save_generation()
            self._invalidate_cluster_view()
            self._refresh_all()
            self.statusBar().showMessage(f"Сгенерировано {len(self.controller.dataset)} величин; след генерации: {saved[2]}")
            QMessageBox.information(
                self,
                "Генерация завершена",
                f"Создано элементов: {len(self.controller.dataset)}\n\n"
                f"Набор: {saved[0]}\nМетаданные: {saved[1]}\nТрасса генерации: {saved[2]}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка генерации", str(exc))

    def _add(self) -> None:
        dialog = NumberEditDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            item = self.controller.add(*dialog.values())
            self._last_selected_id = item.id
            self._invalidate_cluster_view()
            self._refresh_all()
            self.statusBar().showMessage(f"Добавлена величина A{item.id}")
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка добавления", str(exc))

    def _edit_selected(self) -> None:
        item_id = self._selected_id()
        if item_id is None:
            QMessageBox.information(self, "Редактирование", "Сначала выберите строку таблицы.")
            return
        item = self.controller.dataset.get(item_id)
        dialog = NumberEditDialog(item.x0, item.sigma, self)
        dialog.setWindowTitle(f"Редактирование A{item_id}")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.controller.edit(item_id, *dialog.values())
            self._last_selected_id = item_id
            self._invalidate_cluster_view()
            self._refresh_all()
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка редактирования", str(exc))

    def _delete_selected(self) -> None:
        item_id = self._selected_id()
        if item_id is None:
            return
        answer = QMessageBox.question(self, "Удаление", f"Удалить A{item_id} из исходного набора?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.controller.delete(item_id)
        self._last_selected_id = self.controller.dataset.items[0].id if self.controller.dataset.items else None
        self._invalidate_cluster_view()
        self._refresh_all()

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить набор TSV", str(Path.cwd()), "TSV (*.tsv);;Все файлы (*)")
        if not path:
            return
        try:
            self.controller.load(path)
            self._last_selected_id = self.controller.dataset.items[0].id if self.controller.dataset.items else None
            self._invalidate_cluster_view()
            self._refresh_all()
            self.statusBar().showMessage(f"Загружено: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить набор TSV", str(Path.cwd() / "dataset.tsv"), "TSV (*.tsv)")
        if not path:
            return
        try:
            if not path.lower().endswith(".tsv"):
                path += ".tsv"
            self.controller.save(path)
            QMessageBox.information(self, "Сохранение", f"Набор сохранён:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def _save_generation_trace(self) -> None:
        try:
            saved = self.controller.save_generation()
            QMessageBox.information(self, "След генерации", f"Сохранены файлы:\n{saved[0]}\n{saved[1]}\n{saved[2]}")
        except Exception as exc:
            QMessageBox.warning(self, "Нет данных генерации", str(exc))

    def _run_clustering(self, show_errors: bool = True) -> None:
        # Запоминаем намерение пользователя. Флаг нужен, чтобы повторять анализ
        # при последующих изменениях α даже после промежуточной неудачи.
        self._clustering_requested = True
        try:
            mode = self.cluster_mode.currentData()
            result = self.controller.cluster(mode, self.cluster_k.value(), self.cluster_max_k.value())
            self._last_cluster_result = result
            self.cluster_plots.show_result(result)
            self.cluster_tree.clear()

            root = QTreeWidgetItem([
                f"Ω_α (α={result.h:g})",
                f"{len(result.included_items)} включено / {len(result.excluded_items)} исключено",
            ])
            self.cluster_tree.addTopLevelItem(root)
            by_id = {item.id: item for item in result.included_items}

            for cluster_id, item_ids in result.clusters.items():
                medoid = result.medoids[cluster_id]
                cluster_node = QTreeWidgetItem(
                    [f"Кластер {cluster_id} (медоид A{medoid})", str(len(item_ids))]
                )
                root.addChild(cluster_node)
                for item_id in item_ids:
                    item = by_id[item_id]
                    cluster_node.addChild(
                        QTreeWidgetItem([
                            f"A{item.id}: x₀={item.x0:.6g}, σ={item.sigma:.6g}",
                            str(item.id),
                        ])
                    )

            if result.excluded_items:
                excluded_node = QTreeWidgetItem([
                    "Исключены уровнем α",
                    str(len(result.excluded_items)),
                ])
                root.addChild(excluded_node)
                for item in result.excluded_items:
                    excluded_node.addChild(
                        QTreeWidgetItem([
                            f"A{item.id}: f_max={maximum_density(item.sigma):.6g} < α",
                            str(item.id),
                        ])
                    )

            root.setExpanded(True)
            for index in range(root.childCount()):
                root.child(index).setExpanded(True)

            summary = (
                f"Алгоритм: Average Linkage; метрика: Хаусдорф между α-сечениями; "
                f"α={result.h:g}; K={result.selected_k}; "
                f"включено={len(result.included_items)}; "
                f"исключено={len(result.excluded_items)}."
            )
            if result.silhouette_score is not None:
                summary += f" Средний силуэт={result.silhouette_score:.6f}."
            self.cluster_summary.setText(summary)
            self.statusBar().showMessage("Кластеризация завершена для текущего α-уровня")
        except Exception as exc:
            self._last_cluster_result = None
            self.cluster_tree.clear()
            self.cluster_plots.clear()
            message = (
                f"Для текущего α={self.controller.h:g} не удалось выполнить кластеризацию: {exc}. "
                "Измените α или параметры кластеризации."
            )
            self.cluster_summary.setText(message)
            if show_errors:
                QMessageBox.critical(self, "Ошибка кластеризации", str(exc))
            else:
                self.statusBar().showMessage("Кластеризация не выполнена для нового α-уровня")

    def _save_cluster_result(self) -> None:
        try:
            path = self.controller.save_clustering()
            QMessageBox.information(self, "Результат сохранён", f"Файлы результата сохранены в:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Сохранение результата", str(exc))

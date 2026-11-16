"""Диалог настройки генерации множества."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from fuzzy_core.normal_generator import GenerationConfig


class GenerationDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Генерация нечётких величин")
        self.resize(430, 500)

        self.count = QSpinBox()
        self.count.setRange(100, 300)
        self.count.setValue(150)

        self.center_mean = self._double(-1_000_000, 1_000_000, 0.0, 6)
        self.center_std = self._double(0.000001, 1_000_000, 1.0, 6)
        self.n = QSpinBox()
        self.n.setRange(2, 10000)
        self.n.setValue(8)

        self.approximation = QComboBox()
        self.approximation.addItem("Формула 1 (n ≥ 5, знак минус)", "formula_1")
        self.approximation.addItem("Формула 2 (n ≥ 2)", "formula_2")
        self.approximation.currentIndexChanged.connect(self._update_n_minimum)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(42)

        self.sigma_mode = QComboBox()
        self.sigma_mode.addItem("Автоматически: Uniform", "uniform")
        self.sigma_mode.addItem("Одна ширина для всех", "fixed")
        self.sigma_mode.currentIndexChanged.connect(self._update_sigma_fields)

        self.sigma_min = self._double(0.000001, 1_000_000, 0.15, 6)
        self.sigma_max = self._double(0.000002, 1_000_000, 0.45, 6)
        self.fixed_sigma = self._double(0.000001, 1_000_000, 0.3, 6)

        generation_group = QGroupBox("Центры функций: x₀ᵢ = center_mean + center_std · ηᵢ")
        generation_form = QFormLayout(generation_group)
        generation_form.addRow("Количество (100–300)", self.count)
        generation_form.addRow("center_mean", self.center_mean)
        generation_form.addRow("center_std", self.center_std)
        generation_form.addRow("n", self.n)
        generation_form.addRow("Аппроксимация", self.approximation)
        generation_form.addRow("Seed", self.seed)

        sigma_group = QGroupBox("Ширина каждой нечёткой величины σᵢ")
        sigma_form = QFormLayout(sigma_group)
        sigma_form.addRow("Режим", self.sigma_mode)
        sigma_form.addRow("sigma_min (Uniform)", self.sigma_min)
        sigma_form.addRow("sigma_max (Uniform)", self.sigma_max)
        sigma_form.addRow("Единая sigma", self.fixed_sigma)

        note = QGroupBox("Пояснение")
        note_layout = QVBoxLayout(note)
        from PySide6.QtWidgets import QLabel
        label = QLabel(
            "Центры генерируются по формулам задания. В автоматическом режиме "
            "ширина независимо выбирается из равномерного закона на заданном "
            "положительном интервале. Это проектное допущение: отдельный закон "
            "для ширины в задании не указан. Значения по умолчанию 0.15–0.45 "
            "обеспечивают пересечение с h=0.7."
        )
        label.setWordWrap(True)
        note_layout.addWidget(label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(generation_group)
        layout.addWidget(sigma_group)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self._update_n_minimum()
        self._update_sigma_fields()

    @staticmethod
    def _double(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(decimals)
        field.setValue(value)
        field.setKeyboardTracking(False)
        return field

    def _update_n_minimum(self) -> None:
        minimum = 5 if self.approximation.currentData() == "formula_1" else 2
        self.n.setMinimum(minimum)
        if self.n.value() < minimum:
            self.n.setValue(minimum)

    def _update_sigma_fields(self) -> None:
        automatic = self.sigma_mode.currentData() == "uniform"
        self.sigma_min.setEnabled(automatic)
        self.sigma_max.setEnabled(automatic)
        self.fixed_sigma.setEnabled(not automatic)

    def _validate_and_accept(self) -> None:
        if self.sigma_mode.currentData() == "uniform" and self.sigma_max.value() <= self.sigma_min.value():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Некорректный диапазон", "Требуется sigma_max > sigma_min.")
            return
        self.accept()

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            count=self.count.value(),
            center_mean=self.center_mean.value(),
            center_std=self.center_std.value(),
            n=self.n.value(),
            approximation=self.approximation.currentData(),
            seed=self.seed.value(),
            sigma_mode=self.sigma_mode.currentData(),
            sigma_min=self.sigma_min.value(),
            sigma_max=self.sigma_max.value(),
            fixed_sigma=self.fixed_sigma.value(),
        )

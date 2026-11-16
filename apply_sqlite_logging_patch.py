"""Подключить минимальное SQLite-журналирование к Fuzzy Lab."""

from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "src" / "fuzzy_app" / "controller.py"
MAIN_WINDOW = ROOT / "src" / "fuzzy_app" / "ui" / "main_window.py"
GITIGNORE = ROOT / ".gitignore"


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".before_sqlite_logging.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Не удалось однозначно изменить {description}: найдено совпадений {count}"
        )
    return text.replace(old, new, 1)


def patch_controller() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    if "from fuzzy_storage.audit_logger import AuditRepository" in text:
        raise RuntimeError("SQLite уже подключена в controller.py; повторное применение отменено")

    backup(CONTROLLER)

    import_anchor = "from fuzzy_storage import (\n"
    text = replace_once(
        text,
        import_anchor,
        "from fuzzy_storage.audit_logger import AuditRepository\n"
        "from fuzzy_storage import (\n",
        "импорт AuditRepository",
    )

    init_anchor = "        self.last_cluster_result: ClusterResult | None = None\n"
    init_replacement = (
        init_anchor
        + "\n"
        + "        # База и таблица создаются автоматически при первом запуске.\n"
        + "        self.audit = AuditRepository(self.root / \"logs\" / \"audit.sqlite3\")\n"
        + "        self._cluster_parameters = {\"k\": 4, \"max_k\": 10}\n"
    )
    text = replace_once(text, init_anchor, init_replacement, "инициализация SQLite")

    method_start = text.find("    def _record_event(\n")
    method_end = text.find("    def record_action(\n", method_start)
    if method_start < 0 or method_end < 0:
        raise RuntimeError("Не найдены ожидаемые методы _record_event и record_action в controller.py")

    replacement = '''    def _record_event(
        self,
        event_type: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Записывает только события, относящиеся к заданию лабораторной."""
        normalized_type = event_type.strip().upper()
        tracked_types = {
            "GENERATE_DATASET",
            "ADD_NUMBER",
            "CHANGE_ALPHA",
            "CHANGE_CLUSTER_COUNT",
            "CLUSTER_DATASET",
        }
        if normalized_type not in tracked_types:
            return 0

        event_details = dict(details or {})
        included_count = sum(
            level_interval(item, self.h) is not None
            for item in self.dataset.items
        )

        if normalized_type in {"GENERATE_DATASET", "ADD_NUMBER"}:
            event_details.update(
                {
                    "alpha": float(self.h),
                    "included_count": int(included_count),
                    "excluded_count": int(len(self.dataset) - included_count),
                    "dataset_count": int(len(self.dataset)),
                }
            )

        elif normalized_type == "CLUSTER_DATASET":
            clusters = event_details.get("clusters", {})
            cluster_count = len(clusters) if isinstance(clusters, dict) else 0
            if not cluster_count:
                cluster_count = int(event_details.get("selected_k") or 0)
            event_details = {
                "dataset_name": event_details.get("dataset_name", self.dataset.name),
                "mode": event_details.get("mode"),
                "alpha": float(event_details.get("alpha", self.h)),
                "cluster_count": int(cluster_count),
                "requested_k": event_details.get("requested_k"),
                "max_k": event_details.get("max_k"),
                "selected_k": event_details.get("selected_k"),
                "included_count": int(event_details.get("included_count", 0)),
                "excluded_count": int(event_details.get("excluded_count", 0)),
            }

        return self.audit.record_event(normalized_type, summary, event_details)

'''
    text = text[:method_start] + replacement + text[method_end:]

    generate_anchor = "    def generate(self, config: GenerationConfig) -> FuzzyDataset:\n"
    if text.count(generate_anchor) != 1:
        raise RuntimeError("Не найден метод generate()")

    cluster_method_anchor = "    def cluster(\n"
    cluster_change_method = '''    def set_cluster_parameter(self, parameter: str, value: int) -> None:
        """Отдельно фиксирует изменение K или максимального K."""
        if parameter not in self._cluster_parameters:
            raise ValueError("Параметр должен быть 'k' или 'max_k'")
        new_value = int(value)
        if new_value < 1:
            raise ValueError("Количество кластеров должно быть положительным")

        previous_value = self._cluster_parameters[parameter]
        if previous_value == new_value:
            return

        self._cluster_parameters[parameter] = new_value
        parameter_title = "K" if parameter == "k" else "Максимум K"
        self._record_event(
            "CHANGE_CLUSTER_COUNT",
            f"Изменён параметр {parameter_title}: {previous_value} → {new_value}",
            {
                "parameter": parameter,
                "previous_value": int(previous_value),
                "new_value": new_value,
                "alpha": float(self.h),
                "dataset_count": int(len(self.dataset)),
            },
        )

'''
    text = replace_once(text, cluster_method_anchor, cluster_change_method + cluster_method_anchor, "метод фиксации изменений K")

    CONTROLLER.write_text(text, encoding="utf-8")


def patch_main_window() -> None:
    text = MAIN_WINDOW.read_text(encoding="utf-8")
    if 'self._on_cluster_parameters_changed("k", value)' in text:
        raise RuntimeError("Обработчики K уже изменены; повторное применение отменено")

    backup(MAIN_WINDOW)
    text = replace_once(
        text,
        "        self.cluster_k.valueChanged.connect(self._on_cluster_parameters_changed)\n",
        "        self.cluster_k.valueChanged.connect(\n"
        "            lambda value: self._on_cluster_parameters_changed(\"k\", value)\n"
        "        )\n",
        "обработчик изменения K",
    )
    text = replace_once(
        text,
        "        self.cluster_max_k.valueChanged.connect(self._on_cluster_parameters_changed)\n",
        "        self.cluster_max_k.valueChanged.connect(\n"
        "            lambda value: self._on_cluster_parameters_changed(\"max_k\", value)\n"
        "        )\n",
        "обработчик изменения максимального K",
    )
    text = replace_once(
        text,
        "    def _on_cluster_parameters_changed(self, _value: int = 0) -> None:\n"
        "        \"\"\"Автоматически пересчитать результат после изменения K или max K.\"\"\"\n"
        "        self._rerun_clustering_if_requested()\n",
        "    def _on_cluster_parameters_changed(self, parameter: str, value: int) -> None:\n"
        "        \"\"\"Зафиксировать изменение K/max K и обновить существующий результат.\"\"\"\n"
        "        self.controller.set_cluster_parameter(parameter, value)\n"
        "        self._rerun_clustering_if_requested()\n",
        "метод обработки изменения K",
    )
    MAIN_WINDOW.write_text(text, encoding="utf-8")


def patch_gitignore() -> None:
    if GITIGNORE.exists():
        text = GITIGNORE.read_text(encoding="utf-8")
    else:
        text = ""
    additions = [
        "# Локальная база и журналы приложения",
        "logs/*.sqlite3",
        "logs/*.sqlite3-*",
        "logs/*.db",
        "logs/*.db-*",
        "logs/*.log",
        "logs/*.jsonl",
        "*.before_sqlite_logging.bak",
    ]
    missing = [line for line in additions if line not in text.splitlines()]
    if missing:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n".join(missing) + "\n"
        if GITIGNORE.exists():
            backup(GITIGNORE)
        GITIGNORE.write_text(text, encoding="utf-8")


def patch_readme() -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if "logs/audit.sqlite3" in text:
        return
    section = (
        "\n\n## Журнал действий\n\n"
        "При первом запуске приложение автоматически создаёт SQLite-базу "
        "`logs/audit.sqlite3`. В ней сохраняются генерация и количество "
        "величин, ручное добавление, изменения уровня α, изменения параметров "
        "количества кластеров и результаты кластеризации. Сам файл базы "
        "является локальным и не нужен для запуска после клонирования: "
        "он создаётся автоматически.\n"
    )
    backup(readme)
    readme.write_text(text.rstrip() + section, encoding="utf-8")


def main() -> None:
    required = [CONTROLLER, MAIN_WINDOW]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Не найдены файлы проекта: " + ", ".join(missing))
    if not (ROOT / "src" / "fuzzy_storage" / "audit_logger.py").is_file():
        raise SystemExit("Не найден src/fuzzy_storage/audit_logger.py из архива патча")

    patch_controller()
    patch_main_window()
    patch_gitignore()
    patch_readme()
    print("SQLite-журналирование подключено.")
    print("База будет создаваться автоматически: logs/audit.sqlite3")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import csv
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "beto_incremental_naive"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "beto_incremental_naive"

INCREMENTAL_RESULTS_PATH = INPUT_DIR / "incremental_results.csv"
FORGETTING_SUMMARY_PATH = INPUT_DIR / "forgetting_summary.csv"

TASKS = [
    {"task_id": 1, "task_name": "illegal"},
    {"task_id": 2, "task_name": "dark"},
    {"task_id": 3, "task_name": "gray"},
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    return float(value)


def format_float(value: float | str) -> str:
    if value == "":
        return ""

    return f"{float(value):.4f}"


def build_metric_matrix(
    results_rows: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    """
    Construye una tabla triangular.

    Filas:
    - Después de entrenar Task 1
    - Después de entrenar Task 2
    - Después de entrenar Task 3

    Columnas:
    - Evaluación Task 1
    - Evaluación Task 2
    - Evaluación Task 3
    """
    matrix_rows = []

    for after_task in TASKS:
        after_task_id = after_task["task_id"]
        after_task_name = after_task["task_name"]

        row = {
            "after_training": f"Task {after_task_id} ({after_task_name})"
        }

        for eval_task in TASKS:
            eval_task_id = eval_task["task_id"]
            eval_task_name = eval_task["task_name"]

            matching_rows = [
                item for item in results_rows
                if int(item["after_training_task"]) == after_task_id
                and int(item["eval_task_id"]) == eval_task_id
            ]

            column_name = f"eval_task_{eval_task_id}_{eval_task_name}"

            if matching_rows:
                row[column_name] = format_float(matching_rows[0][metric])
            else:
                row[column_name] = ""

        matrix_rows.append(row)

    return matrix_rows


def build_retention_rows(
    results_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Resume cuánto se mantiene o pierde por tarea al final del entrenamiento.
    """
    retention_rows = []

    final_task_id = TASKS[-1]["task_id"]

    for task in TASKS:
        task_id = task["task_id"]
        task_name = task["task_name"]

        first_rows = [
            row for row in results_rows
            if int(row["after_training_task"]) == task_id
            and int(row["eval_task_id"]) == task_id
        ]

        final_rows = [
            row for row in results_rows
            if int(row["after_training_task"]) == final_task_id
            and int(row["eval_task_id"]) == task_id
        ]

        if not first_rows or not final_rows:
            continue

        first_row = first_rows[0]
        final_row = final_rows[0]

        initial_micro = to_float(first_row["micro_f1"])
        final_micro = to_float(final_row["micro_f1"])
        initial_macro = to_float(first_row["macro_f1"])
        final_macro = to_float(final_row["macro_f1"])

        forgetting_match = [
            row for row in forgetting_rows
            if int(row["eval_task_id"]) == task_id
        ]

        if forgetting_match:
            micro_forgetting = to_float(forgetting_match[0]["micro_forgetting"])
            macro_forgetting = to_float(forgetting_match[0]["macro_forgetting"])
        else:
            micro_forgetting = max(0.0, initial_micro - final_micro)
            macro_forgetting = max(0.0, initial_macro - final_macro)

        micro_relative_drop = (
            ((initial_micro - final_micro) / initial_micro) * 100
            if initial_micro > 0
            else 0.0
        )

        macro_relative_drop = (
            ((initial_macro - final_macro) / initial_macro) * 100
            if initial_macro > 0
            else 0.0
        )

        retention_rows.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "initial_micro_f1": format_float(initial_micro),
                "final_micro_f1": format_float(final_micro),
                "micro_absolute_drop": format_float(initial_micro - final_micro),
                "micro_relative_drop_percent": format_float(micro_relative_drop),
                "micro_forgetting_best_to_final": format_float(micro_forgetting),
                "initial_macro_f1": format_float(initial_macro),
                "final_macro_f1": format_float(final_macro),
                "macro_absolute_drop": format_float(initial_macro - final_macro),
                "macro_relative_drop_percent": format_float(macro_relative_drop),
                "macro_forgetting_best_to_final": format_float(macro_forgetting),
            }
        )

    return retention_rows


def forgetting_level(value: float) -> str:
    if value >= 0.25:
        return "alto"
    if value >= 0.10:
        return "moderado"
    if value > 0.02:
        return "bajo"
    return "sin olvido relevante"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    row_lines = [
        "| " + " | ".join(row) + " |"
        for row in rows
    ]

    return "\n".join([header_line, separator_line] + row_lines)


def matrix_to_markdown(title: str, matrix_rows: list[dict[str, Any]]) -> str:
    headers = list(matrix_rows[0].keys())

    rows = [
        [str(row[header]) for header in headers]
        for row in matrix_rows
    ]

    return f"## {title}\n\n{markdown_table(headers, rows)}\n"


def retention_to_markdown(retention_rows: list[dict[str, Any]]) -> str:
    headers = [
        "task",
        "initial_micro_f1",
        "final_micro_f1",
        "micro_drop",
        "micro_forgetting",
        "initial_macro_f1",
        "final_macro_f1",
        "macro_drop",
        "macro_forgetting",
    ]

    rows = []

    for row in retention_rows:
        rows.append(
            [
                f"Task {row['task_id']} ({row['task_name']})",
                row["initial_micro_f1"],
                row["final_micro_f1"],
                row["micro_absolute_drop"],
                row["micro_forgetting_best_to_final"],
                row["initial_macro_f1"],
                row["final_macro_f1"],
                row["macro_absolute_drop"],
                row["macro_forgetting_best_to_final"],
            ]
        )

    return f"## Retención y olvido final\n\n{markdown_table(headers, rows)}\n"


def build_report(
    micro_matrix: list[dict[str, Any]],
    macro_matrix: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
) -> str:
    lines = []

    lines.append("# Análisis del baseline incremental BETO")
    lines.append("")
    lines.append(
        "Este reporte resume el comportamiento del baseline secuencial ingenuo. "
        "El modelo se entrena en una tarea a la vez y luego se evalúa acumulativamente "
        "sobre las tareas vistas hasta ese momento."
    )
    lines.append("")

    lines.append(matrix_to_markdown("Matriz de micro-F1", micro_matrix))
    lines.append("")
    lines.append(matrix_to_markdown("Matriz de macro-F1", macro_matrix))
    lines.append("")
    lines.append(retention_to_markdown(retention_rows))
    lines.append("")

    lines.append("## Lectura automática")
    lines.append("")

    sorted_by_forgetting = sorted(
        retention_rows,
        key=lambda row: float(row["micro_forgetting_best_to_final"]),
        reverse=True,
    )

    for row in sorted_by_forgetting:
        task_label = f"Task {row['task_id']} ({row['task_name']})"
        micro_forgetting = float(row["micro_forgetting_best_to_final"])
        macro_forgetting = float(row["macro_forgetting_best_to_final"])

        lines.append(
            f"- {task_label}: olvido micro-F1 {forgetting_level(micro_forgetting)} "
            f"({micro_forgetting:.4f}) y olvido macro-F1 "
            f"{forgetting_level(macro_forgetting)} ({macro_forgetting:.4f})."
        )

    lines.append("")
    lines.append(
        "Interpretación preliminar: si una tarea antigua baja fuertemente después "
        "de entrenar tareas nuevas, el baseline muestra evidencia de olvido catastrófico. "
        "Este resultado sirve como referencia para comparar posteriormente contra "
        "Nested Learning, EWC y replay."
    )

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_rows = read_csv(INCREMENTAL_RESULTS_PATH)
    forgetting_rows = read_csv(FORGETTING_SUMMARY_PATH)

    micro_matrix = build_metric_matrix(results_rows, metric="micro_f1")
    macro_matrix = build_metric_matrix(results_rows, metric="macro_f1")

    retention_rows = build_retention_rows(
        results_rows=results_rows,
        forgetting_rows=forgetting_rows,
    )

    micro_matrix_path = OUTPUT_DIR / "micro_f1_matrix.csv"
    macro_matrix_path = OUTPUT_DIR / "macro_f1_matrix.csv"
    retention_path = OUTPUT_DIR / "retention_forgetting_summary.csv"
    report_path = OUTPUT_DIR / "incremental_analysis_report.md"

    write_csv(micro_matrix_path, micro_matrix)
    write_csv(macro_matrix_path, macro_matrix)
    write_csv(retention_path, retention_rows)

    report = build_report(
        micro_matrix=micro_matrix,
        macro_matrix=macro_matrix,
        retention_rows=retention_rows,
    )

    with report_path.open("w", encoding="utf-8") as file:
        file.write(report)

    print("Análisis incremental generado correctamente.")
    print(f"Matriz micro-F1: {micro_matrix_path}")
    print(f"Matriz macro-F1: {macro_matrix_path}")
    print(f"Resumen retención/olvido: {retention_path}")
    print(f"Reporte Markdown: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
from pathlib import Path
import json
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCREMENTAL_DIR = PROJECT_ROOT / "data" / "processed" / "incremental"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Error leyendo JSON en {path}, línea {line_number}: {error}"
                )

    return records


def load_label_map() -> list[dict[str, Any]]:
    path = INCREMENTAL_DIR / "label_map.json"

    if not path.exists():
        raise FileNotFoundError(
            "No se encontró label_map.json. "
            "Primero ejecuta src/data/prepare_incremental_dataset.py"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_task_config() -> dict[str, Any]:
    path = INCREMENTAL_DIR / "task_config.json"

    if not path.exists():
        raise FileNotFoundError(
            "No se encontró task_config.json. "
            "Primero ejecuta src/data/prepare_incremental_dataset.py"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_num_labels() -> int:
    label_map = load_label_map()
    return len(label_map)


def get_task_dir(task_id: int, task_name: str | None = None) -> Path:
    if task_name is not None:
        task_dir = INCREMENTAL_DIR / f"task_{task_id}_{task_name}"

        if task_dir.exists():
            return task_dir

    matches = list(INCREMENTAL_DIR.glob(f"task_{task_id}_*"))

    if not matches:
        raise FileNotFoundError(f"No se encontró carpeta para task_id={task_id}")

    if len(matches) > 1:
        raise ValueError(f"Hay más de una carpeta para task_id={task_id}: {matches}")

    return matches[0]


def load_task(task_id: int, split: str, task_name: str | None = None) -> list[dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split debe ser 'train', 'val' o 'test'.")

    task_dir = get_task_dir(task_id=task_id, task_name=task_name)
    path = task_dir / f"{split}.jsonl"

    return read_jsonl(path)


def load_tasks(task_ids: list[int], split: str) -> list[dict[str, Any]]:
    all_records = []

    for task_id in task_ids:
        records = load_task(task_id=task_id, split=split)
        all_records.extend(records)

    return all_records


def labels_to_multihot(labels: list[int], num_labels: int | None = None) -> list[int]:
    if num_labels is None:
        num_labels = get_num_labels()

    vector = [0] * num_labels

    for label in labels:
        if label < 0 or label >= num_labels:
            raise ValueError(
                f"Etiqueta fuera de rango: {label}. "
                f"El número total de etiquetas es {num_labels}."
            )

        vector[label] = 1

    return vector


def add_multihot_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    num_labels = get_num_labels()
    processed_records = []

    for record in records:
        new_record = record.copy()
        new_record["multihot_labels"] = labels_to_multihot(
            labels=record["labels"],
            num_labels=num_labels,
        )
        processed_records.append(new_record)

    return processed_records


def build_cumulative_eval_sets(current_task_id: int, split: str = "test") -> dict[int, list[dict[str, Any]]]:
    """
    Crea los conjuntos de evaluación acumulativa.

    Ejemplo:
    current_task_id = 1 -> evalúa task 1
    current_task_id = 2 -> evalúa task 1 y task 2
    current_task_id = 3 -> evalúa task 1, task 2 y task 3
    """
    if current_task_id not in {1, 2, 3}:
        raise ValueError("current_task_id debe ser 1, 2 o 3.")

    eval_sets = {}

    for task_id in range(1, current_task_id + 1):
        eval_sets[task_id] = add_multihot_labels(
            load_task(task_id=task_id, split=split)
        )

    return eval_sets


def print_dataset_check() -> None:
    label_map = load_label_map()
    task_config = load_task_config()

    print("Dataset incremental cargado correctamente.")
    print(f"Ruta: {INCREMENTAL_DIR}")
    print(f"Número total de etiquetas: {len(label_map)}")
    print(f"Escenario: {task_config['scenario']}")
    print()

    for task in task_config["task_order"]:
        task_id = task["task_id"]
        task_name = task["task_name"]

        print(f"Task {task_id}: {task_name}")

        for split in ["train", "val", "test"]:
            records = load_task(task_id=task_id, task_name=task_name, split=split)
            records = add_multihot_labels(records)

            first_record = records[0]

            print(
                f"  {split}: {len(records)} muestras | "
                f"labels ejemplo: {first_record['labels']} | "
                f"multihot size: {len(first_record['multihot_labels'])}"
            )

        print()


if __name__ == "__main__":
    print_dataset_check()
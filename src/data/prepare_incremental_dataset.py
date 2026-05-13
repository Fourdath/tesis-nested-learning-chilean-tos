from pathlib import Path
import json
import csv
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CLASSIFICATION_DIR = PROJECT_ROOT / "data" / "raw" / "tos_chile_original" / "classification"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "incremental"

TASKS = [
    {
        "task_id": 1,
        "task_name": "illegal",
        "source_prefix": "Illegal",
        "description": "Cláusulas ilegales",
    },
    {
        "task_id": 2,
        "task_name": "dark",
        "source_prefix": "Dark",
        "description": "Cláusulas oscuras",
    },
    {
        "task_id": 3,
        "task_name": "gray",
        "source_prefix": "Gray",
        "description": "Cláusulas grises",
    },
]

SPLITS = ["train", "val", "test"]


def read_jsonl(path: Path) -> list[dict]:
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
                raise ValueError(f"Error leyendo JSON en {path}, línea {line_number}: {error}")

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_global_labels() -> tuple[dict[int, int], dict[int, str]]:
    """
    Construye un mapa global de etiquetas a partir de labels_original.
    Esto permite tener etiquetas consistentes entre Illegal, Dark y Gray.
    """
    original_label_ids = set()
    original_label_names = {}

    for task in TASKS:
        for split in SPLITS:
            input_path = RAW_CLASSIFICATION_DIR / f"{task['source_prefix']}_{split}.jsonl"
            records = read_jsonl(input_path)

            for record in records:
                label_ids = record.get("labels_original", record.get("labels", []))
                label_names = record.get("human_readable_labels", record.get("original_labels", []))

                for label_id, label_name in zip(label_ids, label_names):
                    label_id = int(label_id)
                    original_label_ids.add(label_id)
                    original_label_names[label_id] = label_name

    sorted_label_ids = sorted(original_label_ids)
    label_id_to_incremental_id = {
        original_id: incremental_id
        for incremental_id, original_id in enumerate(sorted_label_ids)
    }

    return label_id_to_incremental_id, original_label_names


def transform_record(
    record: dict,
    task: dict,
    split: str,
    label_id_to_incremental_id: dict[int, int],
) -> dict:
    original_label_ids = [
        int(label_id)
        for label_id in record.get("labels_original", record.get("labels", []))
    ]

    incremental_labels = [
        label_id_to_incremental_id[label_id]
        for label_id in original_label_ids
    ]

    return {
        "text": record["text"],
        "labels": incremental_labels,
        "labels_original": original_label_ids,
        "human_readable_labels": record.get("human_readable_labels", []),
        "original_labels": record.get("original_labels", []),
        "annotations": record.get("annotations", ""),
        "split": split,
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "source_group": task["source_prefix"],
    }


def write_label_map(
    label_id_to_incremental_id: dict[int, int],
    original_label_names: dict[int, str],
) -> None:
    label_map = []

    for original_id, incremental_id in sorted(
        label_id_to_incremental_id.items(),
        key=lambda item: item[1],
    ):
        label_map.append(
            {
                "incremental_id": incremental_id,
                "original_id": original_id,
                "label_name": original_label_names.get(original_id, "UNKNOWN"),
            }
        )

    output_path = OUTPUT_DIR / "label_map.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2, ensure_ascii=False)


def write_task_config() -> None:
    task_config = {
        "scenario": "class_incremental_learning",
        "task_order": TASKS,
        "evaluation_protocol": [
            {
                "after_training_task": 1,
                "evaluate_on_tasks": [1],
            },
            {
                "after_training_task": 2,
                "evaluate_on_tasks": [1, 2],
            },
            {
                "after_training_task": 3,
                "evaluate_on_tasks": [1, 2, 3],
            },
        ],
    }

    output_path = OUTPUT_DIR / "task_config.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(task_config, file, indent=2, ensure_ascii=False)


def write_summary(summary_rows: list[dict]) -> None:
    output_path = OUTPUT_DIR / "dataset_summary.csv"

    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "task_id",
            "task_name",
            "split",
            "num_samples",
            "num_label_assignments",
            "unique_labels",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def prepare_incremental_dataset() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_id_to_incremental_id, original_label_names = collect_global_labels()

    summary_rows = []

    for task in TASKS:
        task_output_dir = OUTPUT_DIR / f"task_{task['task_id']}_{task['task_name']}"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            input_path = RAW_CLASSIFICATION_DIR / f"{task['source_prefix']}_{split}.jsonl"
            output_path = task_output_dir / f"{split}.jsonl"

            records = read_jsonl(input_path)

            transformed_records = [
                transform_record(
                    record=record,
                    task=task,
                    split=split,
                    label_id_to_incremental_id=label_id_to_incremental_id,
                )
                for record in records
            ]

            write_jsonl(output_path, transformed_records)

            label_counter = Counter()

            for record in transformed_records:
                label_counter.update(record["labels"])

            summary_rows.append(
                {
                    
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "split": split,
                    "num_samples": len(transformed_records),
                    "num_label_assignments": sum(label_counter.values()),
                    "unique_labels": sorted(label_counter.keys()),
                }
            )

    write_label_map(label_id_to_incremental_id, original_label_names)
    write_task_config()
    write_summary(summary_rows)

    print("Dataset incremental creado correctamente.")
    print(f"Carpeta de salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_incremental_dataset()
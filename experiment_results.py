"""CSV utilities for protocol-compliant experiment result records.

For failed runs, unavailable result metrics must be supplied as ``None``.
They are written as empty CSV cells and restored to ``None`` when read.
"""

import csv
import json
from collections.abc import Mapping
from pathlib import Path


EXPERIMENT_RESULT_COLUMNS = (
    "dataset",
    "algorithm",
    "seed",
    "best_fitness",
    "final_population_best_fitness",
    "silhouette",
    "error_rate_percent",
    "accuracy_percent",
    "correct_count",
    "incorrect_count",
    "nonempty_cluster_count",
    "empty_cluster_rejections",
    "runtime_seconds",
    "fitness_evaluation_count",
    "accepted_movements",
    "reproduction_events",
    "elimination_events",
    "dispersed_bacteria_count",
    "run_status",
    "failure_description",
    "parameter_configuration",
    "git_commit",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "scipy_version",
    "matplotlib_version",
)

_DUPLICATE_KEY_COLUMNS = ("dataset", "algorithm", "seed")
_RESULT_METRIC_COLUMNS = (
    "best_fitness",
    "final_population_best_fitness",
    "silhouette",
    "error_rate_percent",
    "accuracy_percent",
    "correct_count",
    "incorrect_count",
    "nonempty_cluster_count",
    "empty_cluster_rejections",
    "runtime_seconds",
    "fitness_evaluation_count",
    "accepted_movements",
    "reproduction_events",
    "elimination_events",
    "dispersed_bacteria_count",
)


def _duplicate_key(row):
    return tuple(
        "" if row[column] is None else str(row[column])
        for column in _DUPLICATE_KEY_COLUMNS
    )


def _serialize_configuration(configuration):
    if not isinstance(configuration, Mapping):
        raise ValueError("parameter_configuration must be a mapping.")
    try:
        return json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "parameter_configuration must contain only JSON values."
        ) from error


def _validate_row(row, row_number):
    status = row["run_status"]
    if status not in ("success", "failed"):
        raise ValueError(
            f"Row {row_number} has invalid run_status: {status!r}"
        )

    failure_description = row["failure_description"]
    if status == "success":
        if failure_description != "":
            raise ValueError(
                f"Row {row_number} successful run must have an empty "
                "failure_description."
            )
        missing_metrics = [
            column
            for column in _RESULT_METRIC_COLUMNS
            if row[column] is None or row[column] == ""
        ]
        if missing_metrics:
            raise ValueError(
                f"Row {row_number} successful run is missing result metrics: "
                f"{missing_metrics}"
            )
    else:
        if (
            not isinstance(failure_description, str)
            or not failure_description.strip()
        ):
            raise ValueError(
                f"Row {row_number} failed run must have a nonempty "
                "failure_description."
            )
        empty_string_metrics = [
            column
            for column in _RESULT_METRIC_COLUMNS
            if row[column] == ""
        ]
        if empty_string_metrics:
            raise ValueError(
                f"Row {row_number} unavailable failed-run metrics must be "
                f"None: {empty_string_metrics}"
            )


def write_experiment_results(rows, output_path, overwrite=False):
    rows = list(rows)
    if not rows:
        raise ValueError("At least one experiment result row is required.")

    serialized_rows = []
    seen_keys = set()
    required_columns = set(EXPERIMENT_RESULT_COLUMNS)

    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Row {row_number} must be a mapping.")

        missing_fields = [
            column for column in EXPERIMENT_RESULT_COLUMNS if column not in row
        ]
        unexpected_fields = [field for field in row if field not in required_columns]
        if missing_fields:
            raise ValueError(
                f"Row {row_number} is missing required fields: {missing_fields}"
            )
        if unexpected_fields:
            raise ValueError(
                f"Row {row_number} has unexpected fields: {unexpected_fields}"
            )

        _validate_row(row, row_number)

        key = _duplicate_key(row)
        if key in seen_keys:
            raise ValueError(f"Duplicate dataset-algorithm-seed key: {key}")
        seen_keys.add(key)

        serialized_row = []
        for column in EXPERIMENT_RESULT_COLUMNS:
            value = row[column]
            if column == "parameter_configuration":
                value = _serialize_configuration(value)
            serialized_row.append(value)
        serialized_rows.append(serialized_row)

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(EXPERIMENT_RESULT_COLUMNS)
        writer.writerows(serialized_rows)


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def read_experiment_results(input_path):
    parsed_rows = []
    seen_keys = set()

    try:
        with Path(input_path).open(
            "r", encoding="utf-8", newline=""
        ) as input_file:
            reader = csv.reader(input_file, strict=True)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("Experiment results file is empty.") from error

            if header != list(EXPERIMENT_RESULT_COLUMNS):
                raise ValueError("Experiment results header does not match schema.")

            for row_number, values in enumerate(reader, start=2):
                if len(values) != len(EXPERIMENT_RESULT_COLUMNS):
                    raise ValueError(
                        f"Row {row_number} has {len(values)} fields; "
                        f"expected {len(EXPERIMENT_RESULT_COLUMNS)}."
                    )

                row = dict(zip(EXPERIMENT_RESULT_COLUMNS, values))
                for column in _RESULT_METRIC_COLUMNS:
                    if row[column] == "":
                        row[column] = None

                _validate_row(row, row_number)

                try:
                    configuration = json.loads(
                        row["parameter_configuration"],
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"Row {row_number} has invalid parameter_configuration JSON."
                    ) from error
                if not isinstance(configuration, dict):
                    raise ValueError(
                        f"Row {row_number} parameter_configuration must be a JSON object."
                    )
                row["parameter_configuration"] = configuration

                key = _duplicate_key(row)
                if key in seen_keys:
                    raise ValueError(f"Duplicate dataset-algorithm-seed key: {key}")
                seen_keys.add(key)
                parsed_rows.append(row)
    except csv.Error as error:
        raise ValueError("Experiment results file is malformed CSV.") from error

    if not parsed_rows:
        raise ValueError("Experiment results file contains no data rows.")
    return parsed_rows
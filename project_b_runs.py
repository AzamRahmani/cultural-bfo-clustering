"""Shared run infrastructure for Project B (PROJECT_B_PROTOCOL.md Sections 12-13).

Result rows are written as CSV with a JSON metadata sidecar holding the
environment, the exact command, timestamps, and the SHA-256 fingerprint of
the CSV. Unavailable values are written as empty cells, never substituted.
"""

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
PROTOCOL_TAG = "project-b-protocol-v1"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "project_b"
SEEDS = tuple(range(30))
NUM_CLUSTERS = 6


def _git(*arguments):
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"git {' '.join(arguments)} failed") from error


def git_tree_is_clean():
    """Only tracked files count; untracked files cannot change committed code."""
    return _git("status", "--porcelain", "--untracked-files=no") == ""


def protocol_is_ancestor():
    """True if the frozen protocol tag is part of the current history."""
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", PROTOCOL_TAG, "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment_metadata():
    import numpy
    import onnxruntime
    import scipy
    import sklearn
    import torch

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "protocol_tag": PROTOCOL_TAG,
        "protocol_tag_commit": _git("rev-parse", f"{PROTOCOL_TAG}^{{commit}}"),
        "operating_system": platform.platform(),
        "cpu": cpu_name(),
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "onnxruntime_version": onnxruntime.__version__,
    }


def describe_failure(error):
    message = " ".join(str(error).split()) or "No exception message was provided."
    description = f"{type(error).__name__}: {message}"
    return description if len(description) <= 500 else description[:497] + "..."


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value {value!r} must be recorded as a failure")
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_results(rows, columns, output_path, metadata, overwrite=False):
    """Write rows (dicts with exactly `columns`) and the metadata sidecar."""
    output_path = Path(output_path)
    sidecar = metadata_path_for(output_path)
    if not overwrite:
        for path in (output_path, sidecar):
            if path.exists():
                raise FileExistsError(f"output already exists: {path}")
    for index, row in enumerate(rows):
        if set(row) != set(columns):
            missing = set(columns) - set(row)
            extra = set(row) - set(columns)
            raise ValueError(f"row {index}: missing {sorted(missing)}, extra {sorted(extra)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[column]) for column in columns])

    full_metadata = {
        **metadata,
        "results_file": output_path.name,
        "row_count": len(rows),
        "failed_row_count": sum(row.get("run_status") == "failed" for row in rows),
        "results_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    with sidecar.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(full_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return full_metadata


def metadata_path_for(output_path):
    output_path = Path(output_path)
    return output_path.with_name(output_path.stem + ".metadata.json")


def read_results(path, verify=True):
    """Read rows as dicts of strings; verify the sidecar fingerprint."""
    path = Path(path)
    if verify:
        sidecar = json.loads(metadata_path_for(path).read_text(encoding="utf-8"))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sidecar["results_sha256"]:
            raise ValueError(f"{path.name}: fingerprint {actual} differs from sidecar")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_run_preconditions(allow_dirty_tree):
    """Return (tree_clean, problems). A final run needs a clean tree and the tag."""
    problems = []
    tree_clean = git_tree_is_clean()
    if not tree_clean and not allow_dirty_tree:
        problems.append(
            "Tracked files have uncommitted changes. Commit them first, or use "
            "--allow-dirty-tree for a non-final run."
        )
    if not protocol_is_ancestor():
        problems.append(f"The protocol tag {PROTOCOL_TAG} is not in the current history.")
    return tree_clean, problems


def run_command(script_name, argv):
    return [sys.executable, script_name, *argv]


class CheckpointMismatch(RuntimeError):
    """The checkpoint belongs to a different configuration or code state."""


class Checkpoint:
    """Append-only record of completed units, so an interrupted run resumes.

    The first line is a header with the run configuration and Git commit; a
    checkpoint from any other configuration or commit is refused. Each later
    line is either a session start or one completed unit with its rows. Units
    are only recorded when complete, so a unit interrupted mid-way is rerun.
    Because every unit is seeded independently, a resumed run produces the
    same values as an uninterrupted one.
    """

    def __init__(self, path, header):
        self.path = Path(path)
        self.header = json.loads(json.dumps(header, sort_keys=True))
        self.units = {}
        self.sessions = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line.strip()]
            complete = []
            for index, line in enumerate(lines):
                try:
                    complete.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
                    # A line cut off by the interruption: that unit is rerun.
            if not complete or complete[0].get("header") != self.header:
                raise CheckpointMismatch(
                    f"{self.path} belongs to a different configuration or commit"
                )
            for entry in complete[1:]:
                if "session_started_at_utc" in entry:
                    self.sessions.append(entry["session_started_at_utc"])
                else:
                    self.units[tuple(entry["unit"])] = entry["rows"]
            with self.path.open("w", encoding="utf-8", newline="\n") as handle:
                for entry in complete:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._append({"header": self.header})
        started = utc_timestamp()
        self.sessions.append(started)
        self._append({"session_started_at_utc": started})

    def _append(self, entry):
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())

    def completed(self, unit):
        rows = self.units.get(tuple(unit))
        return None if rows is None else [dict(row) for row in rows]

    def record(self, unit, rows):
        self._append({"unit": list(unit), "rows": rows})
        self.units[tuple(unit)] = rows

    def remove(self):
        self.path.unlink(missing_ok=True)


def checkpoint_path_for(output_path):
    output_path = Path(output_path)
    return output_path.with_name(output_path.stem + ".checkpoint.jsonl")

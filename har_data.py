"""Load the UCI HAR dataset for Project B (PROJECT_B_PROTOCOL.md Section 3).

The data is read directly from the downloaded archive, whose SHA-256
fingerprint is checked first (PROJECT_B_DATA.md). Loading refuses to run if
the archive differs from the recorded one. The provider's subject-wise
train/test split is kept as delivered.
"""

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HAR_ZIP_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
HAR_INNER_ZIP_SHA256 = "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"

DEFAULT_ZIP_PATH = Path(__file__).resolve().parent / "data" / "raw" / "uci_har_240.zip"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"

INNER_ZIP_NAME = "UCI HAR Dataset.zip"
DATASET_DIR = "UCI HAR Dataset"

# Fixed channel order of the raw input, shape (windows, 9, 128).
RAW_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
WINDOW_LENGTH = 128
HAND_FEATURE_COUNT = 561
ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
EXPECTED_WINDOWS = {"train": 7352, "test": 2947}
EXPECTED_SUBJECT_COUNTS = {"train": 21, "test": 9}


class HarDataError(ValueError):
    """Raised when the HAR archive or its contents are not as recorded."""


@dataclass(frozen=True)
class HarSplit:
    """One provider split. Labels are for evaluation only."""

    hand: np.ndarray      # (windows, 561) provider features
    raw: np.ndarray       # (windows, 9, 128) inertial signals
    y: np.ndarray         # (windows,) activity 1..6
    subject: np.ndarray   # (windows,) subject 1..30

    def __len__(self):
        return len(self.y)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_har_zip(zip_path=DEFAULT_ZIP_PATH):
    """Raise HarDataError unless the archive matches the recorded fingerprint."""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise HarDataError(
            f"HAR archive not found: {zip_path}. See PROJECT_B_DATA.md for the "
            "download URL."
        )
    actual = sha256_file(zip_path)
    if actual != HAR_ZIP_SHA256:
        raise HarDataError(
            f"HAR archive fingerprint mismatch: expected {HAR_ZIP_SHA256}, "
            f"got {actual}."
        )
    return actual


def _read_matrix(archive, member, dtype=float):
    with archive.open(member) as handle:
        return np.loadtxt(io.TextIOWrapper(handle, encoding="ascii"), dtype=dtype)


def _read_split(archive, split):
    base = f"{DATASET_DIR}/{split}"
    hand = _read_matrix(archive, f"{base}/X_{split}.txt")
    y = _read_matrix(archive, f"{base}/y_{split}.txt", dtype=np.int64)
    subject = _read_matrix(archive, f"{base}/subject_{split}.txt", dtype=np.int64)
    raw = np.stack(
        [
            _read_matrix(archive, f"{base}/Inertial Signals/{channel}_{split}.txt")
            for channel in RAW_CHANNELS
        ],
        axis=1,
    )
    return HarSplit(hand=hand, raw=raw, y=y, subject=subject)


def validate_split(split_name, split):
    """Check the structure recorded in PROJECT_B_DATA.md."""
    n = EXPECTED_WINDOWS[split_name]
    checks = [
        (split.hand.shape == (n, HAND_FEATURE_COUNT), f"hand shape {split.hand.shape}"),
        (split.raw.shape == (n, len(RAW_CHANNELS), WINDOW_LENGTH), f"raw shape {split.raw.shape}"),
        (split.y.shape == (n,), f"label shape {split.y.shape}"),
        (split.subject.shape == (n,), f"subject shape {split.subject.shape}"),
        (np.isfinite(split.hand).all(), "non-finite hand features"),
        (np.isfinite(split.raw).all(), "non-finite raw signals"),
        (set(np.unique(split.y)) == set(ACTIVITY_NAMES), "unexpected activity labels"),
        (
            len(np.unique(split.subject)) == EXPECTED_SUBJECT_COUNTS[split_name],
            "unexpected subject count",
        ),
    ]
    for ok, message in checks:
        if not ok:
            raise HarDataError(f"{split_name}: {message}")


def validate_subject_split(splits):
    train_subjects = set(np.unique(splits["train"].subject))
    test_subjects = set(np.unique(splits["test"].subject))
    if train_subjects & test_subjects:
        raise HarDataError("train and test splits share subjects")
    if train_subjects | test_subjects != set(range(1, 31)):
        raise HarDataError("subjects are not exactly 1 to 30")


def _cache_path(cache_dir):
    return Path(cache_dir) / f"uci_har_{HAR_ZIP_SHA256[:16]}.npz"


def _save_cache(path, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"source_sha256": np.array(HAR_ZIP_SHA256)}
    for name, split in splits.items():
        for field in ("hand", "raw", "y", "subject"):
            arrays[f"{name}_{field}"] = getattr(split, field)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["source_sha256"]) != HAR_ZIP_SHA256:
            return None
        return {
            name: HarSplit(
                hand=data[f"{name}_hand"],
                raw=data[f"{name}_raw"],
                y=data[f"{name}_y"],
                subject=data[f"{name}_subject"],
            )
            for name in ("train", "test")
        }


def load_har(zip_path=DEFAULT_ZIP_PATH, cache_dir=DEFAULT_CACHE_DIR):
    """Return {"train": HarSplit, "test": HarSplit} from the verified archive.

    The archive fingerprint is always checked. Parsed arrays are cached as an
    .npz file named after the fingerprint; pass cache_dir=None to disable.
    Every load, cached or not, is validated against the recorded structure.
    """
    verify_har_zip(zip_path)

    splits = None
    cache_file = _cache_path(cache_dir) if cache_dir is not None else None
    if cache_file is not None and cache_file.is_file():
        splits = _load_cache(cache_file)

    if splits is None:
        with zipfile.ZipFile(zip_path) as outer:
            inner_bytes = outer.read(INNER_ZIP_NAME)
        inner_sha = hashlib.sha256(inner_bytes).hexdigest()
        if inner_sha != HAR_INNER_ZIP_SHA256:
            raise HarDataError(
                f"inner archive fingerprint mismatch: got {inner_sha}"
            )
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            splits = {name: _read_split(inner, name) for name in ("train", "test")}
        if cache_file is not None:
            _save_cache(cache_file, splits)

    for name, split in splits.items():
        validate_split(name, split)
    validate_subject_split(splits)
    return splits

"""Tests for Project B (PROJECT_B_PROTOCOL.md). Run: python test_project_b.py"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from har_data import (
    DEFAULT_ZIP_PATH,
    HAR_ZIP_SHA256,
    RAW_CHANNELS,
    HarDataError,
    HarSplit,
    load_har,
    verify_har_zip,
)
from har_preprocessing import (
    HarPreprocessor,
    RawChannelScaler,
    fitting_mask,
    flatten_raw,
    prepare_standardized,
    subset,
)


def synthetic_split(rng, n, subjects, shift=0.0):
    return HarSplit(
        hand=rng.normal(shift, 2.0, size=(n, 561)),
        raw=rng.normal(shift, 3.0, size=(n, 9, 128)) + np.arange(9)[None, :, None],
        y=np.resize(np.arange(1, 7), n),
        subject=np.resize(np.array(subjects), n),
    )


rng = np.random.default_rng(0)
train = synthetic_split(rng, 120, range(1, 22))
test = synthetic_split(rng, 60, range(22, 31), shift=5.0)
splits = {"train": train, "test": test}

# --- fitting mask --------------------------------------------------------
assert fitting_mask(train.y).all()
withheld_mask = fitting_mask(train.y, withheld_activity=3)
assert not np.any(train.y[withheld_mask] == 3)
assert withheld_mask.sum() == np.sum(train.y != 3)
for bad in (0, 7, "3"):
    try:
        fitting_mask(train.y, withheld_activity=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"activity {bad!r} was accepted")

# --- raw channel scaler --------------------------------------------------
scaler = RawChannelScaler().fit(train.raw)
assert np.allclose(scaler.mean_, train.raw.mean(axis=(0, 2)))
assert np.allclose(scaler.std_, train.raw.std(axis=(0, 2)))
standardized = scaler.transform(train.raw)
assert np.allclose(standardized.mean(axis=(0, 2)), 0.0)
assert np.allclose(standardized.std(axis=(0, 2)), 1.0)
constant = train.raw.copy()
constant[:, 4, :] = 1.5
try:
    RawChannelScaler().fit(constant)
except ValueError:
    pass
else:
    raise AssertionError("a constant channel was accepted")

# --- flattening order ----------------------------------------------------
flat = flatten_raw(train.raw)
assert flat.shape == (120, 9 * 128)
assert flat[7, 3 * 128 + 50] == train.raw[7, 3, 50]

# --- leakage: fitting uses training windows only -------------------------
prepared = prepare_standardized(splits)
reference = HarPreprocessor.fit(train)
pre = prepared["preprocessor"]
assert np.array_equal(pre.raw_scaler.mean_, reference.raw_scaler.mean_)
assert np.array_equal(pre.hand_scaler.mean_, reference.hand_scaler.mean_)
assert np.array_equal(pre.hand_scaler.scale_, reference.hand_scaler.scale_)

# Test data are transformed with training statistics, not their own.
expected_test_raw = (test.raw - train.raw.mean(axis=(0, 2))[None, :, None]) / (
    train.raw.std(axis=(0, 2))[None, :, None]
)
assert np.allclose(prepared["test"]["raw"], expected_test_raw)
assert not np.allclose(prepared["test"]["raw"].mean(axis=(0, 2)), 0.0)

# Changing the test windows does not change anything that was fitted.
altered = {"train": train, "test": synthetic_split(rng, 60, range(22, 31), shift=1000.0)}
altered_pre = prepare_standardized(altered)["preprocessor"]
assert np.array_equal(altered_pre.raw_scaler.mean_, pre.raw_scaler.mean_)
assert np.array_equal(altered_pre.raw_scaler.std_, pre.raw_scaler.std_)
assert np.array_equal(altered_pre.hand_scaler.mean_, pre.hand_scaler.mean_)

# Fitting does not read labels.
relabelled = HarSplit(hand=train.hand, raw=train.raw, y=np.full(120, -1), subject=train.subject)
assert np.array_equal(HarPreprocessor.fit(relabelled).raw_scaler.mean_, pre.raw_scaler.mean_)

# --- withheld activity (RQ2) ---------------------------------------------
prepared_rq2 = prepare_standardized(splits, withheld_activity=2)
assert not np.any(prepared_rq2["fit"]["y"] == 2)
assert len(prepared_rq2["test"]["y"]) == len(test)
assert set(prepared_rq2["test"]["y"]) == set(range(1, 7))
reference_rq2 = HarPreprocessor.fit(subset(train, train.y != 2))
assert np.array_equal(
    prepared_rq2["preprocessor"].raw_scaler.mean_, reference_rq2.raw_scaler.mean_
)
assert not np.array_equal(prepared_rq2["preprocessor"].raw_scaler.mean_, pre.raw_scaler.mean_)
assert prepared_rq2["fit"]["raw_flat"].shape[1] == 1152

# --- fingerprint check ---------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    missing = Path(tmp) / "missing.zip"
    try:
        verify_har_zip(missing)
    except HarDataError:
        pass
    else:
        raise AssertionError("a missing archive was accepted")
    wrong = Path(tmp) / "wrong.zip"
    wrong.write_bytes(b"not the HAR archive")
    try:
        load_har(wrong, cache_dir=None)
    except HarDataError:
        pass
    else:
        raise AssertionError("an archive with a wrong fingerprint was accepted")

print("Project B preprocessing tests passed.")

# --- autoencoder (protocol Section 5) ------------------------------------
import torch

from har_autoencoder import (
    ConvAutoencoder,
    TrainingConfig,
    encode,
    parameter_count,
    reconstruction_errors,
    train_autoencoder,
)

default_config = TrainingConfig()
assert (default_config.latent_dim, default_config.epochs) == (16, 30)
assert (default_config.batch_size, default_config.learning_rate) == (128, 1e-3)

model = ConvAutoencoder(16)
assert parameter_count(model.encoder) == 1472 + 10304 + 32784
assert parameter_count(model) == 1472 + 10304 + 32784 + 34816 + 8224 + 1161
with torch.no_grad():
    probe = torch.zeros(5, 9, 128)
    assert model.encode(probe).shape == (5, 16)
    assert model(probe).shape == (5, 9, 128)

# Structured synthetic windows: sinusoids with random phase and amplitude.
signal_rng = np.random.default_rng(1)
time_axis = np.linspace(0, 4 * np.pi, 128)
phase = signal_rng.uniform(0, 2 * np.pi, size=(300, 9, 1))
amplitude = signal_rng.uniform(0.5, 1.5, size=(300, 9, 1))
synthetic_raw = amplitude * np.sin(time_axis[None, None, :] + phase)

small = TrainingConfig(epochs=3, batch_size=64)
model_a, history_a = train_autoencoder(synthetic_raw, seed=7, config=small)
model_b, history_b = train_autoencoder(synthetic_raw, seed=7, config=small)
model_c, history_c = train_autoencoder(synthetic_raw, seed=8, config=small)

# Same seed: bit-identical training; different seed: different model.
assert history_a["epoch_losses"] == history_b["epoch_losses"]
for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
    assert torch.equal(p_a, p_b)
assert history_a["epoch_losses"] != history_c["epoch_losses"]
assert len(history_a["epoch_losses"]) == 3
assert history_a["config"]["epochs"] == 3 and history_a["seed"] == 7
assert np.isfinite(history_a["final_training_loss"])

# Training lowers the reconstruction error on structured data.
untrained_error = reconstruction_errors(ConvAutoencoder(16), synthetic_raw).mean()
assert history_a["final_training_loss"] < untrained_error
assert history_a["epoch_losses"][-1] < history_a["epoch_losses"][0]

# Encoding and scoring: shapes, no side effects, batch-size independence.
latent = encode(model_a, synthetic_raw)
assert latent.shape == (300, 16) and latent.dtype == np.float64
assert np.allclose(latent, encode(model_a, synthetic_raw, batch_size=7), atol=1e-5)
errors = reconstruction_errors(model_a, synthetic_raw)
assert errors.shape == (300,) and np.all(errors >= 0)
assert np.allclose(errors.mean(), history_a["final_training_loss"])
assert np.array_equal(latent, encode(model_a, synthetic_raw))

# Invalid input is rejected rather than repaired.
for bad_input in (synthetic_raw[:, :8, :], np.full((4, 9, 128), np.nan)):
    try:
        train_autoencoder(bad_input, seed=0, config=small)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid autoencoder input was accepted")

print("Project B autoencoder tests passed.")

# --- run infrastructure and clustering runner (protocol Section 6) --------
import contextlib
import io
import json

import project_b_runs
import run_project_b_clustering as clustering_runner
from project_b_runs import metadata_path_for, read_results, write_results


def structured_split(rng, per_class, subjects):
    """Six separable activity groups in both HAND and RAW form."""
    y = np.repeat(np.arange(1, 7), per_class)
    centers_hand = rng.normal(0, 6.0, size=(6, 561))
    hand = centers_hand[y - 1] + rng.normal(0, 1.0, size=(len(y), 561))
    t = np.linspace(0, 4 * np.pi, 128)
    raw = (
        np.sin(t[None, None, :] * y[:, None, None] / 2.0 + np.arange(9)[None, :, None])
        + rng.normal(0, 0.1, size=(len(y), 9, 128))
    )
    return HarSplit(hand=hand, raw=raw, y=y, subject=np.resize(np.array(subjects), len(y)))


cluster_rng = np.random.default_rng(3)
cluster_splits = {
    "train": structured_split(cluster_rng, 20, range(1, 22)),
    "test": structured_split(cluster_rng, 10, range(22, 31)),
}
tiny_config = TrainingConfig(epochs=1, batch_size=32)
quiet = lambda message: None  # noqa: E731

rows_first = clustering_runner.run_experiment(
    cluster_splits, (0, 1), clustering_runner.REPRESENTATIONS,
    clustering_runner.ALGORITHMS, tiny_config, log=quiet,
)
assert len(rows_first) == 2 * 4 * 3
keys = {(r["seed"], r["representation"], r["algorithm"]) for r in rows_first}
assert len(keys) == len(rows_first)
for row in rows_first:
    assert set(row) == set(clustering_runner.COLUMNS)
    expected_dim = {"HAND": 561, "RAW": 1152, "PCA16": 16, "AE16": 16}[row["representation"]]
    assert row["representation_dim"] == expected_dim
    if row["representation"] == "AE16":
        assert row["ae_final_training_loss"] is not None
    else:
        assert row["ae_final_training_loss"] is None
    if row["run_status"] == "success":
        assert row["nonempty_cluster_count"] == 6
        assert 0 <= row["error_rate_percent"] <= 100
        assert row["correct_count"] + row["incorrect_count"] == 60
        assert -1 <= row["silhouette"] <= 1
        if row["algorithm"] == "KMEANS":
            assert row["fitness_evaluation_count"] is None
        else:
            assert row["fitness_evaluation_count"] > 0
    else:
        assert row["failure_description"]
assert all(r["run_status"] == "success" for r in rows_first if r["algorithm"] == "KMEANS")

# Deterministic apart from runtimes.
rows_second = clustering_runner.run_experiment(
    cluster_splits, (1,), clustering_runner.REPRESENTATIONS,
    clustering_runner.ALGORITHMS, tiny_config, log=quiet,
)
timing_columns = {"clustering_runtime_seconds", "ae_training_seconds"}
assert len(rows_second) == 12
for first, second in zip(rows_first[12:], rows_second):
    for column in clustering_runner.COLUMNS:
        if column not in timing_columns:
            assert first[column] == second[column], column

# Test labels are used only for evaluation: shuffling them leaves the
# label-free quantities unchanged.
shuffled_test = HarSplit(
    hand=cluster_splits["test"].hand,
    raw=cluster_splits["test"].raw,
    y=np.random.default_rng(9).permutation(cluster_splits["test"].y),
    subject=cluster_splits["test"].subject,
)
rows_shuffled = clustering_runner.run_experiment(
    {"train": cluster_splits["train"], "test": shuffled_test}, (0,),
    ("PCA16", "AE16"), ("KMEANS",), tiny_config, log=quiet,
)
for shuffled in rows_shuffled:
    original = next(
        r for r in rows_first
        if (r["seed"], r["representation"], r["algorithm"])
        == (0, shuffled["representation"], "KMEANS")
    )
    assert shuffled["fitness"] == original["fitness"]
    assert shuffled["silhouette"] == original["silhouette"]

# An autoencoder failure is recorded for every AE row, not skipped.
original_train = clustering_runner.train_autoencoder


def failing_training(*args, **kwargs):
    raise RuntimeError("simulated failure")


clustering_runner.train_autoencoder = failing_training
try:
    failed_rows = clustering_runner.run_experiment(
        cluster_splits, (0,), ("PCA16", "AE16"), ("KMEANS", "BF"), tiny_config, log=quiet,
    )
finally:
    clustering_runner.train_autoencoder = original_train
ae_rows = [r for r in failed_rows if r["representation"] == "AE16"]
assert len(failed_rows) == 4 and len(ae_rows) == 2
assert all(r["run_status"] == "failed" and "simulated failure" in r["failure_description"] for r in ae_rows)
assert all(r["error_rate_percent"] is None for r in ae_rows)

# CSV writing, fingerprint sidecar, and tamper detection.
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "clustering.csv"
    summary = write_results(rows_first, clustering_runner.COLUMNS, out, {"note": "test"})
    assert summary["row_count"] == 24
    back = read_results(out)
    assert len(back) == 24 and tuple(back[0]) == clustering_runner.COLUMNS
    assert float(back[0]["error_rate_percent"]) == rows_first[0]["error_rate_percent"]
    assert back[0]["ae_final_training_loss"] == ""
    try:
        write_results(rows_first, clustering_runner.COLUMNS, out, {})
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing results were overwritten")
    out.write_bytes(out.read_bytes() + b"\n")
    try:
        read_results(out)
    except ValueError:
        pass
    else:
        raise AssertionError("a modified results file was accepted")
    bad_row = dict(rows_first[0], silhouette=float("nan"))
    try:
        write_results([bad_row], clustering_runner.COLUMNS, Path(tmp) / "nan.csv", {})
    except ValueError:
        pass
    else:
        raise AssertionError("a non-finite value was written")

    # Command line: partial designs cannot use the protocol output path.
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            clustering_runner.parse_arguments(["--seeds", "0"])
        except SystemExit:
            pass
        else:
            raise AssertionError("a partial design was allowed on the protocol path")

    cli_out = Path(tmp) / "cli.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        code = clustering_runner.main(
            ["--output", str(cli_out), "--seeds", "0", "--representations", "PCA16",
             "--algorithms", "KMEANS", "--ae-epochs", "1", "--allow-dirty-tree"],
            splits=cluster_splits,
        )
    assert code == 0
    metadata = json.loads(metadata_path_for(cli_out).read_text(encoding="utf-8"))
    assert metadata["full_protocol_design"] is False
    assert metadata["protocol_tag"] == project_b_runs.PROTOCOL_TAG
    assert metadata["row_count"] == 1 and metadata["torch_version"]
    assert len(read_results(cli_out)) == 1

print("Project B clustering runner tests passed.")

# --- anomaly runner (protocol Section 7) ---------------------------------
import run_project_b_anomaly as anomaly_runner

anomaly_rows = anomaly_runner.run_experiment(
    cluster_splits, (0, 1), (2, 5), anomaly_runner.DETECTORS, tiny_config, log=quiet,
)
assert len(anomaly_rows) == 2 * 2 * 3
for row in anomaly_rows:
    assert set(row) == set(anomaly_runner.COLUMNS)
    assert row["run_status"] == "success", row["failure_description"]
    assert 0 <= row["roc_auc"] <= 1 and 0 <= row["pr_auc"] <= 1
    assert row["positive_count"] == 10 and row["test_count"] == 60
    assert row["fit_window_count"] == 100  # 120 training windows minus 20 withheld
    assert (row["ae_final_training_loss"] is None) == (row["detector"] != "AE")

# PCA is deterministic: identical for every seed of one withheld activity.
pca_values = {
    (r["withheld_activity"], r["seed"]): r["roc_auc"] for r in anomaly_rows if r["detector"] == "PCA"
}
assert pca_values[(2, 0)] == pca_values[(2, 1)]
assert pca_values[(5, 0)] == pca_values[(5, 1)]

# Seeded detectors are reproducible.
repeat_rows = anomaly_runner.run_experiment(
    cluster_splits, (1,), (2,), ("AE", "IF"), tiny_config, log=quiet,
)
for repeated in repeat_rows:
    original = next(
        r for r in anomaly_rows
        if (r["withheld_activity"], r["seed"], r["detector"]) == (2, 1, repeated["detector"])
    )
    assert repeated["roc_auc"] == original["roc_auc"]
    assert repeated["pr_auc"] == original["pr_auc"]

# Scores are computed without test labels: relabelling the test windows
# changes only which windows count as positive.
relabelled_test = HarSplit(
    hand=cluster_splits["test"].hand,
    raw=cluster_splits["test"].raw,
    y=np.where(cluster_splits["test"].y == 2, 4, np.where(cluster_splits["test"].y == 4, 2, cluster_splits["test"].y)),
    subject=cluster_splits["test"].subject,
)
direct = anomaly_runner.pca_scores(
    prepare_standardized(cluster_splits, withheld_activity=2)["fit"]["raw_flat"],
    prepare_standardized(cluster_splits, withheld_activity=2)["test"]["raw_flat"],
)
relabelled_direct = anomaly_runner.pca_scores(
    prepare_standardized({"train": cluster_splits["train"], "test": relabelled_test}, withheld_activity=2)["fit"]["raw_flat"],
    prepare_standardized({"train": cluster_splits["train"], "test": relabelled_test}, withheld_activity=2)["test"]["raw_flat"],
)
assert np.array_equal(direct, relabelled_direct)

# A perfect score separates positives completely; a failure is recorded.
perfect = anomaly_runner.score_row({}, 3, np.where(cluster_splits["test"].y == 3, 10.0, 0.0), cluster_splits["test"].y)
assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
try:
    anomaly_runner.score_row({}, 3, np.full(60, np.nan), cluster_splits["test"].y)
except ValueError:
    pass
else:
    raise AssertionError("non-finite anomaly scores were accepted")

with tempfile.TemporaryDirectory() as tmp:
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            anomaly_runner.parse_arguments(["--activities", "1"])
        except SystemExit:
            pass
        else:
            raise AssertionError("a partial design was allowed on the protocol path")
    cli_out = Path(tmp) / "anomaly.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        code = anomaly_runner.main(
            ["--output", str(cli_out), "--seeds", "0", "--activities", "6",
             "--detectors", "PCA", "IF", "--allow-dirty-tree"],
            splits=cluster_splits,
        )
    assert code == 0
    cli_rows = read_results(cli_out)
    assert [r["detector"] for r in cli_rows] == ["PCA", "IF"]
    assert all(r["withheld_activity_name"] == "LAYING" for r in cli_rows)

print("Project B anomaly runner tests passed.")

# --- edge measurements (protocol Section 8) ------------------------------
import logging
import warnings

from sklearn.decomposition import PCA

import run_project_b_edge as edge_runner

logging.getLogger("torch.onnx").setLevel(logging.ERROR)
with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings(), \
        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    warnings.simplefilter("ignore")
    edge_rows = edge_runner.run_experiment(cluster_splits, tiny_config, Path(tmp) / "models", log=quiet)
    model_files = sorted(p.name for p in (Path(tmp) / "models").iterdir())

assert [r["model"] for r in edge_rows] == ["AE16_ENCODER", "PCA16"]
for row in edge_rows:
    assert set(row) == set(edge_runner.COLUMNS)
    assert row["run_status"] == "success", row["failure_description"]
    assert row["float32_weight_bytes"] == 4 * row["parameter_count"]
    assert row["onnx_file_bytes"] > 0 and len(row["onnx_sha256"]) == 64
    assert row["onnx_max_abs_difference"] <= 1e-4
    assert 0 < row["latency_median_ms"] <= row["latency_p95_ms"]
    assert (row["warmup_runs"], row["timed_runs"], row["intra_op_threads"]) == (100, 1000, 1)
assert edge_rows[0]["parameter_count"] == 1472 + 10304 + 32784
assert edge_rows[0]["training_seed"] == 0 and edge_rows[1]["training_seed"] is None
assert edge_rows[1]["parameter_count"] == 1152 * 16 + 16
assert model_files == ["ae16_encoder.onnx", "pca16.onnx"]

# The PCA layer reproduces scikit-learn's projection exactly enough.
pca_check = PCA(n_components=16, svd_solver="full").fit(prepared["fit"]["raw_flat"])
projection = edge_runner.PcaProjection(pca_check).eval()
with torch.no_grad():
    projected = projection(torch.from_numpy(prepared["test"]["raw"].astype(np.float32))).numpy()
assert np.allclose(projected, pca_check.transform(prepared["test"]["raw_flat"]), atol=1e-4)

print("Project B edge measurement tests passed.")

# --- real dataset ----------------------------------------------------------
if not DEFAULT_ZIP_PATH.is_file():
    print(f"SKIPPED real-data checks: {DEFAULT_ZIP_PATH} not found.")
else:
    assert verify_har_zip() == HAR_ZIP_SHA256

    with tempfile.TemporaryDirectory() as tmp:
        tampered = Path(tmp) / "tampered.zip"
        shutil.copyfile(DEFAULT_ZIP_PATH, tampered)
        with open(tampered, "r+b") as handle:
            handle.seek(1000)
            byte = handle.read(1)
            handle.seek(1000)
            handle.write(bytes([byte[0] ^ 0xFF]))
        try:
            verify_har_zip(tampered)
        except HarDataError:
            pass
        else:
            raise AssertionError("a tampered archive was accepted")

    har = load_har()
    assert har["train"].hand.shape == (7352, 561)
    assert har["test"].hand.shape == (2947, 561)
    assert har["train"].raw.shape == (7352, len(RAW_CHANNELS), 128)
    assert har["test"].raw.shape == (2947, len(RAW_CHANNELS), 128)
    assert not set(har["train"].subject) & set(har["test"].subject)
    assert len(har["train"]) + len(har["test"]) == 10299

    real = prepare_standardized(har)
    assert real["fit_mask"].all()
    assert real["fit"]["raw_flat"].shape == (7352, 1152)
    assert real["test"]["hand"].shape == (2947, 561)
    assert np.allclose(real["fit"]["raw"].mean(axis=(0, 2)), 0.0, atol=1e-10)
    assert np.allclose(real["fit"]["raw"].std(axis=(0, 2)), 1.0)
    for activity in range(1, 7):
        rq2 = prepare_standardized(har, withheld_activity=activity)
        assert not np.any(rq2["fit"]["y"] == activity)
        assert len(rq2["test"]["y"]) == 2947

    print("Project B real-data loader tests passed.")

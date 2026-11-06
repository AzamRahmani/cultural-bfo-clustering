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

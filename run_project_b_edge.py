"""Project B edge measurements (PROJECT_B_PROTOCOL.md Section 8).

Measures the seed-0 AE-16 encoder (trained as in the clustering experiment)
and the PCA-16 projection: parameter count, float32 weight size, ONNX file
size, and single-window latency with ONNX Runtime on one thread. Latency on
this PC is a proxy, not a phone measurement.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from sklearn.decomposition import PCA
from torch import nn

from har_autoencoder import TrainingConfig, parameter_count, train_autoencoder
from har_preprocessing import prepare_standardized
from project_b_runs import (
    RESULTS_DIR,
    check_run_preconditions,
    describe_failure,
    environment_metadata,
    run_command,
    utc_timestamp,
    write_results,
)

MODELS = ("AE16_ENCODER", "PCA16")
TRAINING_SEED = 0
LATENT_DIM = 16
WARMUP_RUNS = 100
TIMED_RUNS = 1000
DEFAULT_OUTPUT = RESULTS_DIR / "edge_results.csv"

COLUMNS = (
    "model",
    "run_status",
    "failure_description",
    "parameter_count",
    "float32_weight_bytes",
    "onnx_file_bytes",
    "onnx_sha256",
    "onnx_max_abs_difference",
    "latency_median_ms",
    "latency_p95_ms",
    "warmup_runs",
    "timed_runs",
    "intra_op_threads",
    "training_seed",
    "parameter_configuration",
)


class PcaProjection(nn.Module):
    """PCA as one linear layer: z = (x_flat - mean) @ components.T."""

    def __init__(self, pca):
        super().__init__()
        components = torch.from_numpy(pca.components_.astype(np.float32))
        mean = torch.from_numpy(pca.mean_.astype(np.float32))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(components.shape[1], components.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(components)
            self.linear.bias.copy_(-(components @ mean))

    def forward(self, x):
        return self.linear(self.flatten(x))


def export_onnx(module, example, path):
    module.eval()
    program = torch.onnx.export(
        module, (example,), dynamo=True, input_names=["window"], output_names=["latent"],
    )
    program.save(str(path))
    extra = [p for p in path.parent.iterdir() if p.name.startswith(path.name) and p != path]
    if extra:
        raise RuntimeError(f"unexpected external ONNX data files: {extra}")
    return path


def measure_latency(path, window, warmup=WARMUP_RUNS, runs=TIMED_RUNS):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: window}
    for _ in range(warmup):
        session.run(None, feed)
    timings = np.empty(runs)
    for index in range(runs):
        start = time.perf_counter_ns()
        session.run(None, feed)
        timings[index] = (time.perf_counter_ns() - start) / 1e6
    output = session.run(None, feed)[0]
    return float(np.median(timings)), float(np.percentile(timings, 95, method="linear")), output


def measure_model(name, module, window, model_dir, configuration):
    row = {column: None for column in COLUMNS}
    row.update(
        model=name,
        run_status="failed",
        failure_description="",
        warmup_runs=WARMUP_RUNS,
        timed_runs=TIMED_RUNS,
        intra_op_threads=1,
        training_seed=TRAINING_SEED if name == "AE16_ENCODER" else None,
        parameter_configuration=configuration,
    )
    try:
        count = parameter_count(module)
        row.update(parameter_count=count, float32_weight_bytes=4 * count)
        path = export_onnx(module, torch.from_numpy(window), model_dir / f"{name.lower()}.onnx")
        row["onnx_file_bytes"] = path.stat().st_size
        row["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        median, p95, onnx_output = measure_latency(path, window)
        with torch.no_grad():
            torch_output = module(torch.from_numpy(window)).numpy()
        difference = float(np.max(np.abs(onnx_output - torch_output)))
        if not difference <= 1e-4:
            raise ValueError(f"ONNX output differs from PyTorch by {difference}")
        row.update(
            onnx_max_abs_difference=difference,
            latency_median_ms=median,
            latency_p95_ms=p95,
            run_status="success",
        )
    except Exception as error:
        row["failure_description"] = describe_failure(error)
    return row


def run_experiment(splits, ae_config, model_dir, log=print):
    prepared = prepare_standardized(splits)
    window = np.ascontiguousarray(prepared["test"]["raw"][:1], dtype=np.float32)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        autoencoder, _ = train_autoencoder(prepared["fit"]["raw"], TRAINING_SEED, ae_config)
        encoder = autoencoder.encoder.eval()
        rows.append(measure_model("AE16_ENCODER", encoder, window, model_dir,
                                  {"latent_dim": LATENT_DIM, **ae_config.as_dict()}))
    except Exception as error:
        row = {column: None for column in COLUMNS}
        row.update(model="AE16_ENCODER", run_status="failed",
                   failure_description="autoencoder training failed: " + describe_failure(error),
                   training_seed=TRAINING_SEED, parameter_configuration=ae_config.as_dict())
        rows.append(row)

    pca = PCA(n_components=LATENT_DIM, svd_solver="full").fit(prepared["fit"]["raw_flat"])
    rows.append(measure_model("PCA16", PcaProjection(pca).eval(), window, model_dir,
                              {"n_components": LATENT_DIM, "svd_solver": "full"}))
    for row in rows:
        status = "SUCCESS" if row["run_status"] == "success" else "FAILED"
        log(f"{status} model={row['model']} {row['failure_description'] or ''}".rstrip())
    return rows


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ae-epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    arguments = parser.parse_args(argv)
    arguments.full_design = arguments.ae_epochs == TrainingConfig().epochs
    if not arguments.full_design and arguments.output.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("a modified design must not write to the protocol output path")
    return arguments


def main(argv=None, splits=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    arguments = parse_arguments(argv)
    tree_clean, problems = check_run_preconditions(arguments.allow_dirty_tree)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    if splits is None:
        from har_data import load_har

        splits = load_har()

    ae_config = TrainingConfig(epochs=arguments.ae_epochs)
    model_dir = arguments.output.parent / (arguments.output.stem + "_models")
    metadata = environment_metadata()
    started = utc_timestamp()
    rows = run_experiment(splits, ae_config, model_dir, log=lambda m: print(m, flush=True))
    import onnx

    metadata.update(
        experiment="edge",
        full_protocol_design=arguments.full_design,
        git_tree_clean=tree_clean,
        allow_dirty_tree=arguments.allow_dirty_tree,
        command=run_command(Path(__file__).name, argv),
        started_at_utc=started,
        completed_at_utc=utc_timestamp(),
        autoencoder_config=ae_config.as_dict(),
        onnx_version=onnx.__version__,
        model_directory=model_dir.name,
        latency_input="first standardized test window, shape (1, 9, 128), float32",
    )
    summary = write_results(rows, COLUMNS, arguments.output, metadata, arguments.overwrite)
    print(f"Wrote {summary['row_count']} rows ({summary['failed_row_count']} failed) to {arguments.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

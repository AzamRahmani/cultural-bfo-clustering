"""Project B anomaly experiment (PROJECT_B_PROTOCOL.md Sections 7, 9-10).

Leave-one-activity-out novelty detection: for each withheld activity and
seed, fit the detectors on the training windows without that activity, score
all 2,947 test windows, and evaluate with the withheld activity as positive.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from har_autoencoder import TrainingConfig, reconstruction_errors, train_autoencoder
from har_data import ACTIVITY_NAMES
from har_preprocessing import prepare_standardized
from project_b_runs import (
    RESULTS_DIR,
    SEEDS,
    check_run_preconditions,
    describe_failure,
    environment_metadata,
    run_command,
    utc_timestamp,
    write_results,
)

DETECTORS = ("AE", "PCA", "IF")
ACTIVITIES = tuple(ACTIVITY_NAMES)
LATENT_DIM = 16
IF_PARAMETERS = {"n_estimators": 100}
DEFAULT_OUTPUT = RESULTS_DIR / "anomaly_results.csv"

COLUMNS = (
    "withheld_activity",
    "withheld_activity_name",
    "detector",
    "seed",
    "run_status",
    "failure_description",
    "roc_auc",
    "pr_auc",
    "positive_count",
    "test_count",
    "fit_window_count",
    "detector_runtime_seconds",
    "ae_final_training_loss",
    "ae_training_seconds",
    "parameter_configuration",
)


def detector_configuration(detector, seed):
    if detector == "AE":
        return {"latent_dim": LATENT_DIM, "seed": seed, "score": "mean squared reconstruction error"}
    if detector == "PCA":
        return {"n_components": LATENT_DIM, "svd_solver": "full", "score": "mean squared reconstruction error"}
    return {**IF_PARAMETERS, "random_state": seed, "input": "HAND", "score": "-score_samples"}


def pca_scores(fit_flat, test_flat):
    pca = PCA(n_components=LATENT_DIM, svd_solver="full").fit(fit_flat)
    reconstructed = pca.inverse_transform(pca.transform(test_flat))
    return np.mean((test_flat - reconstructed) ** 2, axis=1)


def if_scores(fit_hand, test_hand, seed):
    forest = IsolationForest(random_state=seed, **IF_PARAMETERS).fit(fit_hand)
    return -forest.score_samples(test_hand)


def empty_row(activity, detector, seed, prepared):
    row = {column: None for column in COLUMNS}
    row.update(
        withheld_activity=activity,
        withheld_activity_name=ACTIVITY_NAMES[activity],
        detector=detector,
        seed=seed,
        run_status="failed",
        failure_description="",
        test_count=len(prepared["test"]["y"]),
        positive_count=int(np.sum(prepared["test"]["y"] == activity)),
        fit_window_count=len(prepared["fit"]["y"]),
        parameter_configuration=detector_configuration(detector, seed),
    )
    return row


def score_row(row, activity, scores, y_test):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y_test.shape or not np.isfinite(scores).all():
        raise ValueError("anomaly scores are missing or non-finite")
    positive = (y_test == activity).astype(int)
    row.update(
        roc_auc=float(roc_auc_score(positive, scores)),
        pr_auc=float(average_precision_score(positive, scores)),
        run_status="success",
    )
    return row


def run_experiment(splits, seeds, activities, detectors, ae_config, log=print):
    rows = []
    for activity in activities:
        prepared = prepare_standardized(splits, withheld_activity=activity)
        fit, test = prepared["fit"], prepared["test"]
        if np.any(fit["y"] == activity):
            raise AssertionError("withheld activity leaked into the fitting set")

        pca_result = None
        if "PCA" in detectors:
            start = time.perf_counter()
            try:
                pca_result = (pca_scores(fit["raw_flat"], test["raw_flat"]), None)
            except Exception as error:
                pca_result = (None, error)
            pca_seconds = time.perf_counter() - start

        for seed in seeds:
            for detector in detectors:
                row = empty_row(activity, detector, seed, prepared)
                try:
                    if detector == "AE":
                        model, history = train_autoencoder(fit["raw"], seed, ae_config)
                        row["ae_final_training_loss"] = history["final_training_loss"]
                        row["ae_training_seconds"] = history["training_seconds"]
                        start = time.perf_counter()
                        scores = reconstruction_errors(model, test["raw"])
                        row["detector_runtime_seconds"] = (
                            history["training_seconds"] + time.perf_counter() - start
                        )
                    elif detector == "PCA":
                        scores, error = pca_result
                        if error is not None:
                            raise error
                        row["detector_runtime_seconds"] = pca_seconds
                    else:
                        start = time.perf_counter()
                        scores = if_scores(fit["hand"], test["hand"], seed)
                        row["detector_runtime_seconds"] = time.perf_counter() - start
                    score_row(row, activity, scores, test["y"])
                except Exception as error:
                    row["run_status"] = "failed"
                    row["roc_auc"] = row["pr_auc"] = None
                    row["failure_description"] = describe_failure(error)
                rows.append(row)
                status = "SUCCESS" if row["run_status"] == "success" else "FAILED"
                detail = "" if status == "SUCCESS" else f": {row['failure_description']}"
                log(f"{status} withheld={activity} seed={seed} detector={detector}{detail}")
    return rows


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--activities", nargs="+", type=int, choices=ACTIVITIES, default=ACTIVITIES)
    parser.add_argument("--detectors", nargs="+", choices=DETECTORS, default=DETECTORS)
    parser.add_argument("--ae-epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    arguments = parser.parse_args(argv)
    full_design = (
        tuple(arguments.seeds) == SEEDS
        and tuple(arguments.activities) == ACTIVITIES
        and tuple(arguments.detectors) == DETECTORS
        and arguments.ae_epochs == TrainingConfig().epochs
    )
    if not full_design and arguments.output.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("a partial or modified design must not write to the protocol output path")
    for name, values in (("seeds", arguments.seeds), ("activities", arguments.activities),
                         ("detectors", arguments.detectors)):
        if len(set(values)) != len(values):
            parser.error(f"{name} must not contain duplicates")
    arguments.full_design = full_design
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
    metadata = environment_metadata()
    started = utc_timestamp()
    rows = run_experiment(
        splits, arguments.seeds, arguments.activities, arguments.detectors,
        ae_config, log=lambda message: print(message, flush=True),
    )
    metadata.update(
        experiment="anomaly",
        full_protocol_design=arguments.full_design,
        git_tree_clean=tree_clean,
        allow_dirty_tree=arguments.allow_dirty_tree,
        command=run_command(Path(__file__).name, argv),
        started_at_utc=started,
        completed_at_utc=utc_timestamp(),
        autoencoder_config=ae_config.as_dict(),
        seeds=list(arguments.seeds),
        activities=list(arguments.activities),
        detectors=list(arguments.detectors),
    )
    summary = write_results(rows, COLUMNS, arguments.output, metadata, arguments.overwrite)
    print(f"Wrote {summary['row_count']} rows ({summary['failed_row_count']} failed) to {arguments.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

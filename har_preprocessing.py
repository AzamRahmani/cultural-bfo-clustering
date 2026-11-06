"""Leakage-safe preprocessing for Project B (PROJECT_B_PROTOCOL.md Section 4).

Every statistic is fitted on the fitting set only: the training windows, or in
RQ2 the training windows without the withheld activity. Test windows are only
transformed with those fitted statistics.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from har_data import ACTIVITY_NAMES, HarSplit


def fitting_mask(y, withheld_activity=None):
    """Boolean mask of training windows used for fitting.

    withheld_activity=None keeps every window (RQ1). An activity 1..6 removes
    that activity (RQ2 novelty design); this is the only use of training
    labels during fitting.
    """
    y = np.asarray(y)
    if withheld_activity is None:
        return np.ones(len(y), dtype=bool)
    if withheld_activity not in ACTIVITY_NAMES:
        raise ValueError(f"unknown activity: {withheld_activity}")
    return y != withheld_activity


class RawChannelScaler:
    """Standardize each inertial channel with one mean and standard deviation.

    Statistics are taken over all fitting windows and all time steps
    (population standard deviation). A constant channel is reported as an
    error instead of being silently repaired.
    """

    def fit(self, raw):
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 3:
            raise ValueError("raw signals must have shape (windows, channels, time)")
        mean = raw.mean(axis=(0, 2))
        std = raw.std(axis=(0, 2))
        if np.any(std == 0) or not np.all(np.isfinite(std)):
            raise ValueError("a raw channel has zero or non-finite variance")
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, raw):
        raw = np.asarray(raw, dtype=float)
        return (raw - self.mean_[None, :, None]) / self.std_[None, :, None]


def flatten_raw(raw):
    """(windows, 9, 128) -> (windows, 1152), channel by channel."""
    raw = np.asarray(raw)
    return raw.reshape(len(raw), -1)


@dataclass
class HarPreprocessor:
    """Scalers for the HAND and RAW representations, fitted together."""

    hand_scaler: StandardScaler
    raw_scaler: RawChannelScaler

    @classmethod
    def fit(cls, fit_split):
        """Fit on the fitting set only. Labels are not read."""
        return cls(
            hand_scaler=StandardScaler().fit(fit_split.hand),
            raw_scaler=RawChannelScaler().fit(fit_split.raw),
        )

    def transform_hand(self, hand):
        return self.hand_scaler.transform(hand)

    def transform_raw(self, raw):
        return self.raw_scaler.transform(raw)


def subset(split, mask):
    return HarSplit(
        hand=split.hand[mask],
        raw=split.raw[mask],
        y=split.y[mask],
        subject=split.subject[mask],
    )


def prepare_standardized(splits, withheld_activity=None):
    """Fit preprocessing on the fitting set and transform fit and test windows.

    Returns a dict with the fitted preprocessor, the fitting mask over the
    training windows, and standardized "fit" and "test" data:
    hand (windows, 561), raw (windows, 9, 128), raw_flat (windows, 1152),
    plus y and subject for evaluation only.
    """
    mask = fitting_mask(splits["train"].y, withheld_activity)
    fit_split = subset(splits["train"], mask)
    preprocessor = HarPreprocessor.fit(fit_split)

    def standardize(split):
        raw = preprocessor.transform_raw(split.raw)
        return {
            "hand": preprocessor.transform_hand(split.hand),
            "raw": raw,
            "raw_flat": flatten_raw(raw),
            "y": split.y,
            "subject": split.subject,
        }

    return {
        "preprocessor": preprocessor,
        "fit_mask": mask,
        "fit": standardize(fit_split),
        "test": standardize(splits["test"]),
    }

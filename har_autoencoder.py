"""Convolutional autoencoder for Project B (PROJECT_B_PROTOCOL.md Section 5).

The architecture and training settings are fixed by the frozen protocol; there
is no early stopping, validation-based selection, or hyperparameter search.
Training is CPU-only and deterministic for a given seed and thread count.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

CHANNELS = 9
WINDOW_LENGTH = 128
ENCODED_LENGTH = 32      # 128 -> 64 -> 32 after two stride-2 convolutions
HIDDEN_CHANNELS = 64


@dataclass(frozen=True)
class TrainingConfig:
    latent_dim: int = 16          # decision D1
    epochs: int = 30              # decision D2
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_threads: int = 4          # fixed; equals the physical cores of the study PC

    def as_dict(self):
        return asdict(self)


class TrainingFailure(RuntimeError):
    """Raised when training produces a non-finite loss (protocol Section 12)."""


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(CHANNELS, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, HIDDEN_CHANNELS, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(HIDDEN_CHANNELS * ENCODED_LENGTH, latent_dim),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_CHANNELS * ENCODED_LENGTH),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(HIDDEN_CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, CHANNELS, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        hidden = self.decoder_input(z).view(-1, HIDDEN_CHANNELS, ENCODED_LENGTH)
        return self.decoder(hidden)

    def forward(self, x):
        return self.decode(self.encode(x))


def configure_determinism(seed, num_threads):
    """Seed every random source and force deterministic CPU kernels."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


def _as_tensor(raw):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (CHANNELS, WINDOW_LENGTH):
        raise ValueError(f"expected (windows, {CHANNELS}, {WINDOW_LENGTH}), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("input contains non-finite values")
    return torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))


def train_autoencoder(raw_fit, seed, config=TrainingConfig()):
    """Train on standardized fitting windows (windows, 9, 128). No labels.

    Returns (model in eval mode, history). history contains the mean batch
    loss of every epoch, the final loss over the whole fitting set after the
    last epoch, and the wall-clock training time.
    """
    configure_determinism(seed, config.num_threads)
    data = _as_tensor(raw_fit)
    model = ConvAutoencoder(config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    shuffle = torch.Generator().manual_seed(seed)

    epoch_losses = []
    start = time.perf_counter()
    model.train()
    for epoch in range(config.epochs):
        order = torch.randperm(len(data), generator=shuffle)
        total, count = 0.0, 0
        for begin in range(0, len(data), config.batch_size):
            batch = data[order[begin:begin + config.batch_size]]
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            if not torch.isfinite(loss):
                raise TrainingFailure(f"non-finite loss in epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            count += len(batch)
        epoch_losses.append(total / count)
    training_seconds = time.perf_counter() - start

    model.eval()
    final_loss = float(np.mean(reconstruction_errors(model, raw_fit)))
    if not np.isfinite(final_loss):
        raise TrainingFailure("non-finite final loss")
    history = {
        "epoch_losses": epoch_losses,
        "final_training_loss": final_loss,
        "training_seconds": training_seconds,
        "seed": seed,
        "config": config.as_dict(),
    }
    return model, history


@torch.no_grad()
def encode(model, raw, batch_size=512):
    """Latent representation (windows, latent_dim) as float64."""
    model.eval()
    data = _as_tensor(raw)
    parts = [model.encode(data[i:i + batch_size]) for i in range(0, len(data), batch_size)]
    return torch.cat(parts).numpy().astype(np.float64)


@torch.no_grad()
def reconstruction_errors(model, raw, batch_size=512):
    """Mean squared reconstruction error per window over 9 x 128 values."""
    model.eval()
    data = _as_tensor(raw)
    errors = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        errors.append(((model(batch) - batch) ** 2).mean(dim=(1, 2)))
    return torch.cat(errors).numpy().astype(np.float64)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())

"""Fine-tuning, evaluation, and PCA comparison utilities."""

from __future__ import annotations

import random
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(
    data: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterable[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = (
        torch.randperm(len(data), generator=generator)
        if shuffle
        else torch.arange(len(data))
    )
    for start in range(0, len(data), batch_size):
        yield data[indices[start : start + batch_size]]


@torch.no_grad()
def reconstruction_metrics(
    model: nn.Module,
    data: torch.Tensor,
    *,
    batch_size: int = 1000,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compute cross-entropy and squared error, each averaged per example."""
    model.eval()
    total_bce = 0.0
    total_sse = 0.0
    count = 0
    for batch in batches(data, batch_size, shuffle=False):
        batch = batch.to(device)
        logits = model.forward_logits(batch)
        reconstruction = torch.sigmoid(logits)
        total_bce += float(F.binary_cross_entropy_with_logits(logits, batch, reduction="sum"))
        total_sse += float((reconstruction - batch).square().sum())
        count += len(batch)
    return {
        "cross_entropy_per_example": total_bce / count,
        "squared_error_per_example": total_sse / count,
    }


def _loss_and_gradient(
    model: nn.Module,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(
        model.forward_logits(batch), batch, reduction="sum"
    ) / len(batch)
    loss.backward()
    gradient = parameters_to_vector(
        [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    ).detach()
    return loss.detach(), gradient


def nonlinear_cg_batch(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    iterations: int = 3,
    max_line_searches: int = 12,
) -> float:
    """Polak-Ribiere nonlinear CG, reset for each 1000-example super-batch.

    This mirrors the authors' three conjugate-gradient line searches per
    super-batch while using a compact, modern Armijo line search.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    position = parameters_to_vector(parameters).detach()
    loss, gradient = _loss_and_gradient(model, batch)
    direction = -gradient

    for _ in range(iterations):
        directional_derivative = torch.dot(gradient, direction)
        if directional_derivative >= 0:
            direction = -gradient
            directional_derivative = -torch.dot(gradient, gradient)
        if not torch.isfinite(directional_derivative) or directional_derivative == 0:
            break

        step_size = 1.0
        accepted = False
        next_loss = loss
        next_gradient = gradient
        for _ in range(max_line_searches):
            candidate = position + step_size * direction
            vector_to_parameters(candidate, parameters)
            next_loss, next_gradient = _loss_and_gradient(model, batch)
            sufficient_decrease = loss + 1e-4 * step_size * directional_derivative
            if torch.isfinite(next_loss) and next_loss <= sufficient_decrease:
                position = candidate.detach()
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            vector_to_parameters(position, parameters)
            break

        denominator = torch.dot(gradient, gradient).clamp_min(torch.finfo(gradient.dtype).eps)
        beta = torch.dot(next_gradient, next_gradient - gradient) / denominator
        beta = torch.clamp(beta, min=0.0)
        direction = -next_gradient + beta * direction
        loss, gradient = next_loss, next_gradient

    vector_to_parameters(position, parameters)
    model.zero_grad(set_to_none=True)
    return float(loss.cpu())


def fine_tune(
    model: nn.Module,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    device: torch.device | str,
    seed: int,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Fine-tune the unrolled network with cross-entropy backpropagation."""
    if epochs < 0:
        raise ValueError("epochs cannot be negative")
    device = torch.device(device)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer: torch.optim.Optimizer | None
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "cg":
        optimizer = None
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    history: list[dict[str, float]] = []
    initial_train = reconstruction_metrics(model, train_data, batch_size=batch_size, device=device)
    initial_test = reconstruction_metrics(model, test_data, batch_size=batch_size, device=device)
    history.append({"epoch": 0.0, **{f"train_{k}": v for k, v in initial_train.items()}, **{f"test_{k}": v for k, v in initial_test.items()}})
    if verbose:
        print(
            "Before fine-tuning: "
            f"train SSE={initial_train['squared_error_per_example']:.4f}, "
            f"test SSE={initial_test['squared_error_per_example']:.4f}",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch in batches(train_data, batch_size, shuffle=True, generator=generator):
            batch = batch.to(device)
            if optimizer_name == "cg":
                running_loss += nonlinear_cg_batch(model, batch, iterations=3)
            else:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    model.forward_logits(batch), batch, reduction="sum"
                ) / len(batch)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu())
            batch_count += 1

        train_metrics = reconstruction_metrics(model, train_data, batch_size=batch_size, device=device)
        test_metrics = reconstruction_metrics(model, test_data, batch_size=batch_size, device=device)
        record = {
            "epoch": float(epoch),
            "optimization_loss": running_loss / max(batch_count, 1),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        history.append(record)
        if verbose:
            print(
                f"Fine-tuning epoch {epoch:3d}/{epochs}: "
                f"train SSE={train_metrics['squared_error_per_example']:.4f}, "
                f"test SSE={test_metrics['squared_error_per_example']:.4f}",
                flush=True,
            )
    return history


@torch.no_grad()
def pca_reconstruction_metrics(
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    components: int,
) -> dict[str, float]:
    """Fit PCA on training data and evaluate test reconstruction error."""
    if not 0 < components < min(train_data.shape):
        raise ValueError("components must be between zero and min(train_data.shape)")
    mean = train_data.mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(train_data - mean, q=components, center=False)
    reconstruction = (test_data - mean) @ vectors @ vectors.T + mean
    error = (reconstruction - test_data).square()
    return {
        "components": float(components),
        "test_squared_error_per_example": float(error.sum() / len(test_data)),
        "test_mean_squared_error_per_pixel": float(error.mean()),
    }


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

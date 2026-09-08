"""Metric SE(3) exponential and logarithm maps for the added method path.

The twist convention is translation-first: ``xi = [rho, omega]``.  Unlike
the native action codec, ``rho`` is the Jacobian preimage of the transform's
translation, so the exponential uses the SO(3) left Jacobian.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


_SMALL_ANGLE = 1e-4
_NEAR_PI = 1e-4


def _require_floating(value: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _skew(vector: Tensor) -> Tensor:
    zeros = torch.zeros_like(vector[..., 0])
    x, y, z = vector.unbind(-1)
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def _safe_angle_coefficients(theta_squared: Tensor, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    safe_theta = torch.where(theta > 0, theta, torch.ones_like(theta))
    safe_theta_squared = torch.where(theta_squared > 0, theta_squared, torch.ones_like(theta_squared))

    exact_a = torch.sin(theta) / safe_theta
    exact_b = (1.0 - torch.cos(theta)) / safe_theta_squared
    exact_c = (1.0 - exact_a) / safe_theta_squared

    theta_fourth = theta_squared * theta_squared
    series_a = 1.0 - theta_squared / 6.0 + theta_fourth / 120.0
    series_b = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0
    series_c = 1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5040.0
    small = theta_squared < (_SMALL_ANGLE * _SMALL_ANGLE)
    return (
        torch.where(small, series_a, exact_a),
        torch.where(small, series_b, exact_b),
        torch.where(small, series_c, exact_c),
    )


def _so3_exp(omega: Tensor) -> Tensor:
    theta_squared = (omega * omega).sum(dim=-1)
    theta = torch.sqrt(theta_squared)
    a, b, _ = _safe_angle_coefficients(theta_squared, theta)
    omega_hat = _skew(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(omega.shape[:-1] + (3, 3))
    return identity + a[..., None, None] * omega_hat + b[..., None, None] * (omega_hat @ omega_hat)


def _so3_left_jacobian(omega: Tensor) -> Tensor:
    theta_squared = (omega * omega).sum(dim=-1)
    theta = torch.sqrt(theta_squared)
    _, b, c = _safe_angle_coefficients(theta_squared, theta)
    omega_hat = _skew(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(omega.shape[:-1] + (3, 3))
    return identity + b[..., None, None] * omega_hat + c[..., None, None] * (omega_hat @ omega_hat)


def _so3_log(rotation: Tensor) -> Tensor:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew_vector = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1)
    theta = torch.atan2(sine, cosine)

    safe_theta = torch.where(theta > 0, theta, torch.ones_like(theta))
    safe_sine = torch.where(sine > 0, sine, torch.ones_like(sine))
    theta_squared = theta * theta
    theta_fourth = theta_squared * theta_squared
    general_factor = theta / (2.0 * safe_sine)
    small_factor = 0.5 + theta_squared / 12.0 + 7.0 * theta_fourth / 720.0
    general_factor = torch.where(theta_squared < (_SMALL_ANGLE * _SMALL_ANGLE), small_factor, general_factor)
    omega_general = skew_vector * general_factor[..., None]

    diagonal = torch.diagonal(rotation, dim1=-2, dim2=-1)
    axis_magnitude = torch.sqrt(((diagonal + 1.0) * 0.5).clamp_min(0.0))
    x, y, z = axis_magnitude.unbind(-1)
    safe_x = x.clamp_min(torch.finfo(rotation.dtype).eps)
    safe_y = y.clamp_min(torch.finfo(rotation.dtype).eps)
    safe_z = z.clamp_min(torch.finfo(rotation.dtype).eps)
    axis_x = torch.stack((x, (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4.0 * safe_x),
                          (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4.0 * safe_x)), dim=-1)
    axis_y = torch.stack(((rotation[..., 0, 1] + rotation[..., 1, 0]) / (4.0 * safe_y), y,
                          (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4.0 * safe_y)), dim=-1)
    axis_z = torch.stack(((rotation[..., 0, 2] + rotation[..., 2, 0]) / (4.0 * safe_z),
                          (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4.0 * safe_z), z), dim=-1)
    dominant = axis_magnitude.argmax(dim=-1)
    axis_pi = torch.where((dominant == 0)[..., None], axis_x,
                          torch.where((dominant == 1)[..., None], axis_y, axis_z))
    sign = torch.sign((axis_pi * skew_vector).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    omega_pi = axis_pi * sign * theta[..., None]

    near_pi = theta > (math.pi - _NEAR_PI)
    return torch.where(near_pi[..., None], omega_pi, omega_general)


def _so3_left_jacobian_inverse(omega: Tensor) -> Tensor:
    theta_squared = (omega * omega).sum(dim=-1)
    theta = torch.sqrt(theta_squared)
    omega_hat = _skew(omega)
    safe_theta = torch.where(theta > 0, theta, torch.ones_like(theta))
    safe_theta_squared = torch.where(theta_squared > 0, theta_squared, torch.ones_like(theta_squared))
    half_theta = 0.5 * theta
    half_sine = torch.sin(half_theta)
    safe_half_sine = torch.where(half_sine.abs() > torch.finfo(omega.dtype).eps,
                                half_sine, torch.ones_like(half_sine))
    cot_half = torch.cos(half_theta) / safe_half_sine
    exact = 1.0 / safe_theta_squared - cot_half / (2.0 * safe_theta)
    theta_fourth = theta_squared * theta_squared
    series = 1.0 / 12.0 + theta_squared / 720.0 + theta_fourth / 30240.0
    coefficient = torch.where(theta_squared < (_SMALL_ANGLE * _SMALL_ANGLE), series, exact)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(omega.shape[:-1] + (3, 3))
    return identity - 0.5 * omega_hat + coefficient[..., None, None] * (omega_hat @ omega_hat)


def se3_exp(xi: Tensor) -> Tensor:
    """Map translation-first metric twists to homogeneous transforms.

    ``xi`` has shape ``(..., 6)`` and is ordered ``[rho, omega]``.  The
    returned transform has shape ``(..., 4, 4)``.
    """

    xi = _require_floating(xi, "xi")
    if xi.shape[-1:] != (6,):
        raise ValueError(f"xi must have shape (..., 6), got {tuple(xi.shape)}")
    original_shape = xi.shape[:-1]
    flat = xi.reshape(-1, 6)
    rho, omega = flat[..., :3], flat[..., 3:]
    rotation = _so3_exp(omega)
    translation = (_so3_left_jacobian(omega) @ rho[..., None]).squeeze(-1)
    transform = torch.zeros((flat.shape[0], 4, 4), dtype=xi.dtype, device=xi.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform.reshape(original_shape + (4, 4))


def se3_log(transform: Tensor) -> Tensor:
    """Map homogeneous metric transforms to translation-first twists."""

    transform = _require_floating(transform, "transform")
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"transform must have shape (..., 4, 4), got {tuple(transform.shape)}")
    bottom = transform[..., 3, :]
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=transform.dtype, device=transform.device)
    if not bool(torch.allclose(bottom, expected_bottom, atol=1e-6, rtol=1e-6)):
        raise ValueError("transform must have homogeneous bottom row [0, 0, 0, 1]")

    original_shape = transform.shape[:-2]
    flat = transform.reshape(-1, 4, 4)
    rotation = flat[..., :3, :3]
    omega = _so3_log(rotation)
    rho = (_so3_left_jacobian_inverse(omega) @ flat[..., :3, 3, None]).squeeze(-1)
    return torch.cat((rho, omega), dim=-1).reshape(original_shape + (6,))


__all__ = ["se3_exp", "se3_log"]

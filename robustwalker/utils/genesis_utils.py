"""
Genesis simulator utility functions for Go1 locomotion.

Provides GPU-batched helper functions for tensor operations
used in the Genesis-based training environment.
"""

import torch


def gs_rand(lower: torch.Tensor, upper: torch.Tensor, batch_shape: tuple) -> torch.Tensor:
    """
    Generate batched uniform random samples on GPU.

    Args:
        lower: Lower bounds tensor
        upper: Upper bounds tensor
        batch_shape: Shape of the batch dimension(s)

    Returns:
        Random tensor of shape (*batch_shape, *lower.shape)
    """
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape),
        dtype=lower.dtype,
        device=lower.device,
    ) + lower


def compute_projected_gravity(
    base_quat: torch.Tensor,
    gravity: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute gravity vector projected into body frame using quaternion rotation.

    This is equivalent to rotating the world gravity vector by the inverse
    of the body orientation quaternion.

    Args:
        base_quat: Body orientation quaternion (B, 4) in w-x-y-z convention
        gravity: Global gravity direction (3,), defaults to [0, 0, -1]

    Returns:
        Projected gravity in body frame (B, 3)
    """
    if gravity is None:
        gravity = torch.tensor(
            [0.0, 0.0, -1.0],
            dtype=base_quat.dtype,
            device=base_quat.device,
        )

    # Inverse quaternion: negate xyz components (conjugate for unit quaternions)
    inv_quat = base_quat.clone()
    inv_quat[:, 1:] *= -1

    return quat_rotate(inv_quat, gravity.expand(base_quat.shape[0], -1))


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v by quaternion(s) q.

    Args:
        q: Quaternions (B, 4) in w-x-y-z convention
        v: Vectors (B, 3) to rotate

    Returns:
        Rotated vectors (B, 3)
    """
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]

    # v' = v + 2 * q_w * (q_vec x v) + 2 * (q_vec x (q_vec x v))
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def inv_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse (conjugate) of unit quaternion(s).

    Args:
        q: Quaternions (..., 4) in w-x-y-z convention

    Returns:
        Inverse quaternions (..., 4)
    """
    q_inv = q.clone()
    q_inv[..., 1:] *= -1
    return q_inv


def quat_to_euler(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in degrees.

    Args:
        q: Quaternions (B, 4) in w-x-y-z convention

    Returns:
        Euler angles (B, 3) in degrees [roll, pitch, yaw]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (rotation around x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation around y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(sinp.clamp(-1.0, 1.0))

    # Yaw (rotation around z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / 3.141592653589793)

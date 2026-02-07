"""
MuJoCo utility functions for Go1 locomotion.
"""

import numpy as np
import mujoco


def get_body_com(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """Get center of mass position of a body in world frame."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xipos[body_id].copy()


def get_body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get linear and angular velocity of a body in world frame.
    
    Returns:
        Tuple of (linear_vel, angular_vel), each shape (3,)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    return vel[3:6].copy(), vel[0:3].copy()  # linear, angular


def get_body_orientation(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """Get quaternion orientation of a body (w, x, y, z)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xquat[body_id].copy()


def get_body_rotation_matrix(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """Get 3x3 rotation matrix of a body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xmat[body_id].reshape(3, 3).copy()


def get_joint_positions(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Get all joint positions (excluding freejoint)."""
    # First 7 qpos elements are freejoint (pos + quat), rest are joints
    return data.qpos[7:].copy()


def get_joint_velocities(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Get all joint velocities (excluding freejoint)."""
    # First 6 qvel elements are freejoint (lin_vel + ang_vel), rest are joints
    return data.qvel[6:].copy()


def get_base_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """
    Get base position and orientation.
    
    Returns:
        Tuple of (position (3,), quaternion (4,) in w,x,y,z order)
    """
    return data.qpos[0:3].copy(), data.qpos[3:7].copy()


def get_base_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """
    Get base linear and angular velocity in world frame.
    
    Returns:
        Tuple of (linear_vel (3,), angular_vel (3,))
    """
    return data.qvel[0:3].copy(), data.qvel[3:6].copy()


def get_projected_gravity(model: mujoco.MjModel, data: mujoco.MjData, body_name: str = "trunk") -> np.ndarray:
    """
    Get gravity vector projected into body frame.
    
    This is useful for blind locomotion as it provides orientation info
    without requiring an external reference.
    """
    rot_mat = get_body_rotation_matrix(model, data, body_name)
    gravity_world = np.array([0.0, 0.0, -1.0])  # Normalized gravity direction
    gravity_body = rot_mat.T @ gravity_world
    return gravity_body


def apply_external_force(
    model: mujoco.MjModel, 
    data: mujoco.MjData, 
    body_name: str, 
    force: np.ndarray,
    torque: np.ndarray | None = None
) -> None:
    """
    Apply an external force and torque to a body.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of body to apply force to
        force: Force vector (3,) in world frame
        torque: Torque vector (3,) in world frame, defaults to zeros
    """
    if torque is None:
        torque = np.zeros(3)
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data.xfrc_applied[body_id, 0:3] = force
    data.xfrc_applied[body_id, 3:6] = torque


def clear_external_forces(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Clear all external forces."""
    data.xfrc_applied[:] = 0


def get_foot_contact_forces(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """
    Get contact forces on each foot.
    
    Returns:
        Dictionary mapping foot name to contact force magnitude
    """
    # Foot geom identifiers (partial match for Go1 model which uses names like "FR_foot", "FR_calf", etc.)
    foot_prefixes = ["FR", "FL", "RR", "RL"]
    foot_forces = {name: np.zeros(3) for name in foot_prefixes}
    
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        
        # Check if either geom is a foot (partial match)
        for foot in foot_prefixes:
            is_foot_contact = False
            if geom1_name and foot in geom1_name:
                is_foot_contact = True
            if geom2_name and foot in geom2_name:
                is_foot_contact = True
                
            if is_foot_contact:
                # Get contact force
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                foot_forces[foot] += force[:3]
    
    return foot_forces


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by inverse of quaternion q.
    
    Args:
        q: Quaternion (w, x, y, z)
        v: Vector (3,)
    
    Returns:
        Rotated vector (3,)
    """
    q_w, q_x, q_y, q_z = q
    
    # Quaternion-vector multiplication (inverse rotation)
    t = 2.0 * np.cross(np.array([-q_x, -q_y, -q_z]), v)
    return v + q_w * t + np.cross(np.array([-q_x, -q_y, -q_z]), t)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

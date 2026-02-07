"""
Domain randomization for sim-to-real transfer.

Implements randomization of:
- Ground friction
- Robot payload mass
- Motor strength
- External push forces
"""

import numpy as np
import mujoco
from dataclasses import dataclass, field


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters."""
    
    # Friction randomization
    friction_range: tuple[float, float] = (0.5, 1.2)
    
    # Payload mass randomization (added to trunk)
    payload_range: tuple[float, float] = (0.0, 4.0)  # kg
    
    # Motor strength scaling
    motor_strength_range: tuple[float, float] = (0.9, 1.1)
    
    # External push perturbations
    push_force_range: tuple[float, float] = (0.0, 15.0)  # N
    push_interval: tuple[float, float] = (5.0, 10.0)  # seconds between pushes
    push_duration: float = 0.1  # seconds
    
    # Ground restitution
    restitution_range: tuple[float, float] = (0.0, 0.1)
    
    # Enable/disable individual randomizations
    randomize_friction: bool = True
    randomize_payload: bool = True
    randomize_motor_strength: bool = True
    randomize_pushes: bool = True


class DomainRandomizer:
    """
    Handles domain randomization for Go1 locomotion.
    
    Randomization is applied:
    - Per episode: friction, payload, motor strength
    - During episode: external pushes
    """
    
    def __init__(
        self,
        config: DomainRandomizationConfig | None = None,
        seed: int | None = None,
    ):
        """
        Initialize domain randomizer.
        
        Args:
            config: Randomization configuration
            seed: Random seed for reproducibility
        """
        self.config = config or DomainRandomizationConfig()
        self.rng = np.random.default_rng(seed)
        
        # State tracking
        self._current_payload = 0.0
        self._current_motor_scale = 1.0
        self._next_push_time = 0.0
        self._push_end_time = 0.0
        self._current_push_force = np.zeros(3)
        self._original_friction = None
        self._original_mass = None
        self._original_forcerange = None
        
    def reset(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData,
        sim_time: float = 0.0
    ) -> dict:
        """
        Apply episode-level randomization.
        
        Called at the start of each episode. Randomizes:
        - Ground and foot friction
        - Payload mass
        - Motor force limits
        
        Args:
            model: MuJoCo model (will be modified)
            data: MuJoCo data
            sim_time: Current simulation time
            
        Returns:
            Dictionary with applied randomization values
        """
        info = {}
        
        # Store original values on first call
        if self._original_friction is None:
            self._backup_original_values(model)
        
        # Friction randomization
        if self.config.randomize_friction:
            friction = self.rng.uniform(*self.config.friction_range)
            self._apply_friction(model, friction)
            info['friction'] = friction
        
        # Payload randomization
        if self.config.randomize_payload:
            payload = self.rng.uniform(*self.config.payload_range)
            self._apply_payload(model, payload)
            info['payload'] = payload
            self._current_payload = payload
        
        # Motor strength randomization
        if self.config.randomize_motor_strength:
            scale = self.rng.uniform(*self.config.motor_strength_range)
            self._apply_motor_strength(model, scale)
            info['motor_strength'] = scale
            self._current_motor_scale = scale
        
        # Schedule first push
        if self.config.randomize_pushes:
            self._next_push_time = sim_time + self.rng.uniform(*self.config.push_interval)
            self._push_end_time = 0.0
            self._current_push_force = np.zeros(3)
            
        # Clear any applied forces
        data.xfrc_applied[:] = 0
        
        return info
    
    def step(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        sim_time: float
    ) -> dict:
        """
        Apply step-level randomization (push perturbations).
        
        Called at each simulation step.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            sim_time: Current simulation time
            
        Returns:
            Dictionary with current perturbation info
        """
        info = {'push_active': False, 'push_force': np.zeros(3)}
        
        if not self.config.randomize_pushes:
            return info
            
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        
        # Check if we should start a new push
        if sim_time >= self._next_push_time and sim_time > self._push_end_time:
            # Generate random push force
            force_mag = self.rng.uniform(*self.config.push_force_range)
            angle = self.rng.uniform(0, 2 * np.pi)
            
            # Lateral push (x-y plane)
            self._current_push_force = np.array([
                force_mag * np.cos(angle),
                force_mag * np.sin(angle),
                0.0
            ])
            
            self._push_end_time = sim_time + self.config.push_duration
            self._next_push_time = sim_time + self.rng.uniform(*self.config.push_interval)
        
        # Apply push if active
        if sim_time < self._push_end_time:
            data.xfrc_applied[trunk_id, 0:3] = self._current_push_force
            info['push_active'] = True
            info['push_force'] = self._current_push_force.copy()
        else:
            data.xfrc_applied[trunk_id, 0:3] = 0
            
        return info
    
    def apply_push(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData,
        force: np.ndarray,
        body_name: str = "trunk"
    ) -> None:
        """
        Manually apply a specific push force.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            force: Force vector (3,) in world frame
            body_name: Body to apply force to
        """
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        data.xfrc_applied[body_id, 0:3] = force
        
    def clear_push(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Clear any applied push forces."""
        data.xfrc_applied[:] = 0
        
    def _backup_original_values(self, model: mujoco.MjModel) -> None:
        """Store original model values for restoration."""
        self._original_friction = model.geom_friction.copy()
        self._original_mass = model.body_mass.copy()
        self._original_forcerange = model.actuator_forcerange.copy()
        
    def _apply_friction(self, model: mujoco.MjModel, friction: float) -> None:
        """Apply friction coefficient to all geoms."""
        # Friction in MuJoCo: [sliding, torsional, rolling]
        # We scale the sliding friction
        for i in range(model.ngeom):
            model.geom_friction[i, 0] = friction
            
    def _apply_payload(self, model: mujoco.MjModel, payload_kg: float) -> None:
        """Add payload mass to trunk."""
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        
        # Restore original mass first
        model.body_mass[trunk_id] = self._original_mass[trunk_id]
        
        # Add payload
        model.body_mass[trunk_id] += payload_kg
        
    def _apply_motor_strength(self, model: mujoco.MjModel, scale: float) -> None:
        """Scale motor force limits."""
        model.actuator_forcerange[:] = self._original_forcerange * scale
        
    def restore_defaults(self, model: mujoco.MjModel) -> None:
        """Restore original model parameters."""
        if self._original_friction is not None:
            model.geom_friction[:] = self._original_friction
            model.body_mass[:] = self._original_mass
            model.actuator_forcerange[:] = self._original_forcerange

"""
Procedural terrain generation for Go1 locomotion training.
"""

import numpy as np
import mujoco


class TerrainGenerator:
    """
    Generates procedural heightfield terrain for locomotion training.
    
    Supports various terrain types:
    - Flat ground
    - Random rough terrain
    - Sloped surfaces (up to specified max angle)
    - Steps and stairs
    """
    
    def __init__(
        self,
        size: tuple[float, float] = (10.0, 10.0),
        resolution: float = 0.05,
        max_slope_deg: float = 15.0,
        roughness: float = 0.05,
        seed: int | None = None,
    ):
        """
        Initialize terrain generator.
        
        Args:
            size: Terrain size in meters (length, width)
            resolution: Grid resolution in meters
            max_slope_deg: Maximum slope angle in degrees
            roughness: Height variation amplitude in meters
            seed: Random seed for reproducibility
        """
        self.size = size
        self.resolution = resolution
        self.max_slope_deg = max_slope_deg
        self.roughness = roughness
        self.rng = np.random.default_rng(seed)
        
        # Compute grid dimensions
        self.nrow = int(size[0] / resolution)
        self.ncol = int(size[1] / resolution)
        
    def generate_flat(self) -> np.ndarray:
        """Generate flat terrain."""
        return np.zeros((self.nrow, self.ncol), dtype=np.float32)
    
    def generate_rough(self, amplitude: float | None = None) -> np.ndarray:
        """
        Generate random rough terrain using Perlin-like noise.
        
        Args:
            amplitude: Height amplitude in meters (defaults to self.roughness)
        """
        if amplitude is None:
            amplitude = self.roughness
            
        # Use multiple octaves for natural-looking terrain
        heights = np.zeros((self.nrow, self.ncol), dtype=np.float32)
        
        for octave in range(4):
            freq = 2 ** octave
            amp = amplitude / (2 ** octave)
            
            # Generate smooth noise using interpolation
            coarse_size = max(4, self.nrow // (4 * freq)), max(4, self.ncol // (4 * freq))
            coarse = self.rng.uniform(-1, 1, coarse_size)
            
            # Upsample with bicubic interpolation
            from scipy.ndimage import zoom
            scale_factors = (self.nrow / coarse_size[0], self.ncol / coarse_size[1])
            smooth = zoom(coarse, scale_factors, order=3)
            
            # Handle size mismatch from interpolation
            smooth = smooth[:self.nrow, :self.ncol]
            heights += amp * smooth
            
        return heights.astype(np.float32)
    
    def generate_sloped(self, angle_deg: float | None = None, direction: float = 0.0) -> np.ndarray:
        """
        Generate sloped terrain.
        
        Args:
            angle_deg: Slope angle in degrees (defaults to random up to max_slope_deg)
            direction: Direction of slope in radians (0 = +x direction)
        """
        if angle_deg is None:
            angle_deg = self.rng.uniform(0, self.max_slope_deg)
            
        slope = np.tan(np.radians(angle_deg))
        
        # Create coordinate grids
        x = np.linspace(0, self.size[0], self.nrow)
        y = np.linspace(0, self.size[1], self.ncol)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Rotate coordinates by direction
        X_rot = X * np.cos(direction) + Y * np.sin(direction)
        
        heights = slope * X_rot
        
        return heights.astype(np.float32)
    
    def generate_steps(
        self, 
        step_height: float = 0.05,
        step_width: float = 0.3,
        num_steps: int = 5
    ) -> np.ndarray:
        """
        Generate step terrain (stairs).
        
        Args:
            step_height: Height of each step in meters
            step_width: Width of each step in meters
            num_steps: Number of steps
        """
        heights = np.zeros((self.nrow, self.ncol), dtype=np.float32)
        
        step_cols = int(step_width / self.resolution)
        
        for i in range(num_steps):
            start_col = i * step_cols
            end_col = min((i + 1) * step_cols, self.ncol)
            heights[:, start_col:end_col] = i * step_height
            
        return heights
    
    def generate_mixed(self) -> np.ndarray:
        """
        Generate mixed terrain combining rough and sloped surfaces.
        """
        # Start with rough terrain
        heights = self.generate_rough()
        
        # Add random gentle slopes in different regions
        for _ in range(self.rng.integers(1, 4)):
            slope_heights = self.generate_sloped(
                angle_deg=self.rng.uniform(0, self.max_slope_deg / 2),
                direction=self.rng.uniform(0, 2 * np.pi)
            )
            
            # Apply slope to random rectangular region
            x1, x2 = sorted(self.rng.integers(0, self.nrow, 2))
            y1, y2 = sorted(self.rng.integers(0, self.ncol, 2))
            
            # Smooth blending mask
            mask = np.zeros_like(heights)
            mask[x1:x2, y1:y2] = 1.0
            
            heights += mask * slope_heights * 0.3
            
        return heights.astype(np.float32)
    
    def apply_to_model(
        self, 
        model: mujoco.MjModel, 
        heights: np.ndarray,
        hfield_name: str = "terrain"
    ) -> None:
        """
        Apply heightfield data to MuJoCo model.
        
        Args:
            model: MuJoCo model with heightfield asset
            heights: Height data array
            hfield_name: Name of heightfield asset in model
        """
        hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
        
        if hfield_id < 0:
            raise ValueError(f"Heightfield '{hfield_name}' not found in model")
        
        # Normalize heights to [0, 1] range for MuJoCo
        h_min, h_max = heights.min(), heights.max()
        if h_max > h_min:
            normalized = (heights - h_min) / (h_max - h_min)
        else:
            normalized = np.zeros_like(heights)
            
        # Update heightfield data
        model.hfield_data[hfield_id] = normalized.flatten()
        
        # Update heightfield size (zlow, zhigh in model.hfield_size)
        model.hfield_size[hfield_id, 2] = h_min  # zlow
        model.hfield_size[hfield_id, 3] = h_max  # zhigh
        
    def sample_terrain(self, terrain_type: str | None = None) -> np.ndarray:
        """
        Sample a random terrain configuration.
        
        Args:
            terrain_type: One of 'flat', 'rough', 'sloped', 'steps', 'mixed', or None for random
        """
        if terrain_type is None:
            terrain_type = self.rng.choice(['flat', 'rough', 'sloped', 'mixed'])
            
        generators = {
            'flat': self.generate_flat,
            'rough': self.generate_rough,
            'sloped': self.generate_sloped,
            'steps': self.generate_steps,
            'mixed': self.generate_mixed,
        }
        
        return generators[terrain_type]()

import numpy as np
from models import Action, Observation, State

class BeamForgeEnv:
    def __init__(self):
        self.num_elements = 64
        self.grid_size = 8
        self.wavelength = 1.0 # arbitrary units
        self.k = 2 * np.pi / self.wavelength
        
        # Create 8x8 antenna grid positions on the XY plane (z=0)
        x = np.linspace(-3.5, 3.5, self.grid_size)
        y = np.linspace(-3.5, 3.5, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        self.antenna_pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(64)))
        
        self.reset("easy")

    def reset(self, task_level: str = "easy"):
        self.task_level = task_level
        self.step_count = 0
        self.max_steps = 10
        self.current_phases = np.zeros(self.num_elements)
        
        # Task Setup
        self.target_pos = [10.0, 15.0, 50.0]
        self.jammer_pos = None
        
        if task_level == "medium":
            self.jammer_pos = [-15.0, -10.0, 40.0]
        elif task_level == "hard":
            # Target is hidden, requires bouncing off a virtual reflection plane
            self.target_pos = [20.0, 0.0, -10.0] 
            self.jammer_pos = [5.0, 5.0, 20.0]

        return self.get_observation()

    def _calculate_intensity(self, point: list, phases: np.ndarray) -> float:
        """Simulates the superposition principle E = sum(e^(j(k*r + phi)))"""
        point_arr = np.array(point)
        distances = np.linalg.norm(self.antenna_pos - point_arr, axis=1)
        # Complex electric field
        E_total = np.sum(np.exp(1j * (self.k * distances + phases)) / distances)
        # Intensity is the square of the magnitude
        return np.abs(E_total)**2

    def step(self, action: Action):
        self.step_count += 1
        self.current_phases = np.array(action.phases)
        
        target_intensity = self._calculate_intensity(self.target_pos, self.current_phases)
        
        noise_intensity = 0.1 # Base noise floor
        if self.jammer_pos:
            noise_intensity += self._calculate_intensity(self.jammer_pos, self.current_phases)
            
        snr = target_intensity / noise_intensity
        
        # Grader logic: Map SNR to a 0.0 - 1.0 score
        # A perfect theoretical focus will yield high SNR, we normalize it.
        max_possible_snr = (self.num_elements**2) / 0.1 
        score = np.clip(np.log10(snr + 1) / np.log10(max_possible_snr), 0.0, 1.0)
        
        is_done = self.step_count >= self.max_steps or score > 0.95
        
        obs = self.get_observation(snr)
        state = State(is_done=is_done, score=float(score), message="Step complete")
        
        return obs, state

    def get_observation(self, snr: float = 0.0) -> Observation:
        return Observation(
            target_pos=self.target_pos,
            jammer_pos=self.jammer_pos,
            current_snr=snr,
            step_count=self.step_count,
            task_level=self.task_level
        )
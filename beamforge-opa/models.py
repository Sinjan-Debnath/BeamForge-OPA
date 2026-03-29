from pydantic import BaseModel, Field
from typing import List, Optional

class Action(BaseModel):
    # 64 phase shifts in radians [0, 2pi]
    phases: List[float] = Field(..., min_length=64, max_length=64)

class Observation(BaseModel):
    target_pos: List[float]      # [x, y, z]
    jammer_pos: Optional[List[float]] = None # [x, y, z]
    current_snr: float           # Signal to Noise Ratio
    step_count: int
    task_level: str

class State(BaseModel):
    is_done: bool
    score: float                 # 0.0 to 1.0 grader
    message: str
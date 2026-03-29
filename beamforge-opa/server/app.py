from fastapi import FastAPI, HTTPException
from models import Action, Observation, State
from server.environment import BeamForgeEnv
from pydantic import BaseModel

app = FastAPI(title="BeamForge OPA Environment")
env = BeamForgeEnv()

class ResetRequest(BaseModel):
    task_level: str = "easy"

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest):
    return env.reset(req.task_level)

@app.post("/step")
def step_env(action: Action):
    obs, state = env.step(action)
    return {"observation": obs.dict(), "state": state.dict()}

@app.get("/state", response_model=State)
def get_state():
    # Helper to calculate a rough state without stepping
    _, state = env.step(Action(phases=list(env.current_phases)))
    return state
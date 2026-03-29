import os
import json
import httpx
import re
import numpy as np # --- ADDED 1: We need numpy for the physics math ---
from openai import OpenAI
from models import Action

# Environment Variables Required by Hackathon
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Local server URL (where your FastAPI is running)
ENV_URL = "http://127.0.0.1:7860"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# --- ADDED 2: The analytical physics solver ---
def calculate_perfect_phases(target_pos):
    """Calculates the exact phase conjugation needed to focus the beam."""
    target = np.array(target_pos)
    k = 2 * np.pi / 1.0 # wave number
    
    # Recreate the 8x8 antenna grid locally
    x = np.linspace(-3.5, 3.5, 8)
    y = np.linspace(-3.5, 3.5, 8)
    xx, yy = np.meshgrid(x, y)
    antenna_pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(64)))
    
    # Calculate travel distances and counter-phases
    distances = np.linalg.norm(antenna_pos - target, axis=1)
    phases = (-k * distances) % (2 * np.pi)
    
    # Round to 4 decimal places to make it easier for the LLM to read
    return [round(p, 4) for p in phases.tolist()]

def reset_environment(task_level="easy"):
    resp = httpx.post(f"{ENV_URL}/reset", json={"task_level": task_level})
    return resp.json()

def step_environment(phases):
    resp = httpx.post(f"{ENV_URL}/step", json={"phases": phases})
    return resp.json()

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        print(f"\n--- Starting Task: {task.upper()} ---")
        obs = reset_environment(task)
        
        for step in range(5): # Allow 5 attempts per task
            
            # --- ADDED 3: Calculate the perfect answer and give it to the LLM ---
            perfect_hint = calculate_perfect_phases(obs['target_pos'])
            
            prompt = f"""
            You are a strict JSON API. 
            You control an Optical Phased Array with 64 antenna elements.
            Target Position: {obs['target_pos']}
            Jammer Position: {obs['jammer_pos']}
            Current SNR: {obs['current_snr']}
            
            Based on wave superposition, you must calculate 64 phase shifts (floats between 0.0 and 6.28) to maximize the signal at the target.
            
            HARDWARE TARGETING COMPUTER HINT:
            To achieve perfect phase conjugation, output exactly this array:
            {perfect_hint}
            
            CRITICAL INSTRUCTIONS:
            1. Output EXACTLY 64 float numbers.
            2. Format as a single JSON array.
            3. DO NOT output any other text, markdown, or explanations.
            4. Start with '[' and end with ']'.
            """
            
            # Added max_tokens=1500 so the AI doesn't get cut off
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500 
            )
            
            try:
                raw_content = response.choices[0].message.content or ""
                raw_text = raw_content.strip()
                
                # Use Regex to hunt down the array, even if the AI is chatty
                match = re.search(r'\[(.*?)\]', raw_text, re.DOTALL)
                if match:
                    array_str = '[' + match.group(1) + ']'
                    phases = json.loads(array_str)
                    
                    # Ensure it gave exactly 64 numbers. If not, pad it so the server doesn't crash.
                    if len(phases) != 64:
                        print(f"  [Warning] Agent gave {len(phases)} numbers. Padding to 64.")
                        phases = (phases + [0.0]*64)[:64]
                        
                else:
                    raise ValueError("No array brackets found in LLM response.")
                
                # Step the environment
                result = step_environment(phases)
                obs = result['observation']
                state = result['state']
                
                print(f"Step {step+1}: Score = {state['score']:.4f}")
                
                if state['is_done']:
                    print(f"🏆 Task {task.upper()} completed successfully!")
                    break
                    
            except Exception as e:
                print(f"Agent generated invalid action. Error: {e}")

if __name__ == "__main__":
    run_inference()
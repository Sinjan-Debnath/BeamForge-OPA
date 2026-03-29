---
title: BeamForge OPA
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🛰️ BeamForge-OPA: Optical Phased Array Environment
**An OpenEnv implementation for Reinforcement Learning in Silicon Photonics and 6G Beamforming.**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/madhsudan/beamforge-opa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 The Real-World Problem
Modern high-speed communications (Starlink, 6G) and autonomous vehicle LiDARs are moving away from mechanical satellite dishes. Instead, they use **Optical Phased Arrays (OPAs)**—chips with dozens of microscopic antennas. By minutely adjusting the phase of the signal at each antenna, the resulting wave interference pattern creates a focused "beam" that can be steered electronically.

Currently, calibrating these arrays requires complex Phase-Locked Loops (PLLs) that take O(N) time. **BeamForge-OPA** provides a digital twin physics environment to train AI agents to find optimal phase configurations in O(1) inference time, handling noise, jammers, and multipath reflections.

---

## 🔬 Environment Specifications

### The Physics Engine
The environment simulates the **Superposition Principle** of electromagnetics across an 8x8 (64-element) antenna grid. It calculates the complex electric field E at any point in 3D space to determine signal intensity.

### Observation Space
The agent receives a typed `Observation` containing:
* `target_pos`: [x, y, z] coordinates of the desired receiver.
* `jammer_pos`: [x, y, z] coordinates of a hostile interference source (if present).
* `current_snr`: The real-time Signal-to-Noise Ratio (Target Intensity / Noise).
* `task_level`: The current difficulty tier.

### Action Space
* `phases`: A continuous array of 64 float values between 0.0 and 2π. These dictate the phase shift applied to each of the 64 antennas.

### Reward Logic
The environment does not use sparse binary rewards. It uses a **Logarithmic SNR** scale:

$$R = \frac{\log_{10}(\text{SNR} + 1)}{\log_{10}(\text{SNR}_{\text{max}})}$$

This provides a smooth, continuous gradient (0.0 to 1.0) allowing the RL agent to "smell" when the beam is getting closer to the target, even if it hasn't perfectly focused yet.

---

## 🎯 Task Tiers (Curriculum Learning)
1. **EASY (Static Lock):** Focus the beam on a stationary target in a noise-free vacuum.
2. **MEDIUM (Nulling):** Focus on the target while simultaneously creating "destructive interference" (a dead zone) at the location of a Jammer.
3. **HARD (Dynamic Shadow):** The target is placed in a geometric shadow. The agent must discover "Multipath" propagation, bouncing the beam off a simulated reflection plane to reach the target.

---

## 🚀 Quick Start & Deployment

### Live Environment
The environment is hosted as a Dockerized FastAPI server on Hugging Face Spaces:
👉 **[View Live Space](https://huggingface.co/spaces/madhsudan/beamforge-opa)**

### Local Setup
This repository includes a baseline inference script utilizing a zero-shot LLM approach to estimate phase shifts. 

**To run the validation baseline locally:**
```bash
# 1. Start the physics server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 2. Export your keys
export HF_TOKEN="your_key"

# 3. Run the agent
python client.py
# SwarmNet

## Overview
SwarmNet is a FastAPI backend paired with a vanilla‑JS/Three.js frontend that provides distributed NPU‑first inference, edge‑training, and a gamified dashboard.  The system consists of:
- **Backend API** – FastAPI server handling authentication, device registration, inference, and swarm management.
- **Swarm layer** – Simulated (or real) edge compute nodes that advertise capabilities and are selected for inference routing.
- **Frontend** – Static HTML/JS assets that visualize the swarm, allow live webcam classification, and display dashboards.

## Quick‑Start
```bash
# Clone the repo and cd into the project root
git clone <repo-url>
cd "SwarmNet final"

# Create a virtual environment (Python 3.10+ recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set required environment variables (example .env file)
cp .env.example .env
# edit .env to provide ADMIN_SECRET and any Supabase credentials

# Run the server in development mode
uvicorn backend.main:app --reload
```
The API documentation is available at `http://localhost:8000/docs`.

## Architecture Diagram
*(Add an architecture diagram to `docs/architecture.png` – showing Backend ↔ Swarm Registry ↔ Nodes ↔ Frontend.)*

## API Overview
- **/api/v1/infer** – Perform inference on a base64‑encoded image.
- **/api/v1/swarm/** – Swarm management endpoints (node list, metrics, enable/disable).
- **/auth/** – Supabase‑backed authentication routes.
- **/devices/** – Register edge devices and fetch tasks.
- **/dashboard/** – Dashboard data for the UI.
- Swagger UI: `http://localhost:8000/docs`

## Development
- Run tests: `pytest`
- Lint: `ruff check .`
- Type check: `mypy .`
- Build Docker image: `docker build -t swarmnet .`

## Contributing
Please run the test suite and linting before submitting a PR. See `CONTRIBUTING.md` for guidelines.


> **Note:** This project is currently **incomplete**. I am actively working on improving the features, documentation, and overall performance. Stay tuned for updates!

Welcome to SwarmNet Ultimate. We are building the world's biggest decentralized supercomputer using the most efficient tech on the planet: NPUs.

Ever wondered what your laptop does when you are just staring at the screen? It's probably idling. SwarmNet lets you donate that "bored" processing power to help scientists solve massive puzzles like protein folding, climate changes, and finding new medicines.

---

## What is the Big Idea?
Most AI today runs in giant, power-hungry data centers. But your new laptop (especially those with Snapdragon, AMD Ryzen AI, or Apple M-series chips) has a dedicated "AI Brain" called an NPU (Neural Processing Unit).

NPUs are like specialist Olympic sprinters—they do AI math incredibly fast while using almost zero battery. SwarmNet connects thousands of these "sprinters" into a giant Swarm to do big science for everyone.

---

## How the Magic Happens (The Simple Version)

1.  **The Hub**: Think of this as the "Mission Control". It lives in the cloud (or your server) and keeps track of everyone in the swarm.
2.  **The Swarm**: This is the networking layer. It uses some cool tricks (UDP Multicast) to find other nodes nearby and talk to them without you having to set up a single thing.
3.  **The Agent**: This is the little app you run. It sits in your system tray, watches your NPU, and does small "missions" (AI tasks) whenever it's free. You earn XP for every task!

---

## Features You Will Love

### NPU-First Speed
We don't just use your CPU. We go straight for the good stuff: Qualcomm Hexagon, AMD VitisAI, and Apple CoreML. It is like switching from a bicycle to a rocket ship.

### Training on the Edge
We use something called Evolutionary Strategies (ES). Instead of the old-school way of training AI, we "evolve" it directly on the NPU. You can watch it happen live on a 3D dashboard.

### 3D Global Dashboard
See the whole world's nodes lighting up in real-time. It looks like something out of a sci-fi movie.

### Gamification (Earn While You Help)
Earn Experience Points (XP), unlock cool hexagonal badges, and compete on the global leaderboard. Who knew saving the world could be a competition?

---

## For the Geeks: Under the Hood

If you want the deep technical deets, here is what is really going on:

### The Inference Pipeline
When an image comes in:
- **Base64 to Tensor**: We grab the image text, turn it into numbers, and squish it into a shape the AI likes (usually 224x224).
- **NPU Routing**: Our code checks if you have a QNN (Qualcomm) or VitisAI (AMD) driver. If yes, the NPU takes over. If not, it falls back to the GPU or CPU so it never fails.
- **Matrix Magic**: The NPU does thousands of parallel math operations using FP16 precision (super fast, low power).
- **Softmax**: We turn the raw "scores" into a percentage you can actually understand (e.g., "99% Cat").

### Tech We Use
- **Backend**: FastAPI (Python) — fast, modern, and easy.
- **Frontend**: Vanilla JS + Three.js — for that sweet 3D globe.
- **Networking**: WebSockets for live updates and UDP Multicast for finding friends.
- **Database**: Supabase — it handles our logins and leaderboards with ease.

---

## Quick Start (Let's Go!)

### 1. Get the Backend Ready
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your secrets here!
python server.py
```
Open your browser to http://localhost:8000 and enjoy the views!

### 2. Join the Swarm as a Node
```bash
cd swarmnet_agent
python tray.py
```
A little bee icon will appear in your tray. You are now a scientist!

---

## Where is Everything?

- `backend/`: The brains. Routes, ML logic, and the training engine.
- `frontend/`: The beauty. All the HTML/CSS/JS for the dashboard.
- `swarmnet_agent/`: The worker. The tray app that powers your node.
- `model/`: The memory. Where the AI's "knowledge" (.onnx files) lives.

---

*SwarmNet: Making science happen, one NPU pulse at a time.*

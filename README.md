<p align="center">
  <img src="https://img.shields.io/badge/version-2.0.0-blue?style=flat-square" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/react-19-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/fastapi-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/NPU-Qualcomm%20Hexagon-EE0000?style=flat-square" alt="NPU" />
</p>

<h1 align="center">🐝 SwarmNet</h1>
<h3 align="center">Decentralized NPU-First Supercomputing Platform</h3>

<p align="center">
  <strong>Harness idle Neural Processing Units across edge devices to train open-source AI models using Evolutionary Strategies — no backpropagation required.</strong>
</p>

---

SwarmNet is a full-stack decentralized computing platform that turns idle NPU hardware on consumer laptops (Qualcomm Snapdragon, AMD XDNA, Apple ANE, Intel Meteor Lake) into a distributed supercomputer. Contributors donate compute cycles to power Evolutionary Strategy (ES) training algorithms, earning XP and climbing a global leaderboard — while scientists solve complex genomic, medical, and protein-folding problems.

The platform features a **FastAPI backend** running ONNX Runtime with QNN Execution Provider for Qualcomm Hexagon NPU inference, a **React + TypeScript frontend** with real-time swarm visualization, and a **system tray agent** that silently donates compute in the background.

---

## 📑 Table of Contents

- [Tech Stack \& Features](#-tech-stack--features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation \& Setup](#-installation--setup)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🛠 Tech Stack & Features

### Backend
- **FastAPI** — High-performance async Python web framework with auto-generated OpenAPI docs
- **ONNX Runtime** — Hardware-accelerated inference with QNN Execution Provider for Qualcomm Hexagon NPU
- **Pydantic v2** — Strict configuration validation via `pydantic-settings`
- **Supabase** — PostgreSQL-backed auth, user management, XP tracking, and leaderboard
- **SlowAPI** — Per-IP rate limiting (configurable requests/minute)
- **Uvicorn** — ASGI server with hot-reload support

### Frontend
- **React 19** + **TypeScript** — Modern SPA with type-safe component architecture
- **Vite 8** — Lightning-fast HMR dev server and optimized production bundler
- **Recharts** — Real-time training accuracy and loss line charts
- **Lucide React** — Premium icon system
- **Plus Jakarta Sans** + **JetBrains Mono** — Custom typography via Google Fonts

### AI / ML Engine
- **Evolutionary Strategies (ES)** — Gradient-free training using only forward passes on NPU
- **MobileNetV2** — Pre-trained ONNX model for image classification inference
- **NPU Simulation** — Transparent fallback for development on non-NPU hardware

### Swarm Infrastructure
- **UDP Multicast Discovery** — Zero-config peer-to-peer node detection
- **Heartbeat Registry** — Live health monitoring with automatic fault detection
- **Queue-Aware Routing** — Intelligent request distribution based on node load
- **WebSocket Streaming** — Real-time inference results and training telemetry

### Platform Features
- 🔬 **NPU vs CPU Benchmark** — Side-by-side latency and power consumption comparison
- 🎮 **XP Gamification** — Earn experience points, badges, and climb the leaderboard
- 📸 **Image Classification Playground** — Upload or webcam-based live inference
- 📈 **Live ES Training Dashboard** — Watch accuracy curves evolve in real-time
- 🛡️ **Admin Console** — Task seeding, node management, and system health monitoring

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React SPA (Vite)                      │
│  Landing · Dashboard · Training · Benchmark · Playground │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / WebSocket
┌──────────────────────────▼──────────────────────────────┐
│                  FastAPI Backend (Uvicorn)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │   Auth   │ │Dashboard │ │ Devices  │ │ Inference  │  │
│  │  Router  │ │  Router  │ │  Router  │ │   Router   │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘  │
│       │             │            │              │         │
│  ┌────▼─────────────▼────────────▼──────────────▼──────┐ │
│  │              Service Layer                          │ │
│  │  InferenceService · TaskService · XPService         │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────────┐ │
│  │         ONNX Runtime (QNN / CPU Provider)           │ │
│  │    MobileNetV2 · ES Training Engine · NPU Eval      │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Swarm Mesh (UDP Multicast)                │ │
│  │   Registry · Discovery · Node · Heartbeat Monitor   │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    Supabase (Cloud)     │
              │  Auth · DB · XP Ledger  │
              └─────────────────────────┘
```

---

## 📂 Project Structure

```
SwarmNet/
├── server.py                    # Application entry point — starts Uvicorn
├── pytest.ini                   # Pytest configuration
├── .env                         # Environment variables (git-ignored)
├── .gitignore                   # Git ignore rules
│
├── backend/                     # FastAPI backend application
│   ├── main.py                  # App factory — mounts routers, CORS, static files
│   ├── config.py                # Pydantic-based centralized configuration
│   ├── logging_config.py        # Structured JSON logging setup
│   ├── supabase_client.py       # Supabase client initialization helpers
│   ├── static_routes.py         # SPA catch-all router for frontend delivery
│   ├── preprocess.py            # Image preprocessing utilities
│   ├── model_loader.py          # ONNX model loading helpers
│   ├── migrate.py               # Database migration script
│   ├── requirements.txt         # Python dependency manifest
│   ├── .env.example             # Environment variable template
│   │
│   ├── routers/                 # API route handlers
│   │   ├── inference.py         # Core inference, benchmark, swarm, & ES endpoints
│   │   ├── auth.py              # Registration, login, JWT session management
│   │   ├── dashboard.py         # Stats, leaderboard, and science progress
│   │   ├── devices.py           # Device registration, task dispatch, XP reward
│   │   └── admin.py             # Admin-only operations (task seeding)
│   │
│   ├── services/                # Business logic layer
│   │   ├── inference_service.py # ONNX inference orchestration
│   │   ├── task_service.py      # Compute task lifecycle management
│   │   └── xp_service.py        # XP calculation and badge assignment
│   │
│   ├── models/                  # Pydantic request/response schemas
│   │   ├── requests.py          # InferenceRequest, FrameRequest, etc.
│   │   └── responses.py         # InferenceResponse schema
│   │
│   ├── validators/              # Input validation
│   │   └── image.py             # Image format, size, and dimension checks
│   │
│   ├── middleware/              # Custom middleware
│   │   └── ws_auth.py           # WebSocket authentication middleware
│   │
│   ├── swarm/                   # Distributed swarm subsystem
│   │   ├── models.py            # NodeInfo, HeartbeatPayload, SwarmMetrics
│   │   ├── registry.py          # Node registry with heartbeat monitoring
│   │   ├── discovery.py         # UDP multicast broadcaster & listener
│   │   └── node.py              # Individual SwarmNode logic
│   │
│   ├── npu_es/                  # Evolutionary Strategies training engine
│   │   ├── es_engine.py         # Core ES algorithm (OpenAI-style)
│   │   ├── evaluator.py         # NPU-accelerated fitness evaluation
│   │   ├── onnx_model.py        # Dynamic ONNX model builder
│   │   ├── dataset.py           # MNIST dataset loader
│   │   └── train.py             # Training orchestrator
│   │
│   └── model/                   # ONNX model artifacts
│       ├── mobilenetv2-12.onnx  # Pre-trained MobileNetV2 (ImageNet)
│       ├── es_trained.onnx      # ES-trained MNIST classifier
│       └── imagenet_labels.json # ImageNet class label mapping
│
├── frontend/                    # React + TypeScript SPA
│   ├── package.json             # Node.js dependencies & scripts
│   ├── vite.config.ts           # Vite bundler configuration
│   ├── tsconfig.json            # TypeScript project references
│   ├── index.html               # HTML entry point (Vite dev server)
│   ├── src/
│   │   ├── main.tsx             # React DOM mount point
│   │   ├── App.tsx              # Monolithic SPA — all views & state
│   │   ├── index.css            # Global stylesheet (Nest Zero design)
│   │   └── assets/              # Static images (hero, icons)
│   └── dist/                    # Production build output (git-ignored)
│
├── swarmnet_agent/              # Desktop system tray client agent
│   ├── tray.py                  # System tray icon & menu (pystray)
│   ├── monitor.py               # Hardware telemetry collector
│   ├── task_runner.py           # Background compute task executor
│   ├── api_client.py            # HTTP client for backend communication
│   └── config.py                # Agent-local configuration
│
└── tests/                       # Unit test suite
    ├── test_config.py           # Configuration validation tests
    ├── test_image_validator.py  # Image validator edge cases
    ├── test_preprocess.py       # Preprocessing pipeline tests
    └── test_swarm_selection.py  # Swarm node selection algorithm tests
```

---

## 📋 Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Backend runtime |
| **Node.js** | 18+ | Frontend build toolchain |
| **npm** | 9+ | Package manager for frontend |
| **Git** | 2.x | Version control |

### Optional (for NPU acceleration)
| Hardware | SDK |
|----------|-----|
| Qualcomm Snapdragon X Elite / Plus | QNN SDK with `QnnHtp.dll` |
| AMD Ryzen AI (XDNA) | Vitis AI Runtime |
| Apple M1/M2/M3/M4 | CoreML (via ONNX Runtime) |
| Intel Meteor Lake | OpenVINO Runtime |

> **Note:** SwarmNet runs without NPU hardware — it transparently simulates NPU execution using the CPU provider for development and demo purposes.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NipunAdarsh/SwarmNet..git
cd SwarmNet
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Copy and configure environment variables
cp backend/.env.example .env
# Edit .env with your credentials (see Environment Variables section)
```

### 3. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Build the production SPA bundle
npm run build

cd ..
```

### 4. Start the Server

```bash
python server.py
```

The application will be available at **http://localhost:8000**.

---

## 🔐 Environment Variables

Create a `.env` file in the project root. Reference `backend/.env.example` for the full template.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ADMIN_SECRET` | ✅ Yes | — | Admin authentication secret (min 16 chars for production) |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8000` | Server bind port |
| `DEBUG` | No | `false` | Enable hot-reload and debug logging |
| `CORS_ORIGINS` | No | `http://localhost:8000` | Comma-separated allowed origins |
| `SUPABASE_URL` | No* | — | Supabase project URL |
| `SUPABASE_ANON_KEY` | No* | — | Supabase anonymous key |
| `SUPABASE_SERVICE_ROLE_KEY` | No* | — | Supabase service role key |
| `MAX_IMAGE_SIZE_MB` | No | `10` | Max upload image size (MB) |
| `ALLOWED_IMAGE_FORMATS` | No | `jpg,jpeg,png,webp,gif` | Accepted image formats |
| `ES_POPULATION_SIZE` | No | `50` | ES training population size |
| `ES_SIGMA` | No | `0.02` | ES mutation noise sigma |
| `ES_LEARNING_RATE` | No | `0.03` | ES learning rate |
| `USE_NPU_SIMULATION` | No | `true` | Simulate NPU on non-NPU hardware |
| `MULTICAST_GROUP` | No | `224.1.1.1` | Swarm UDP multicast group |
| `MULTICAST_PORT` | No | `5007` | Swarm discovery port |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | No | `120` | API rate limit per IP |
| `WS_AUTH_REQUIRED` | No | `false` | Require WebSocket authentication |

> \* Supabase variables are required only when using auth, leaderboard, and device registration features. The core inference and training engine works without them.

---

## ▶️ Usage

### Development Mode

```bash
# Backend with hot-reload (set DEBUG=true in .env)
python server.py

# Frontend dev server with HMR (separate terminal)
cd frontend
npm run dev
```

### Production Mode

```bash
# Build frontend assets
cd frontend && npm run build && cd ..

# Start production server
python server.py
```

### API Health Check

```bash
curl http://localhost:8000/health
# → {"status": "ok", "version": "2.0.0"}
```

### Run an Image Classification

```bash
curl -X POST http://localhost:8000/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<BASE64_IMAGE_DATA>"}'
```

### Run NPU vs CPU Benchmark

```bash
curl -X POST http://localhost:8000/api/v1/benchmark
```

### Query Swarm Status

```bash
# List all nodes
curl http://localhost:8000/api/v1/swarm/nodes

# Aggregate metrics
curl http://localhost:8000/api/v1/swarm/metrics
```

---

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Application health check |
| `GET` | `/api/v1/stats` | Real-time aggregate statistics |
| `GET` | `/api/v1/models` | List loaded ONNX models |
| `GET` | `/api/v1/energy` | NPU energy efficiency metrics |
| `GET` | `/api/v1/cloud-comparison` | NPU vs cloud latency comparison |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/infer` | NPU-accelerated image classification |
| `POST` | `/api/v1/infer-cpu` | CPU-only inference (for benchmarking) |
| `POST` | `/api/v1/infer-frame` | Webcam frame classification |
| `POST` | `/api/v1/webcam-stop` | Stop webcam streaming session |

### Benchmark

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/benchmark` | NPU vs CPU benchmark race |
| `POST` | `/api/v1/benchmark-controlled` | Controlled benchmark with custom iterations |

### Swarm Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/swarm/nodes` | List all registered swarm nodes |
| `GET` | `/api/v1/swarm/metrics` | Aggregate swarm-wide metrics |
| `POST` | `/api/v1/swarm/infer` | Route inference through swarm mesh |
| `POST` | `/api/v1/swarm/node/{id}/disable` | Disable a swarm node |
| `POST` | `/api/v1/swarm/node/{id}/enable` | Re-enable a disabled node |

### Auth & Users

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Create a new user account |
| `POST` | `/login` | Authenticate and get JWT session |
| `GET` | `/me` | Get current user profile |

### Dashboard & Leaderboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/leaderboard` | Global XP leaderboard |
| `GET` | `/api/science/progress` | Scientific research progress |

### Device Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/device/register` | Register a compute donor device |
| `GET` | `/api/tasks/next` | Fetch next available compute task |
| `POST` | `/api/tasks/complete` | Submit completed task results |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/admin/tasks/seed` | Seed new compute tasks (admin-only) |

> **Interactive Docs**: Visit `http://localhost:8000/docs` for the auto-generated Swagger UI, or `http://localhost:8000/redoc` for ReDoc.

---

## 🧪 Testing

```bash
# Run the full test suite
python -m pytest

# Run with verbose output
python -m pytest -v

# Run a specific test file
python -m pytest tests/test_swarm_selection.py
```

**Current test coverage:**
- ✅ Configuration validation & secret enforcement
- ✅ Image validator (format, size, dimension checks)
- ✅ Image preprocessing pipeline
- ✅ Swarm node selection algorithm (queue-aware routing)

---

## 🤝 Contributing

Contributions are welcome! Follow these steps to get started:

### 1. Fork & Clone

```bash
git fork https://github.com/NipunAdarsh/SwarmNet..git
git clone https://github.com/<your-username>/SwarmNet..git
cd SwarmNet
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes & Test

```bash
# Ensure all tests pass
python -m pytest

# Lint the frontend
cd frontend && npm run lint
```

### 4. Commit with Descriptive Messages

```bash
git add .
git commit -m "feat: add webcam frame rate throttling for battery savings"
```

### 5. Push & Open a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request against `main` on GitHub.

### Commit Message Convention

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `refactor:` | Code restructuring |
| `test:` | Adding or updating tests |
| `chore:` | Build, CI, or tooling changes |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/NipunAdarsh">Nipun Adarsh</a> — Powered by NPUs, driven by the swarm.</sub>
</p>

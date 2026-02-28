# SwarmNet

> **Democratizing AI training for open science — one idle NPU at a time.**

SwarmNet is a decentralized platform that harnesses millions of idle consumer NPUs (Neural Processing Units) to create the world's largest distributed supercomputer for fine-tuning open-source scientific AI models — protein folding, climate modeling, and disease prediction.

---

## Features

-  **Three.js Interactive Globe** — Real-time 3D visualization of the swarm network with animated nodes and connections
-  **Scroll Animations** — Smooth entrance effects powered by IntersectionObserver
-  **Live Stats Counters** — Animated count-up for network metrics (TOPS, active nodes, models, parameters)
-  **Leaderboard & Badges** — Gamified community section with global rankings and milestone badges
-  **Privacy-First Architecture** — Federated Learning with LoRA fine-tuning; data never leaves your device
-  **Fully Responsive** — Optimized for desktop, tablet, and mobile

---

## Tech Stack

| Layer | Technology |
|---|---|
| Markup | Semantic HTML5 |
| Styling | Pure CSS3 (Grid, Flexbox, CSS Variables, Keyframes) |
| Interactivity | Vanilla JavaScript (ES6+) |
| 3D Visualization | Three.js |
| Typography | Google Fonts (Syne, Space Mono, Inter) |

---

## Project Structure

```
SwarmNet/
├── index.html              # Single-page landing
├── css/
│   └── styles.css          # Full design system + all section styles
├── js/
│   ├── main.js             # Nav, scroll animations, counters, parallax
│   └── globe.js            # Three.js globe visualization
├── assets/
│   └── images/
│       └── hero_background.webp
├── prd.md                  # Product Requirements Document
├── design.md               # Visual Design Guidelines
├── tech_stack.md            # Technology Architecture
├── walkthrough.md          # Build walkthrough & verification
└── README.md               # ← You are here
```

---

## How to Run

### Prerequisites

- [Node.js](https://nodejs.org/) (v16 or higher) — only needed for the local dev server
- A modern web browser (Chrome, Edge, Firefox, Safari)

### Option 1: Using a Local Server (Recommended)

```bash
# Clone or navigate to the project folder
cd SwarmNet

# Start a local server using npx (no install needed)
npx -y serve . -l 3000
```

Then open **http://localhost:3000** in your browser.

### Option 2: Direct File Open

Simply double-click `index.html` to open it in your default browser.

> **Note:** The Three.js globe loads from a CDN, so you'll need an internet connection for the 3D visualization. All other features work offline.

### Option 3: VS Code Live Server

1. Open the project folder in VS Code
2. Install the **Live Server** extension
3. Right-click `index.html` → **Open with Live Server**

---

## Design System

| Token | Value | Usage |
|---|---|---|
| `--bg-deep` | `#040404` | Page background |
| `--text` | `#F9F7F7` | Primary text |
| `--accent` | `#E51B0C` | CTAs, highlights, active states |
| `--surface` | `#565B62` | Borders, secondary text |
| `--teal` | `#7D9D9C` | Secondary accent, tags |
| `--rust` | `#C03620` | Gradient accent |
| `--maroon` | `#501F25` | Deep accent |

---

## Specification Files

| File | Description |
|---|---|
| [`prd.md`](prd.md) | Product vision, user personas, features, user flow, KPIs |
| [`design.md`](design.md) | Visual identity, color palette, layout components, animation guidelines |
| [`tech_stack.md`](tech_stack.md) | Client (Tauri/Rust), ML layer (Flower/LoRA), networking (libp2p/gRPC), backend (Go/FastAPI), data (PostgreSQL/Redis/ClickHouse) |

---

## License

Open Source — Built for open science. 🧬🌍

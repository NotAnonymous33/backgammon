# Backgammon

**Summary**  
This repository implements a full-stack Backgammon game with both human‑vs‑human (via WebSockets) and AI opponents (random, first‑move, MCTS, and neural network strategies). The **backend** is a Python Flask application using Flask‑SocketIO for real‑time communication and SQLAlchemy for game state persistence. The **frontend** is a React + TypeScript app bootstrapped with Vite, providing a responsive UI for rolling dice, making moves, and watching AI play.

## 📖 Overview  
A turn‑based Backgammon game server and client:  
- **Multiplayer**: create or join “rooms” in your browser; real‑time sync via WebSockets.  
- **AI Opponents**: choose between `random`, `first`, Monte Carlo Tree Search (`mcts`), or Neural Network (`neural`) agents.  
- **Persistence**: game states stored in a local SQLite database, enabling reconnects and replay logs.  
- **Frontend**: lightweight React + Vite SPA with ESLint and TypeScript support.

## ✨ Features  
- **Room Management**: auto‑generated 5‑letter codes to host new games.  
- **Real‑time Updates**: moves and dice rolls pushed instantly to all clients via Socket.IO.  
- **AI Strategies**:  
  - **RandomAgent**: picks any legal move at random  
  - **FirstAgent**: always takes the first legal option  
  - **MCTSAgent**: uses Monte Carlo Tree Search with a configurable time budget  
  - **NNAgent**: uses a trained value neural with 2 layers of 128 nodes to choose the best move
- **Game Persistence**: uses Flask‑SQLAlchemy (SQLite) to record every board state for history and recovery.

## 🛠 Tech Stack  
### Backend  
- **Python 3.11**
- **Cython 3**
- **pybind11 2**
- **Flask 3.1.0** 
- **Socket.IO** for WebSocket communication  
- **SQLite** via SQLAlchemy ORM 
- **PyTorch 2.6**

### Frontend  
- **Node.js** & **npm**  
- **React 18** + **TypeScript** + **Vite**  
- **ESLint** with Type‑aware rules  

## 📋 Prerequisites  
- **Python 3.11** (the pinned dependencies do not support Python 3.14)
- **C and C++ compilers** for the native extensions (GCC/G++ and Python development headers on Linux, Xcode Command Line Tools on macOS, or Visual Studio Build Tools with C++ on Windows)
- **Node.js** ≥ 16  
- **npm** (bundled with Node.js)  

## ⚙️ Installation  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/NotAnonymous33/backgammon.git
   cd backgammon
   ```

2. **Backend setup**

    Linux/macOS (start in the repository root):
    ```bash
    cd backend
    python3.11 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    cd classes
    python setup_cython.py build_ext --inplace
    python setup.py build_ext --inplace
    cd ..
    python app.py
    ```

    Windows PowerShell (start in the repository root):
    ```powershell
    cd backend
    py -3.11 -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    cd classes
    python setup_cython.py build_ext --inplace
    python setup.py build_ext --inplace
    cd ..
    python app.py
    ```

    Use only the activation command for your operating system. If `venv` was
    created with a different Python version, deactivate it and rename it before
    creating a new one. If Python 3.11 is installed through `uv`, you can replace
    the environment creation command with `uv venv --python 3.11 --seed venv`.

    The CPU build of PyTorch is sufficient to run the backend. For NVIDIA GPU
    support, replace `/whl/cpu` with `/whl/cu126` in the PyTorch install command
    ([PyTorch 2.6 installation options](https://pytorch.org/get-started/previous-versions/#v260)).
    Always build both extensions with the same virtual environment used to run
    the server. Run `app.py` from `backend`, not `backend/classes`; it creates
    the database tables automatically and serves http://localhost:5000.

3. **Frontend setup**

    In a separate terminal, start in the repository root:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

- Runs the SPA on http://localhost:5173 (Vite’s default).

## 🚀 Usage  

1. Open your browser at `http://localhost:5173`.  
2. **Create** a new game or **Join** using a 5‑letter code.  
3. Select your **side** (White/Black) and choose **Human** or **AI** (random, first, mcts, neural).  
4. **Roll dice** and **Make moves** by clicking checkers—valid moves are highlighted.  
5. Watch the game auto‑update across all connected clients.

## 📊 Benchmarks

Measure rules correctness, engine speed, MCTS iterations per second, neural-network training throughput and learning, and AI skill, and compare them across changes. Skill is measured three ways: head-to-head matches, the winning chance each agent gives up per decision on labelled reference positions (exact in the bearoff), and Elo ratings for a pool of agents:

```bash
cd backend
python -m bench run --label baseline   # record a run
python -m bench run                    # after a change: record and compare with the previous run
```

Results are also browsable at http://localhost:5173/bench. See [backend/bench/README.md](backend/bench/README.md).  

## 📂 Project Structure 
```pgsql 
backgammon/ 
│ 
├─ backend/ ← Flask backend
│ ├─ app.py ← main server and Socket.IO handlers
│ ├─ models.py ← SQLAlchemy Game model
│ ├─ classes/ ← Board logic & agent implementations (the most interesting code lives here)
│ ├─ requirements.txt ← Python dependencies
│ └─ test.db ← SQLite file (auto‑generated)
│ 
├─ frontend/ ← React + Vite frontend
│ ├─ src/ ← React components & assets
│ ├─ index.html ← SPA entry
│ ├─ package.json ← npm dependencies & scripts
│ └─ tsconfig.json ← TypeScript config
│ 
└─ ProjectReport/ ← LaTeX & report
```

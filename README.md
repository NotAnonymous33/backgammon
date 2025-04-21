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
- **Python 3.x**  
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
- **Python** ≥ 3.8  
- **Node.js** ≥ 16  
- **npm** (bundled with Node.js)  

## ⚙️ Installation  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/NotAnonymous33/backgammon.git
   cd backgammon

2. **Backend setup**
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate      # Linux/Mac
    .\venv\Scripts\activate       # Windows
    pip install -r requirements.txt
    # Initialize database & start server
    cd classes
    python setup_cython.py build_ext --inplace
    python setup.py build_ext --inplace
    python app.py
    ```
- Runs on http://localhost:5000 by default.

3. **Frontend setup**
    ```bash
    cd ../frontend
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
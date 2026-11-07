# 🤖 Robot Teleoperation Interface (FastAPI + WebSockets)

A real-time robot teleoperation interface built with **FastAPI**, **WebSockets**, and **OpenCV**.

## Project Structure
```text
robot-teleop/
├── main.py              # FastAPI app & WebSocket endpoint (/ws/stream)
├── requirements.txt     # Python dependencies
└── static/
    └── index.html       # Web-based Teleoperation console & camera stream viewer
```

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server with Uvicorn
From the `robot-teleop` directory, run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the Interface
Open your browser and navigate to:
```text
http://localhost:8000
```
- Real-time video frames will stream over WebSockets.
- Use **W / A / S / D** or the on-screen D-Pad to send directional velocity commands.
- Press **Spacebar** or the red **STOP** button to issue an immediate stop command.

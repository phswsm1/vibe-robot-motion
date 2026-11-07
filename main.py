import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tracker import MediaPipeVisionTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robot-teleop")

app = FastAPI(
    title="Robot Teleoperation Interface with MediaPipe AI",
    description="Web-based Robot Teleoperation backend with FastAPI, WebSockets & MediaPipe AI",
    version="2.0.0"
)

# 1. Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def get_index():
    """Serves the teleoperation frontend HTML."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Robot Teleoperation Backend Running."}


class CameraStreamer:
    """Manages OpenCV webcam video capture with fallback frame generation."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_camera()

    def _init_camera(self):
        try:
            logger.info(f"Attempting to initialize camera index {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                logger.warning(f"Could not open camera index {self.camera_index}. Fallback frames will be generated.")
                self.cap = None
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                logger.info(f"Camera index {self.camera_index} initialized successfully.")
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            self.cap = None

    def read_frame(self) -> np.ndarray:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        return self._generate_fallback_frame()

    def _generate_fallback_frame(self) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(0, 480, 60):
            cv2.line(frame, (0, y), (640, y), (30, 30, 30), 1)
        for x in range(0, 640, 80):
            cv2.line(frame, (x, 0), (x, 480), (30, 30, 30), 1)

        cv2.putText(frame, "[ ROBOT TELEOP CAMERA STANDBY ]", (100, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Camera offline / Busy - Check index or permissions", (80, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            logger.info("Camera released.")


# 2. WebSocket Endpoint with MediaPipe Vision Tracking
@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Teleop client connected from {client_ip}")

    streamer = CameraStreamer(camera_index=0)
    tracker = MediaPipeVisionTracker(pinch_threshold=0.065)

    telemetry_counter = 0

    async def send_frames_task():
        nonlocal telemetry_counter
        loop = asyncio.get_running_loop()
        try:
            while True:
                # 1. Grab raw frame in executor
                raw_frame = await loop.run_in_executor(None, streamer.read_frame)

                # 2. Process frame with MediaPipe (Hands + FaceMesh + Pinch) in executor
                processed_frame, telemetry = await loop.run_in_executor(
                    None, tracker.process_frame, raw_frame
                )

                # 3. JPEG encode processed frame
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                success, encoded_image = cv2.imencode(".jpg", processed_frame, encode_param)
                if success:
                    await websocket.send_bytes(encoded_image.tobytes())

                # 4. Stream telemetry JSON packet every 3 frames (~10 Hz) to conserve bandwidth
                telemetry_counter += 1
                if telemetry_counter % 3 == 0:
                    payload = {
                        "type": "telemetry",
                        "data": telemetry
                    }
                    await websocket.send_text(json.dumps(payload))

                # Maintain ~30 FPS
                await asyncio.sleep(0.033)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Frame sender finished: {e}")

    async def receive_commands_task():
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    data = json.loads(message)
                    logger.info(f"Received Teleop Command: {data}")
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
        except asyncio.CancelledError:
            pass
        except WebSocketDisconnect:
            logger.info(f"Client disconnected ({client_ip})")
        except Exception as e:
            logger.warning(f"Command receiver exception: {e}")

    sender = asyncio.create_task(send_frames_task())
    receiver = asyncio.create_task(receive_commands_task())

    try:
        done, pending = await asyncio.wait(
            [sender, receiver],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    finally:
        tracker.release()
        streamer.release()
        logger.info(f"Cleaned up session & resources for {client_ip}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger("robot-teleop.tracker")


class MediaPipeVisionTracker:
    """
    Combined Real-Time Computer Vision Pipeline:
    1. Google MediaPipe: Hands (21 3D landmarks + Pinch Gesture) & Face Mesh.
    2. Ultralytics YOLO (YOLOv8n): Object detection & Bounding Box extraction.
    3. Spatial Collision & Gesture Interaction: Hand-to-Object selection via Pinch.
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        max_num_hands: int = 2,
        max_num_faces: int = 1,
        pinch_threshold: float = 0.065,
        yolo_conf_threshold: float = 0.5,
        yolo_inference_interval: int = 2  # Run YOLO every 2 frames for CPU optimization
    ):
        self.pinch_threshold = pinch_threshold
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_inference_interval = yolo_inference_interval
        self.frame_count = 0

        # Cached detections from previous YOLO inference
        self.cached_detections: List[Dict[str, Any]] = []

        # 1. Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,  # Fast inference
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 2. Initialize Ultralytics YOLO
        self.yolo = None
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {yolo_model_name}...")
            self.yolo = YOLO(yolo_model_name)
            logger.info(f"YOLO model {yolo_model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def _run_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Runs YOLO object detection and extracts filtered bounding boxes."""
        detections = []
        if self.yolo is None:
            return detections

        try:
            # Run inference with lightweight 480px resize for fast CPU processing
            results = self.yolo(frame, conf=self.yolo_conf_threshold, imgsz=480, verbose=False)
            boxes = results[0].boxes
            names = results[0].names

            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.yolo_conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2],
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")

        return detections

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a video frame through both YOLO and MediaPipe, checks spatial interaction,
        draws combined overlays, and outputs structured telemetry.
        """
        h, w, _ = frame.shape
        self.frame_count += 1

        telemetry: Dict[str, Any] = {
            "hand_detected": False,
            "pinch": False,
            "pinch_distance_norm": 0.0,
            "pinch_distance_px": 0.0,
            "wrist": None,
            "thumb_tip": None,
            "index_tip": None,
            "face_detected": False,
            "detected_objects": [],
            "hovered_object": None,
            "selected_object": None
        }

        # -------------------------------------------------------------
        # 1. YOLO Object Detection (Throttled for CPU Optimization)
        # -------------------------------------------------------------
        if self.frame_count % self.yolo_inference_interval == 0 or not self.cached_detections:
            self.cached_detections = self._run_yolo(frame)

        telemetry["detected_objects"] = self.cached_detections

        # -------------------------------------------------------------
        # 2. MediaPipe Hand & Face Mesh Inferences
        # -------------------------------------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        hands_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)

        rgb_frame.flags.writeable = True

        # Draw Face Mesh contours
        if face_results.multi_face_landmarks:
            telemetry["face_detected"] = True
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 200, 0), thickness=1, circle_radius=1
                    )
                )

        # Hand Tracking & Keypoints
        index_px = None
        thumb_px = None
        is_pinch = False

        if hands_results.multi_hand_landmarks:
            telemetry["hand_detected"] = True
            for hand_landmarks, handedness in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness
            ):
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                wrist_lm = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                thumb_lm = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_lm = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_px = (int(thumb_lm.x * w), int(thumb_lm.y * h))
                index_px = (int(index_lm.x * w), int(index_lm.y * h))
                wrist_px = (int(wrist_lm.x * w), int(wrist_lm.y * h))

                # Euclidean Distances
                norm_dist = math.sqrt(
                    (thumb_lm.x - index_lm.x) ** 2 +
                    (thumb_lm.y - index_lm.y) ** 2 +
                    (thumb_lm.z - index_lm.z) ** 2
                )
                px_dist = math.hypot(thumb_px[0] - index_px[0], thumb_px[1] - index_px[1])
                is_pinch = norm_dist < self.pinch_threshold

                telemetry["pinch"] = is_pinch
                telemetry["pinch_distance_norm"] = round(norm_dist, 4)
                telemetry["pinch_distance_px"] = round(px_dist, 1)
                telemetry["wrist"] = {"x": round(wrist_lm.x, 4), "y": round(wrist_lm.y, 4), "px": wrist_px}
                telemetry["thumb_tip"] = {"x": round(thumb_lm.x, 4), "y": round(thumb_lm.y, 4), "px": thumb_px}
                telemetry["index_tip"] = {"x": round(index_lm.x, 4), "y": round(index_lm.y, 4), "px": index_px}

                # Simulated Robot Arm Joint Angles & Gripper Width
                # Joint 1 (Base Yaw): Wrist X mapped to -90° to +90°
                j1 = int((wrist_lm.x - 0.5) * -180)  # Inverted for intuitive mirror control
                # Joint 2 (Shoulder/Elbow Pitch): Wrist Y mapped to 10° to 130°
                j2 = int((1.0 - wrist_lm.y) * 120 + 10)
                # Gripper Opening Width: Pinch distance mapped from 0mm (closed) to 85mm (fully open)
                gripper_mm = int(np.clip((norm_dist - 0.02) / (0.12 - 0.02) * 85, 0, 85))
                if is_pinch:
                    gripper_mm = 0

                telemetry["joint_1_deg"] = j1
                telemetry["joint_2_deg"] = j2
                telemetry["gripper_width_mm"] = gripper_mm

                # Visual HUD Indicator for Pinch
                mid_point = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
                if is_pinch:
                    cv2.line(frame, thumb_px, index_px, (0, 255, 0), 3)
                    cv2.circle(frame, mid_point, 10, (0, 255, 0), cv2.FILLED)
                else:
                    cv2.line(frame, thumb_px, index_px, (0, 220, 255), 1)

        # -------------------------------------------------------------
        # 3. Spatial Collision & Pinch Interaction (Index Finger inside YOLO Box)
        # -------------------------------------------------------------
        selected_obj_info = None

        for det in self.cached_detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class"]
            conf = det["confidence"]
            cx, cy = det["center"]

            # Collision check: Does index finger fall inside bounding box?
            is_hovered = False
            if index_px is not None:
                if x1 <= index_px[0] <= x2 and y1 <= index_px[1] <= y2:
                    is_hovered = True
                    telemetry["hovered_object"] = cls_name
                    if is_pinch:
                        telemetry["selected_object"] = cls_name
                        selected_obj_info = det
                        logger.info(f"🎯 Object Selected: [{cls_name}] (Confidence: {conf})")

            # -------------------------------------------------------------
            # 4. Draw Combined YOLO Overlays
            # -------------------------------------------------------------
            if is_hovered and is_pinch:
                # SELECTED (Pinch Active): Vibrant Magenta Glowing Box
                box_color = (255, 0, 255)
                thickness = 3
                # Draw dynamic laser line from finger to object center
                cv2.line(frame, index_px, (cx, cy), (255, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 8, (255, 0, 255), -1)
                label_text = f"GRABBED: {cls_name.upper()} ({int(conf * 100)}%)"
            elif is_hovered:
                # HOVERED (Finger inside box, no pinch): Amber Box
                box_color = (0, 215, 255)
                thickness = 2
                label_text = f"HOVER: {cls_name} ({int(conf * 100)}%)"
            else:
                # Normal detected object: Cyan Box
                box_color = (255, 180, 0)
                thickness = 1
                label_text = f"{cls_name} {int(conf * 100)}%"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # Draw label banner
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(frame, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Top HUD Status
        if telemetry["selected_object"]:
            hud_text = f"TARGET: [{telemetry['selected_object'].upper()}] LOCKED & PINCHED!"
            cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            pinch_str = "PINCHED" if telemetry["pinch"] else "OPEN"
            yolo_count = len(self.cached_detections)
            hud_text = f"Pinch: {pinch_str} | YOLO Objs: {yolo_count}"
            cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return frame, telemetry

    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()

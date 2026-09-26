"""Find the face and its landmarks with MediaPipe FaceLandmarker.

In a full-body photo the face is small (often 30–60 px wide), which is too small for the
face model to find reliably. So we crop a square around the head, using the pose
landmarks we already have, and scale the crop up before running the face model.

FaceLandmarker returns 478 points (the "face mesh"). We only need a few of them to pick
skin sample spots (see skin_tone.py). Index reference:
https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from app.config import MODELS_DIR
from app.errors import AppError, ErrorCode
from app.services.pose import L, Pose

logger = logging.getLogger(__name__)

# The head crop is scaled up to at least this size before face detection
CROP_MIN_SIDE = 384


@dataclass
class Face:
    points: np.ndarray  # (478, 2) face-mesh landmarks in pixels of the FULL image
    width_px: float  # face width in the full image (cheek edge to cheek edge)


class FaceFinder:
    """Wraps a MediaPipe FaceLandmarker (one instance, reused, calls serialised with a lock)."""

    def __init__(self) -> None:
        model_path = MODELS_DIR / "face_landmarker.task"
        if not model_path.exists():
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                f"Face model missing ({model_path.name}). Run: python scripts/download_models.py",
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        logger.info("Loaded face model %s", model_path.name)

    def find(self, image_rgb: np.ndarray, pose: Pose) -> Face | None:
        """Face landmarks for the person in `pose`, or None if no face is found."""
        box = head_box(pose)
        x1, y1, x2, y2 = box
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Scale the crop up so the face is big enough for the model
        scale = max(1.0, CROP_MIN_SIDE / max(crop.shape[:2]))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        with self._lock:
            result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        crop_h, crop_w = crop.shape[:2]
        # Normalised crop coordinates → crop pixels → full-image pixels
        points = np.array([[lm.x * crop_w, lm.y * crop_h] for lm in result.face_landmarks[0]]) / scale
        points += np.array([x1, y1])
        width_px = float(np.linalg.norm(points[FACE_LEFT_EDGE] - points[FACE_RIGHT_EDGE]))
        return Face(points=points, width_px=width_px)


# Face-mesh indices of the left/right face edge at cheekbone height
FACE_RIGHT_EDGE = 234
FACE_LEFT_EDGE = 454


def head_box(pose: Pose) -> tuple[int, int, int, int]:
    """A square (x1, y1, x2, y2) around the head, sized from the shoulder width.

    A head is roughly 0.4× shoulder width wide; the box is generously larger (1.0×
    shoulder width) so the whole head fits even if the person tilts it.
    """
    shoulder_px = float(np.linalg.norm(pose[L.LEFT_SHOULDER].xy - pose[L.RIGHT_SHOULDER].xy))
    eyes = (pose[L.LEFT_EYE].xy + pose[L.RIGHT_EYE].xy) / 2
    half = max(24.0, 0.5 * shoulder_px)
    cx, cy = eyes[0], eyes[1] + 0.1 * half  # the eyes are a little above the middle of the head
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(pose.width, cx + half)), int(min(pose.height, cy + half))
    return x1, y1, x2, y2


@lru_cache
def get_face_finder() -> FaceFinder:
    """The shared face finder (FastAPI dependency, so tests can swap in a fake one)."""
    return FaceFinder()

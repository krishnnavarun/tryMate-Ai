"""MediaPipe PoseLandmarker → body landmarks (in pixels) + a person segmentation mask.

MediaPipe finds 33 landmarks per person (nose, eyes, shoulders, elbows, wrists, hips,
knees, ankles, heels, toes...). Each has:
  x, y        position, normalised 0..1 (we convert to pixels)
  z           rough depth (smaller = closer to the camera), same scale as x
  visibility  0..1, how sure it is that the point is visible (not hidden/out of frame)
With output_segmentation_masks=True it also returns a mask: for every pixel, a 0..1
value for "this pixel belongs to the person". We use the mask to find the top of the
head, the soles of the feet, and the width of the chest and waist.

Landmark indices: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

import logging
import threading
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from app.config import MODELS_DIR, get_settings
from app.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)


class L(IntEnum):
    """The landmark indices we use. "LEFT" means the person's left (image right in a selfie-style front photo)."""

    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Without these, we can't measure: the head, shoulders, hips and ankles must all be in the photo.
REQUIRED_LANDMARKS = (
    L.NOSE,
    L.LEFT_SHOULDER,
    L.RIGHT_SHOULDER,
    L.LEFT_HIP,
    L.RIGHT_HIP,
    L.LEFT_ANKLE,
    L.RIGHT_ANKLE,
)
MIN_VISIBILITY = 0.5


@dataclass(frozen=True)
class Landmark:
    x: float  # pixels from the left edge
    y: float  # pixels from the top edge
    z: float  # rough depth, in pixels (same scale as x)
    visibility: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Pose:
    landmarks: list[Landmark]  # 33 items, index with L.*
    mask: np.ndarray | None  # float32 (height × width), 0..1 = "person" probability
    width: int
    height: int

    def __getitem__(self, index: int) -> Landmark:
        return self.landmarks[index]


class PoseDetector:
    """Wraps a MediaPipe PoseLandmarker.

    Loading the model takes ~1 s, so one detector is created and reused for every request.
    MediaPipe doesn't promise that one landmarker can be used by several threads at once,
    so calls are serialised with a lock (FastAPI runs our CPU work in a thread pool).
    """

    def __init__(self, model_name: str) -> None:
        model_path = MODELS_DIR / f"pose_landmarker_{model_name}.task"
        if not model_path.exists():
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                f"Pose model missing ({model_path.name}). Run: python scripts/download_models.py",
            )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            # Look for up to 2 people so we can tell the user when there's more than one
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            output_segmentation_masks=True,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        logger.info("Loaded pose model %s", model_path.name)

    def detect(self, image_rgb: np.ndarray) -> list[Pose]:
        """Return every person found (0, 1 or 2), with landmarks in pixels."""
        height, width = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        with self._lock:
            result = self._landmarker.detect(mp_image)

        poses = []
        for i, normalised in enumerate(result.pose_landmarks):
            landmarks = [
                Landmark(x=lm.x * width, y=lm.y * height, z=lm.z * width, visibility=lm.visibility or 0.0)
                for lm in normalised
            ]
            mask = None
            if result.segmentation_masks and i < len(result.segmentation_masks):
                # numpy_view() is read-only and owned by MediaPipe: copy it
                mask = np.array(result.segmentation_masks[i].numpy_view(), dtype=np.float32).squeeze()
            poses.append(Pose(landmarks=landmarks, mask=mask, width=width, height=height))
        return poses


@lru_cache
def get_pose_detector() -> PoseDetector:
    """The shared detector (FastAPI dependency, so tests can swap in a fake one)."""
    return PoseDetector(get_settings().pose_model)


# ---------------------------------------------------------------------------
# Checks that turn a raw detection into NO_PERSON / MULTIPLE_PEOPLE / PARTIAL_BODY
# ---------------------------------------------------------------------------


def _torso_box(pose: Pose) -> tuple[float, float, float, float]:
    """Bounding box (x1, y1, x2, y2) around shoulders and hips."""
    points = np.array([pose[i].xy for i in (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)])
    (x1, y1), (x2, y2) = points.min(axis=0), points.max(axis=0)
    return x1, y1, x2, y2


def _overlap_ratio(a: tuple, b: tuple) -> float:
    """How much of the smaller box is covered by the other (0 = apart, 1 = one inside the other)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: max(1.0, (box[2] - box[0]) * (box[3] - box[1]))  # noqa: E731
    return (ix * iy) / min(area(a), area(b))


def _is_real_person(pose: Pose) -> bool:
    """A second detection only counts if its torso is clearly visible (not a faint false hit)."""
    torso = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)
    return float(np.mean([pose[i].visibility for i in torso])) >= MIN_VISIBILITY


def _is_inside_frame(pose: Pose, lm: Landmark) -> bool:
    # MediaPipe guesses positions for body parts outside the photo; those land outside 0..width/height
    margin = 0.01
    return -margin * pose.width <= lm.x <= (1 + margin) * pose.width and -margin * pose.height <= lm.y <= (
        1 + margin
    ) * pose.height


def select_single_full_body(poses: list[Pose]) -> Pose:
    """Validate the detections and return the one person to measure.

    Raises AppError with NO_PERSON_DETECTED, MULTIPLE_PEOPLE or PARTIAL_BODY.
    """
    if not poses:
        raise AppError(ErrorCode.NO_PERSON_DETECTED)

    person = poses[0]
    for other in poses[1:]:
        # Ignore a second detection that is really the same person found twice
        if _is_real_person(other) and _overlap_ratio(_torso_box(person), _torso_box(other)) < 0.5:
            raise AppError(ErrorCode.MULTIPLE_PEOPLE)

    missing = [
        idx.name.lower().replace("_", " ")
        for idx in REQUIRED_LANDMARKS
        if person[idx].visibility < MIN_VISIBILITY or not _is_inside_frame(person, person[idx])
    ]
    if missing:
        raise AppError(
            ErrorCode.PARTIAL_BODY,
            f"The full body (head to feet) must be visible in the photo. Couldn't see: {', '.join(missing)}.",
        )
    return person

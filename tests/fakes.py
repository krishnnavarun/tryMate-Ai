"""A synthetic "paper doll" person for testing the measurement maths without real photos.

The doll is drawn on a 600 × 1000 canvas with exactly known geometry:
  top of head y = 50, soles y = 950  → 900 px tall
  shoulders 180 px apart, shoulder line y = 230, hip line y = 530 (torso 300 px)
  torso silhouette 140 px wide (x 230..369) from y = 220 to 540
so every expected measurement can be calculated by hand in the tests.
"""

from dataclasses import replace

import cv2
import numpy as np

from app.services.pose import Landmark, Pose

WIDTH, HEIGHT = 600, 1000
HEAD_TOP_Y, SOLE_Y = 50, 950
BODY_PX = SOLE_Y - HEAD_TOP_Y
TORSO_LEFT, TORSO_RIGHT = 230, 369  # inclusive → 140 px wide
TORSO_WIDTH_PX = TORSO_RIGHT - TORSO_LEFT + 1

# Landmark positions (x, y) by MediaPipe index. "Left" = the person's left = image right.
BASE_POINTS = {
    0: (300, 130),  # nose
    2: (310, 115), 5: (290, 115),  # eyes
    9: (305, 145), 10: (295, 145),  # mouth
    11: (390, 230), 12: (210, 230),  # shoulders
    13: (470, 400), 14: (130, 400),  # elbows: arms held away from the body
    15: (500, 540), 16: (100, 540),  # wrists
    23: (340, 530), 24: (260, 530),  # hips
    25: (335, 740), 26: (265, 740),  # knees
    27: (330, 920), 28: (270, 920),  # ankles
    29: (330, 940), 30: (270, 940),  # heels
    31: (345, 945), 32: (255, 945),  # toes
}  # fmt: skip

# Elbows/wrists hugging the torso (arms touching the body)
ARMS_TOUCHING = {13: (375, 400), 14: (225, 400), 15: (380, 540), 16: (220, 540)}


def _landmarks(points: dict[int, tuple[float, float]], visibility: float = 0.99) -> list[Landmark]:
    landmarks = []
    for i in range(33):
        x, y = points.get(i, (300, 500))  # indices we don't use sit harmlessly in the middle
        landmarks.append(Landmark(x=float(x), y=float(y), z=0.0, visibility=visibility))
    return landmarks


def _draw_mask(points: dict[int, tuple[float, float]]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    p = lambda i: (int(points[i][0]), int(points[i][1]))  # noqa: E731
    cv2.ellipse(mask, (300, 120), (50, 70), 0, 0, 360, 1, -1)  # head: top at y = 50
    cv2.rectangle(mask, (280, 170), (320, 230), 1, -1)  # neck
    cv2.rectangle(mask, (TORSO_LEFT, 220), (TORSO_RIGHT, 540), 1, -1)  # torso
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):  # arms, 30 px thick
        cv2.line(mask, p(a), p(b), 1, 30)
    for a, b in ((23, 25), (25, 27), (24, 26), (26, 28)):  # legs, 40 px thick
        cv2.line(mask, p(a), p(b), 1, 40)
    for ankle in (27, 28):  # feet down to the soles at y = 950
        x = int(points[ankle][0])
        cv2.rectangle(mask, (x - 20, 900), (x + 25, SOLE_Y), 1, -1)
    return mask.astype(np.float32)


def make_pose(
    overrides: dict[int, tuple[float, float]] | None = None,
    visibility: dict[int, float] | None = None,
    with_mask: bool = True,
    shift_x: float = 0.0,
) -> Pose:
    points = {**BASE_POINTS, **(overrides or {})}
    points = {i: (x + shift_x, y) for i, (x, y) in points.items()}
    landmarks = _landmarks(points)
    for i, vis in (visibility or {}).items():
        landmarks[i] = replace(landmarks[i], visibility=vis)
    mask = _draw_mask(points) if with_mask and shift_x == 0 else None
    return Pose(landmarks=landmarks, mask=mask, width=WIDTH, height=HEIGHT)


class FakeDetector:
    """Stands in for PoseDetector: returns the given poses, whatever the image."""

    def __init__(self, poses: list[Pose]) -> None:
        self.poses = poses

    def detect(self, _image: np.ndarray) -> list[Pose]:
        return self.poses

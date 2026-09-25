"""Draws landmarks, the silhouette and every measured line on the photo (for `debug=true`).

Returned as base64 JPEG, only in the response. It is never saved or logged.
"""

import base64

import cv2
import numpy as np

from app.services.measurements import MASK_THRESHOLD, MeasurementResult
from app.services.pose import Pose

# Pairs of landmark indices to connect with lines (a simple skeleton)
SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # shoulders and arms
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32),  # legs and feet
]  # fmt: skip

# BGR colours (OpenCV order)
SKELETON_COLOR = (255, 200, 0)
POINT_COLOR = (0, 220, 255)
MEASURE_COLOR = (60, 60, 255)
MASK_TINT = np.array([80, 200, 80], dtype=np.float32)


def render_debug_image(image_rgb: np.ndarray, pose: Pose, result: MeasurementResult) -> str:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    thickness = max(1, round(max(image.shape[:2]) / 500))

    # Lightly tint the pixels the model thinks are the person
    if pose.mask is not None:
        person = pose.mask > MASK_THRESHOLD
        image[person] = (0.65 * image[person] + 0.35 * MASK_TINT).astype(np.uint8)

    def pt(p) -> tuple[int, int]:
        return int(round(p[0])), int(round(p[1]))

    for a, b in SKELETON:
        cv2.line(image, pt(pose[a].xy), pt(pose[b].xy), SKELETON_COLOR, thickness, cv2.LINE_AA)
    for landmark in pose.landmarks:
        cv2.circle(image, pt(landmark.xy), thickness + 2, POINT_COLOR, -1, cv2.LINE_AA)

    font_scale = 0.4 * thickness
    for line in result.lines:
        start, end = pt(line.start), pt(line.end)
        cv2.line(image, start, end, MEASURE_COLOR, thickness + 1, cv2.LINE_AA)
        # Horizontal lines: label above the left end. Vertical-ish lines: label beside the middle.
        if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
            text_at = (min(start[0], end[0]) + 4, min(start[1], end[1]) - 6)
        else:
            text_at = ((start[0] + end[0]) // 2 + 6, (start[1] + end[1]) // 2)
        # Dark outline behind the text so it's readable on any background
        cv2.putText(image, line.label, text_at, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, line.label, text_at, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")

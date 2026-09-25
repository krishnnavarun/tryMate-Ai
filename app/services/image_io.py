"""Read, validate and decode uploaded images, in memory only.

Hard rule: user photos are never written to disk and never logged.
(Starlette would normally spill uploads over 1 MB to a temp file; app/main.py raises
that limit so uploads stay in RAM. See `keep_uploads_in_memory` there.)
"""

import cv2
import numpy as np
from fastapi import UploadFile

from app.config import MAX_IMAGE_SIDE, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, MIN_IMAGE_SIDE
from app.errors import AppError, ErrorCode


def detect_image_type(data: bytes) -> str | None:
    """Detect JPEG / PNG / WEBP from the file's first bytes ("magic numbers").

    We don't trust the Content-Type header or the file name: any client can set those
    to anything. The first bytes of the file tell us what it really is.
    """
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # WEBP files start with "RIFF", 4 bytes of file size, then "WEBP"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def read_image_upload(upload: UploadFile, field_name: str) -> bytes:
    """Read an uploaded image into memory and check it's an allowed type and size.

    Raises AppError(INVALID_INPUT) with a message that names the field.
    """
    # Read at most one byte over the limit: enough to know it's too big
    # without pulling a huge file into memory.
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    await upload.close()

    if not data:
        raise AppError(ErrorCode.INVALID_INPUT, f"{field_name}: the file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AppError(ErrorCode.INVALID_INPUT, f"{field_name}: the file is larger than {MAX_UPLOAD_MB} MB.")
    if detect_image_type(data) is None:
        raise AppError(ErrorCode.INVALID_INPUT, f"{field_name}: only JPEG, PNG and WEBP images are accepted.")
    return data


def decode_image(data: bytes, field_name: str = "image") -> np.ndarray:
    """Decode image bytes into an RGB NumPy array (height × width × 3, uint8).

    - Phone photos are often stored sideways with an EXIF "orientation" tag that tells
      viewers to rotate them. cv2.imdecode applies that tag by default, so the person is
      upright here (a test in tests/test_image_io.py checks this).
    - Large photos are shrunk so the longest side is MAX_IMAGE_SIDE pixels.
    """
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AppError(ErrorCode.INVALID_INPUT, f"{field_name}: the image could not be read (corrupt file?).")

    height, width = image.shape[:2]
    if min(height, width) < MIN_IMAGE_SIDE:
        raise AppError(
            ErrorCode.INVALID_INPUT,
            f"{field_name}: the image is too small ({width}×{height}). Use a photo at least {MIN_IMAGE_SIDE} px wide.",
        )

    longest = max(height, width)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        # INTER_AREA is the best OpenCV filter for shrinking (avoids jagged edges)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)

    # OpenCV uses BGR channel order; MediaPipe (and everyone else) expects RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

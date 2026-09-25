import struct

import cv2
import numpy as np
import pytest

from app.config import MAX_IMAGE_SIDE
from app.errors import AppError, ErrorCode
from app.services.image_io import decode_image


def _jpeg(width, height) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    assert ok
    return buffer.tobytes()


def _with_exif_orientation(jpeg: bytes, orientation: int) -> bytes:
    """Insert a minimal EXIF block with the given orientation tag, like phone cameras do."""
    tiff = (
        b"II*\x00" + struct.pack("<I", 8)  # little-endian TIFF header, first IFD at offset 8
        + struct.pack("<H", 1)  # 1 entry
        + struct.pack("<HHII", 0x0112, 3, 1, orientation)  # Orientation, SHORT, count 1
        + struct.pack("<I", 0)  # no next IFD
    )  # fmt: skip
    app1 = b"\xff\xe1" + struct.pack(">H", len(tiff) + 8) + b"Exif\x00\x00" + tiff
    return jpeg[:2] + app1 + jpeg[2:]


def test_decodes_to_rgb_array():
    image = decode_image(_jpeg(400, 600))
    assert image.shape == (600, 400, 3)
    assert image.dtype == np.uint8


def test_exif_rotation_is_applied():
    # Stored landscape (600 × 400) but tagged "rotate 90°": should come out portrait
    sideways = _with_exif_orientation(_jpeg(600, 400), orientation=6)
    assert decode_image(sideways).shape[:2] == (600, 400)


def test_large_images_are_shrunk():
    image = decode_image(_jpeg(2000, 3000))
    assert max(image.shape[:2]) == MAX_IMAGE_SIDE
    assert image.shape[:2] == (MAX_IMAGE_SIDE, round(2000 * MAX_IMAGE_SIDE / 3000))


def test_too_small_image_is_invalid():
    with pytest.raises(AppError) as info:
        decode_image(_jpeg(200, 150))
    assert info.value.code == ErrorCode.INVALID_INPUT


def test_corrupt_image_is_invalid():
    with pytest.raises(AppError) as info:
        decode_image(b"\xff\xd8\xff" + b"\x00" * 500)
    assert info.value.code == ErrorCode.INVALID_INPUT

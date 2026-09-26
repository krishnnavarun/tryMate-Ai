"""Mock try-on: builds a placeholder image locally, for free, in milliseconds.

Used when TRYON_MOCK=true or TRYON_PROVIDER=mock, so development and tests never spend
Replicate credits. The image is the person photo with the garment pasted in a corner and a
"MOCK TRY-ON" banner, so it's obviously not a real result.
"""

import base64

import cv2
import numpy as np

from app.tryon.base import TryOnProvider, TryOnRequest, TryOnResult

MAX_SIDE = 768


def _decode(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class MockTryOnProvider(TryOnProvider):
    name = "mock"

    async def run(self, request: TryOnRequest) -> TryOnResult:
        person = _decode(request.person_image)
        if person is None:
            person = np.full((1024, 768, 3), 230, np.uint8)
        scale = min(1.0, MAX_SIDE / max(person.shape[:2]))
        person = cv2.resize(person, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = person.shape[:2]

        # Garment thumbnail in the top-right corner (only when the garment was uploaded as a file)
        garment = _decode(request.garment_image) if request.garment_image else None
        if garment is not None:
            side = max(1, width // 3)
            thumb = cv2.resize(garment, (side, int(side * garment.shape[0] / garment.shape[1])))
            thumb = thumb[: height - 10]
            person[10 : 10 + thumb.shape[0], width - side - 10 : width - 10] = thumb

        # Banner across the top
        cv2.rectangle(person, (0, 0), (width, 36), (40, 40, 40), -1)
        cv2.putText(person, "MOCK TRY-ON", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        ok, buffer = cv2.imencode(".jpg", person, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return TryOnResult(image_base64=base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None)

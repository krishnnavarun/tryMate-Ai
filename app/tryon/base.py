"""The try-on provider interface.

/try-on only talks to a TryOnProvider, never to Replicate directly, so the model behind it
can be swapped (CatVTON, a self-hosted GPU model...) by adding one class and changing
TRYON_PROVIDER, without touching the API.

Honesty rule (PROJECT_SPEC.md §3): try-on shows how a garment LOOKS on the person, not how
it FITS. Fit comes from /recommend-size.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas import Category


@dataclass
class TryOnRequest:
    person_image: bytes  # validated JPEG/PNG/WEBP bytes, in memory only
    category: Category
    garment_image: bytes | None = None  # exactly one of garment_image / garment_image_url
    garment_image_url: str | None = None
    garment_description: str | None = None


@dataclass
class TryOnResult:
    """Exactly one of image_url / image_base64 is set."""

    image_url: str | None = None
    image_base64: str | None = None


class TryOnProvider(ABC):
    name: str  # returned to the caller as "provider"

    @abstractmethod
    async def run(self, request: TryOnRequest) -> TryOnResult:
        """Generate the try-on image.

        Raise AppError(TRYON_FAILED) for provider errors and AppError(TRYON_TIMEOUT) when it
        takes too long; never let provider-specific exceptions escape.
        """

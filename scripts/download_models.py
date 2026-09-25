"""Download the MediaPipe model files into ./models (gitignored).

Usage:
    python scripts/download_models.py                 # all models
    python scripts/download_models.py --only pose_landmarker_full.task face_detector.tflite
    python scripts/download_models.py --force         # re-download even if present

Which pose model to use (lite / full / heavy) is decided in R&D; all three are listed
so we can compare them on real photos. Files are downloaded to a ".part" file first and
renamed when complete, so an interrupted download never leaves a broken model behind.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://storage.googleapis.com/mediapipe-models"

# local file name -> download URL
MODELS: dict[str, str] = {
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "face_detector.tflite": f"{BASE}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}


def download(name: str, url: str, force: bool) -> None:
    target = MODELS_DIR / name
    if target.exists() and not force:
        print(f"  skip      {name} (already downloaded)")
        return

    partial = target.with_name(target.name + ".part")
    print(f"  download  {name} ...", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
            while chunk := response.read(1024 * 256):
                out.write(chunk)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        print(" failed")
        raise
    print(f" done ({target.stat().st_size / 1_048_576:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(MODELS), help="download only these files")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    names = args.only or list(MODELS)
    print(f"Saving models to {MODELS_DIR}")
    for name in names:
        download(name, MODELS[name], args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

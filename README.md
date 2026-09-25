# tryMate AI Service

Python + FastAPI service for the tryMate store: body measurements, skin tone,
size recommendation and virtual try-on. Only the store's Express server calls it
(server-to-server, `X-API-Key` header). The full spec is in [PROJECT_SPEC.md](PROJECT_SPEC.md).

> **Status: Phase 2 (pose + measurements) done.** `/analyze` returns real body
> measurements from the photo. Skin tone and colors are still placeholders (Phase 3).

| Endpoint | Auth | Current behaviour |
|---|---|---|
| `GET /health` | none | real |
| `POST /analyze` | `X-API-Key` | **real measurements**, confidence, warnings, debug image; skin tone/colors are placeholders (Phase 3) |
| `POST /recommend-size` | `X-API-Key` | validates input, recommends the middle size of the chart (Phase 4) |
| `POST /try-on` | `X-API-Key` | validates input, returns a placeholder image URL (Phase 5) |

---

## How the measurements work

Full details, with the maths, are in the docstring at the top of
[`app/services/measurements.py`](app/services/measurements.py). In short:

1. **MediaPipe PoseLandmarker** (heavy model) finds 33 body landmarks and a person
   segmentation mask. Checks: nobody found → `NO_PERSON_DETECTED`; a second clearly
   visible person → `MULTIPLE_PEOPLE`; nose/shoulders/hips/ankles hidden or outside the
   photo → `PARTIAL_BODY` (the message names the missing parts).
2. **Scale:** the top of the head and the soles come from the mask; `cm_per_px = height_cm / pixel height`.
3. **Lengths** from landmarks: shoulder (× a 1.15 correction, because the landmarks are the
   shoulder *joints*), torso (shoulder line → hip line), arm (shoulder → elbow → wrist),
   leg (hip → knee → ankle).
4. **Chest / waist:** the torso's silhouette width at chest/waist height, multiplied by
   circumference ÷ breadth ratios computed from **ANSUR II** (US Army anthropometric survey,
   2012, 4,082 men): chest × 3.658, waist × 2.879. Chest/waist heights (27.7% / 71.2% of the
   way from shoulders to hips) come from the same data.
5. **Arms touching the body:** the width search never goes past the arm. If the arm touches
   the torso, the breadth is estimated from shoulder width instead and a warning is added.
6. **Confidence** = average landmark visibility × a penalty for each fallback/problem.

### Expected accuracy (honest)
- Even with a perfect width, turning a front-view **width into a circumference** has a
  typical error of **~4.7 cm (chest)** and **~2.6 cm (waist)** in ANSUR II, because bodies
  differ in depth. Real photos add error on top: realistically **±4–8 cm** for chest/waist
  and **±2–3 cm** for lengths with a good photo.
- **Loose clothing inflates chest/waist** (a baggy tee can add 10 cm), and forearms pointing
  at the camera make arms look shorter.
- A single front photo can't see body depth, so it will never be as accurate as a tape measure.

### Photo rules that matter most
Full body head to feet · fitted clothes · stand straight, facing the camera · arms held
~30° away from the body (an "A" pose) · camera at chest height, 2–3 m away · plain background, good light.

### Calibrating (Phase 2 "done when")
Compare with a measuring tape, then tune the constants at the top of
`app/services/measurements.py`. The most likely ones to need a change:
`SHOULDER_WIDTH_FACTOR` (1.15, set from one test photo), `CHEST_LEVEL` / `WAIST_LEVEL`,
and `DIP_THRESHOLD`. Use `debug=true` to see exactly which lines were measured.

**What was verified:** on synthetic "paper doll" poses, every number matches the
hand-calculated value (tests). On a public-domain test photo, all the steps ran and the
debug image showed the right lines. The clean photo gave plausible values; the photo with
a leg hidden behind a chair correctly returned `PARTIAL_BODY`. **Not yet verified:**
accuracy against real tape measurements (needs your photo + tape).

---

## Setup

Requires **Python 3.11+** (developed and tested on 3.13).

```bash
# 1. Create a virtual environment (a private folder of packages for this project)
python -m venv .venv

# 2. Activate it
.venv\Scripts\activate            # Windows (PowerShell / cmd)
source .venv/Scripts/activate     # Windows (Git Bash)
source .venv/bin/activate         # macOS / Linux

# 3. Install packages (runtime + test tools)
pip install -r requirements-dev.txt

# 4. Config: copy the example and edit if needed
copy .env.example .env            # Windows
cp .env.example .env              # macOS / Linux

# 5. Download the MediaPipe model files into ./models (≈ 50 MB, used from Phase 2)
python scripts/download_models.py
```

The MERN mapping: `.venv` ≈ `node_modules`, `requirements.txt` ≈ `package.json`
dependencies, `pip install -r` ≈ `npm install`, `pytest` ≈ `jest`.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs (click **Authorize** and enter your `SERVICE_API_KEY`)
- OpenAPI JSON: http://localhost:8000/openapi.json

## Test

```bash
pytest
```

## Environment variables

| Name | Default | Meaning |
|---|---|---|
| `SERVICE_API_KEY` | `change-me` | Shared secret. Must match `AI_SERVICE_KEY` in the store's `server/.env`. |
| `REPLICATE_API_TOKEN` | *(empty)* | Replicate token for try-on (Phase 5). |
| `TRYON_PROVIDER` | `replicate_idm` | `replicate_idm` \| `catvton` \| `mock` |
| `TRYON_MOCK` | `false` | `true` = placeholder image, no Replicate cost. |
| `LOG_LEVEL` | `info` | `debug` \| `info` \| `warning` \| `error` |
| `PORT` | `8000` | Used by `python -m app.main`. |
| `POSE_MODEL` | `heavy` | `lite` \| `full` \| `heavy`. Heavy is the most accurate (~70 ms per photo on a laptop CPU). |

---

## curl examples

Examples use Git Bash / macOS / Linux syntax. `photo.jpg` is any JPEG/PNG/WEBP under 10 MB.

```bash
# Health (no key)
curl http://localhost:8000/health

# Analyze
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: change-me" \
  -F "image=@photo.jpg" \
  -F "height_cm=175" \
  -F "weight_kg=72"

# Analyze + save the debug image (landmarks and measured lines drawn on the photo)
curl -s -X POST http://localhost:8000/analyze \
  -H "X-API-Key: change-me" -F "image=@photo.jpg" -F "height_cm=175" -F "debug=true" \
  | python -c "import sys,json,base64; open('debug.jpg','wb').write(base64.b64decode(json.load(sys.stdin)['debug_image_base64']))"

# Recommend size
curl -X POST http://localhost:8000/recommend-size \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "measurements": {"shoulder_cm": 44.1, "chest_cm": 96.5, "waist_cm": 84.0, "torso_cm": 62.3, "arm_cm": 60.2, "leg_cm": 81.7},
    "size_chart": {
      "S": {"chest": [86, 92], "waist": [74, 80], "length": [68, 70], "shoulder": [41, 43]},
      "M": {"chest": [92, 98], "waist": [80, 86], "length": [70, 72], "shoulder": [43, 45]},
      "L": {"chest": [98, 104], "waist": [86, 92], "length": [72, 74], "shoulder": [45, 47]}
    },
    "category": "upper_body",
    "fit_preference": "regular"
  }'

# Try-on with a garment URL
curl -X POST http://localhost:8000/try-on \
  -H "X-API-Key: change-me" \
  -F "person_image=@photo.jpg" \
  -F "category=upper_body" \
  -F "garment_image_url=https://example.com/shirt.jpg" \
  -F "garment_description=navy blue cotton polo shirt"

# Try-on with a garment file
curl -X POST http://localhost:8000/try-on \
  -H "X-API-Key: change-me" \
  -F "person_image=@photo.jpg" \
  -F "garment_image=@shirt.jpg" \
  -F "category=upper_body"

# Errors come back as { "error_code": "...", "message": "..." }
curl -X POST http://localhost:8000/analyze -H "X-API-Key: wrong" -F "image=@photo.jpg" -F "height_cm=175"
# → 401 {"error_code":"UNAUTHORIZED", ...}
```

In PowerShell use `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest`) and put
everything on one line, or replace `\` with a backtick.

---

## Project layout

```
tryMate-Ai/
├── app/
│   ├── main.py            # creates the FastAPI app: middleware, error handlers, routers
│   ├── config.py          # settings from .env + fixed limits (10 MB uploads)
│   ├── security.py        # X-API-Key dependency
│   ├── errors.py          # ErrorCode, AppError, handlers → {error_code, message}
│   ├── middleware.py      # request-size limit (keeps in-memory uploads bounded)
│   ├── schemas.py         # Pydantic models = the API contract
│   ├── routers/           # health, analyze, sizing, tryon (thin: validate → call services)
│   ├── services/
│   │   ├── image_io.py    # in-memory upload validation + decode (EXIF rotation, resize)
│   │   ├── pose.py        # MediaPipe PoseLandmarker + person checks
│   │   ├── measurements.py # landmarks + mask + height → cm (the maths, explained)
│   │   ├── body_analysis.py # decode → pose → measure, used by /analyze
│   │   └── debug_image.py # draws landmarks and measured lines (debug=true)
│   └── tryon/             # try-on providers (Phase 5)
├── models/                # MediaPipe .task files (gitignored; scripts/download_models.py)
├── scripts/download_models.py
├── tests/                 # pytest; images + a synthetic "paper doll" person (fakes.py) are generated in memory
│   └── images/            # your own photos for Phase 2+ (gitignored)
├── .env.example
├── requirements.txt       # runtime packages
├── requirements-dev.txt   # + pytest, httpx2
└── pytest.ini
```

Planned (added in later phases): `services/skin_tone.py`,
`services/colors.py`, `services/sizing.py`, `tryon/base.py`, `tryon/replicate_idm.py`, `tryon/mock.py`, `Dockerfile`.

---

## Implementation notes

- **Photos never touch disk.** Starlette normally writes uploads over 1 MB to a temp
  file. `app/main.py` (`keep_uploads_in_memory`) raises that threshold above the max
  request size, and `BodySizeLimitMiddleware` caps requests at 21 MB (two 10 MB images +
  form fields), so uploads always stay in RAM. There's a test for it.
- **File type is checked from the bytes**, not the `Content-Type` header or file name.
- **Validation errors** (bad fields, bad JSON, too large) all become
  `422 {"error_code": "INVALID_INPUT", "message": "<field>: <reason>"}`.
- **Unknown routes** return `404` in the same error shape.
- **`/try-on`** requires exactly one of `garment_image` / `garment_image_url`; sending both is
  `INVALID_INPUT`, so it's always clear which one was used.
- **OpenCV package:** `mediapipe` 1.0.1 depends on `opencv-contrib-python`. Don't also install
  `opencv-python-headless`: both install into the same `cv2` folder and break each other.
  For Docker (Phase 6) the non-headless build needs `libgl1` + `libglib2.0-0` in the image.

## Open decisions (for R&D)

These are marked in the code and are yours to decide:

1. **FACE_NOT_FOUND**: fail the whole `/analyze`, or return measurements with
   `skin_tone: null`, `color_suggestions: []` and a warning? The schema already allows
   `skin_tone: null`, so both options fit without a contract change.
2. **Skin `tone` labels**: a plain string for now; pick a fixed list in Phase 3.
3. **Size chart field names**: any name is accepted (`chest`, `waist`, `length`, `shoulder`,
   and later e.g. `hip`, `inseam` for lower body). Scoring rules come in Phase 4.
4. ~~**Pose model**~~: decided in Phase 2: **heavy** (most accurate, still ~70 ms per photo). Change with `POSE_MODEL`.

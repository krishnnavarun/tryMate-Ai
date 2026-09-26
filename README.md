# tryMate AI Service

[![CI](https://github.com/krishnnavarun/tryMate-Ai/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnnavarun/tryMate-Ai/actions/workflows/ci.yml)

Python + FastAPI service for the tryMate store: body measurements, skin tone,
size recommendation and virtual try-on. Only the store's Express server calls it
(server-to-server, `X-API-Key` header). The full spec is in [PROJECT_SPEC.md](PROJECT_SPEC.md).

> **Status: Phases 1–6 built.** All four endpoints are real. What's left needs you:
> calibrating measurements against a tape measure and one real Replicate try-on (needs a token).
> See [What still needs you](#what-still-needs-you).

| Endpoint | Auth | What it does |
|---|---|---|
| `GET /health` | none | `{status, version}` |
| `POST /analyze` | `X-API-Key` | photo + height → measurements, skin tone, 8 suggested colors, confidence, warnings, optional debug image |
| `POST /recommend-size` | `X-API-Key` | measurements + size chart → best size + score and fit note per size |
| `POST /try-on` | `X-API-Key` | person photo + garment → try-on image (Replicate IDM-VTON, or a free local mock) |

---

## How it works

### 1. Measurements (`/analyze`)
Full maths in [`app/services/measurements.py`](app/services/measurements.py).

1. **MediaPipe PoseLandmarker** (heavy model) finds 33 body landmarks and a person
   segmentation mask. Checks: nobody → `NO_PERSON_DETECTED`; a second clearly visible
   person → `MULTIPLE_PEOPLE`; nose/shoulders/hips/ankles hidden or cut off → `PARTIAL_BODY`
   (the message names what's missing).
2. **Scale:** top of head and soles come from the mask; `cm_per_px = height_cm / pixel height`.
3. **Lengths** from landmarks: shoulder (× 1.15, because the landmarks are the shoulder
   *joints*), torso (shoulder line → hip line), arm (shoulder → elbow → wrist), leg (hip → knee → ankle).
4. **Chest / waist:** silhouette width at chest/waist height × circumference-to-breadth ratios
   computed from **ANSUR II** (US Army anthropometric survey 2012, 4,082 men): chest × 3.658,
   waist × 2.879. Chest/waist heights (27.7% / 71.2% of the way from shoulders to hips) come
   from the same data.
5. **Arms touching the body:** the width search stops at the arm; if the arm touches the torso,
   the breadth is estimated from shoulder width and a warning is added.
6. **Confidence** = average landmark visibility × a penalty for each fallback/problem.

**Accuracy (honest):** even with a perfect width, width → circumference has a typical error of
~4.7 cm (chest) and ~2.6 cm (waist) in ANSUR II. With a good photo expect roughly ±4–8 cm for
chest/waist and ±2–3 cm for lengths. Loose clothes inflate chest/waist; forearms pointing at
the camera make arms look short. A front photo can't see depth, so it won't match a tape measure.

**Photo rules:** full body head to feet · fitted clothes · facing the camera · arms ~30° away
from the body ("A" pose) · camera at chest height, 2–3 m away · plain background · daylight.

### 2. Skin tone and colors (`/analyze`)
Full details in [`app/services/skin_tone.py`](app/services/skin_tone.py) and
[`app/services/colors.py`](app/services/colors.py).

1. **Face:** the head is cropped using the pose landmarks, enlarged, and MediaPipe
   **FaceLandmarker** finds the 478-point face mesh (faces in full-body photos are too small
   to find otherwise).
2. **Sampling:** three small circles on the mid-forehead and both cheeks (away from eyes,
   brows, lips and hairline). Pixels outside the classic YCrCb skin range (Chai & Ngan 1999)
   and the darkest 15% / brightest 10% are dropped; the median colour in CIELAB is used.
3. **Lighting:** "white patch" correction scales the skin pixels so the photo's brightest
   near-grey areas become white (bounded; skipped when there are none). Strong colour casts and
   dark photos add a warning.
4. **Tone:** the **Individual Typology Angle** `ITA = atan((L* − 50) / b*)`, a standard
   dermatology measure, with the Del Bino et al. (2006) bands →
   `fair` · `light` · `medium` · `tan` · `brown` · `deep`.
5. **Undertone:** hue angle `atan2(b*, a*)`: more golden = `warm`, more pink = `cool`, else
   `neutral` (heuristic thresholds 40° / 52°, tune with real photos).
6. **Colors:** undertone picks the family (warm = earthy/golden, cool = blue-based/jewel,
   neutral = soft mix), depth picks the strength (lighter skin → softer colours, deeper skin →
   brighter ones). 9 palettes × 8 named colours.

**FACE_NOT_FOUND decision:** if the body is fine but the face can't be found, `/analyze` still
returns **200** with the measurements, `skin_tone: null`, `color_suggestions: []` and a warning.
Measurements are the main value of a scan, so a photo that's good for sizing isn't rejected.

**Known limitation:** a camera records skin × light, so the tone depends on lighting. Test
photos taken indoors or against the light came out too deep. Results are most reliable in
soft daylight facing a window. On one photo, the result stayed the same across mirroring,
resizing, JPEG quality and ±15% exposure, except +15% brightness moved deep → medium.

### 3. Size recommendation (`/recommend-size`)
Full details in [`app/services/sizing.py`](app/services/sizing.py).

- Chart fields compared: `chest`, `waist`, `shoulder`, `length` (tops/dresses: ≈ torso × 1.55)
  and `inseam` (lower body). Other fields (e.g. `hip`) are ignored.
- **Fit preference** moves the ideal point inside each range: regular = middle,
  slim = 75% up (snug), loose = 25% up (roomy). Length ignores the preference.
- **Score** = `exp(−½ × Σ wᵢ dᵢ² / Σ wᵢ)`, where `dᵢ` = (body − ideal) / σ (σ ≈ 4 cm for
  chest/waist, 2 cm for shoulders, close to the measurement error). **Weights** by category:
  upper body = chest 0.45 · shoulder 0.25 · waist 0.15 · length 0.15.
- **Best size** = highest score (a tie goes to the larger size).
- **Notes** from the direction of each deviation over 0.75σ: "Tight at chest and shoulders",
  "Loose at shoulders, slightly long", "Good fit".
- A chart with no comparable fields → `422 INVALID_INPUT`.

Hand-checked: textbook M → M ("Good fit"); exactly between M and L → slim M, regular L,
loose L; bigger than every size → XL; smaller than every size → S; broad shoulders → M with
"Tight at shoulders". All of these are tests.

### 4. Virtual try-on (`/try-on`)
- A provider interface ([`app/tryon/base.py`](app/tryon/base.py)) with two providers:
  - **`replicate_idm`**: [IDM-VTON on Replicate](https://replicate.com/cuuupid/idm-vton).
    ~17 s typical on an A100 (longer on a cold start), ≈ **$0.023 per run**.
    **Licence: CC BY-NC-SA 4.0 — non-commercial only.** Fine for a demo/portfolio; a commercial
    store needs a commercially licensed model, which can be added as another provider.
  - **`mock`**: builds a placeholder locally (the person photo + garment thumbnail + "MOCK
    TRY-ON" banner), free and instant. Used when `TRYON_MOCK=true`.
- Timeout **120 s** (the Replicate prediction is cancelled, so you stop paying) → `TRYON_TIMEOUT`.
  Model error → `TRYON_FAILED`. Network error / Replicate 5xx / 429 → **retried once**.
- Rate limit: at most `TRYON_RATE_LIMIT_PER_MINUTE` (default 30) try-ons per minute in
  total → `429 RATE_LIMITED`. The store also limits try-ons per user.
- Try-on shows how the garment **looks**, not how it **fits**; fit comes from `/recommend-size`.

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

# 5. Download the MediaPipe model files into ./models (≈ 50 MB)
python scripts/download_models.py
```

**Linux (Debian/Ubuntu) only:** MediaPipe and OpenCV need a few system libraries, even
without a screen or GPU. Windows and macOS need nothing extra.

```bash
sudo apt-get install -y libgl1 libglib2.0-0 libegl1 libgles2
```

Without `libegl1`/`libgles2`, loading a model fails with
`OSError: libGLESv2.so.2: cannot open shared object file`. The service still starts and
`/health` answers, but the log shows the error and `/analyze` fails until they're installed.

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
pytest          # 165 tests, a few seconds; never calls Replicate
```

The API tests use fake detectors (also at startup), so only a handful of tests load the real
MediaPipe models.

**CI** (`.github/workflows/ci.yml`) runs on every push to `main` and every pull request:
- `test`: installs the system libraries and the requirements on Ubuntu, downloads the models
  (cached), runs `pytest`.
- `docker`: builds the Docker image, starts it and smoke-tests the running container (`/health`,
  the API key check, `/recommend-size`, and `/analyze` running MediaPipe on an empty picture).

## Environment variables

| Name | Default | Meaning |
|---|---|---|
| `SERVICE_API_KEY` | `change-me` | Shared secret. Must match `AI_SERVICE_KEY` in the store's `server/.env`. |
| `REPLICATE_API_TOKEN` | *(empty)* | Replicate token for real try-on. |
| `TRYON_PROVIDER` | `replicate_idm` | `replicate_idm` \| `mock` (`catvton` is reserved, not implemented) |
| `TRYON_MOCK` | `false` | `true` = always use the free mock provider. |
| `REPLICATE_IDM_VERSION` | *(empty)* | Pin an IDM-VTON version id; empty = the model's latest version. |
| `TRYON_RATE_LIMIT_PER_MINUTE` | `30` | Global cap on try-ons per minute (0 = off). |
| `POSE_MODEL` | `heavy` | `lite` \| `full` \| `heavy` (heavy ≈ 70 ms per photo on a laptop CPU). |
| `LOG_LEVEL` | `info` | `debug` \| `info` \| `warning` \| `error` |
| `LOG_FORMAT` | `text` | `text` (readable) \| `json` (one JSON object per line, for production) |
| `PORT` | `8000` | Used by `python -m app.main` and the Docker image. |

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

# Analyze + save the debug image (landmarks, measured lines and skin sample spots)
curl -s -X POST http://localhost:8000/analyze \
  -H "X-API-Key: change-me" -F "image=@photo.jpg" -F "height_cm=175" -F "debug=true" \
  | python -c "import sys,json,base64; open('debug.jpg','wb').write(base64.b64decode(json.load(sys.stdin)['debug_image_base64']))"

# Recommend size
curl -X POST http://localhost:8000/recommend-size \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "measurements": {"shoulder_cm": 44.0, "chest_cm": 95.0, "waist_cm": 83.0, "torso_cm": 46.5, "arm_cm": 60.0, "leg_cm": 82.0},
    "size_chart": {
      "S": {"chest": [86, 92], "waist": [74, 80], "length": [68, 70], "shoulder": [41, 43]},
      "M": {"chest": [92, 98], "waist": [80, 86], "length": [70, 72], "shoulder": [43, 45]},
      "L": {"chest": [98, 104], "waist": [86, 92], "length": [72, 74], "shoulder": [45, 47]}
    },
    "category": "upper_body",
    "fit_preference": "regular"
  }'
# → {"recommended_size":"M","per_size":{"S":{"score":0.43,"note":"Tight at chest and waist, ..."}, "M":{"score":0.99,"note":"Good fit"}, ...}}

# Try-on with a garment URL (TRYON_MOCK=true returns result_image_base64 instantly)
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

### Error codes

| Code | HTTP | When |
|---|---|---|
| `UNAUTHORIZED` | 401 | Missing or wrong `X-API-Key` |
| `INVALID_INPUT` | 422 | Bad fields, wrong file type, too large/small, unusable size chart |
| `NO_PERSON_DETECTED` | 422 | No pose found |
| `MULTIPLE_PEOPLE` | 422 | More than one person |
| `PARTIAL_BODY` | 422 | Head, shoulders, hips or ankles hidden or cut off |
| `FACE_NOT_FOUND` | 422 | In the contract, but `/analyze` returns 200 + a warning instead (see above) |
| `RATE_LIMITED` | 429 | Too many try-ons this minute (**added**, not in the original contract table) |
| `TRYON_FAILED` | 502 | Provider error, or try-on not configured |
| `TRYON_TIMEOUT` | 504 | Provider took more than 120 s |
| `INTERNAL_ERROR` | 500 | Anything else |

---

## Production

### Docker
```bash
docker build -t trymate-ai .
docker run --rm -p 8000:8000 --env-file .env trymate-ai
```
- Python 3.13 slim + `libgl1`/`libglib2.0-0` (the OpenCV build that MediaPipe uses) and
  `libegl1`/`libgles2` (MediaPipe's native library links against `libEGL` and `libGLESv2`).
- Models (pose heavy + face landmarker) are downloaded **at build time**.
- Runs as a non-root user, `LOG_FORMAT=json`, one worker, a `HEALTHCHECK` on `/health`.
- The Dockerfile hasn't been built on the development machine (no Docker there); the CI
  `docker` job builds and smoke-tests it on every push.
- **With the store:** `tryMate-store/docker-compose.yml` builds this image as its `ai` service
  (`docker compose --profile ai up --build -d`, with this repo next to tryMate-store), so the whole
  stack runs with one command. See the store README → "Docker".

### Logging and request ids
- Every request gets an id: the caller's `X-Request-ID` (the store can send one), or a new one.
  It's returned in the `X-Request-ID` response header and included in every log line.
- One access-log line per request (method, path, status, duration). **No bodies, no image
  bytes, no API keys** are ever logged.

### Deployment notes
- **Memory:** with both models loaded the process uses ~255 MB resident / ~690 MB committed,
  and more while analysing a photo. Use an instance with **at least 1 GB RAM** (Render
  "Standard", Railway 1 GB, or a small VM). Free 512 MB tiers will run out of memory.
- **CPU:** one analysis takes ~0.1–0.5 s of CPU. Scale by running more containers, not more
  workers per container (each worker loads its own models).
- **Render / Railway:** deploy from the Dockerfile, set the env vars (at least
  `SERVICE_API_KEY`, `REPLICATE_API_TOKEN` or `TRYON_MOCK=true`). The platform sets `PORT`.
- **Small VM:** `docker run -d --restart unless-stopped -p 8000:8000 --env-file .env trymate-ai`.
- **Security:** keep the service private if the platform allows it (only the store's server
  needs to reach it); otherwise rely on a long random `SERVICE_API_KEY`. CORS is intentionally off.

---

## Project layout

```
tryMate-Ai/
├── app/
│   ├── main.py             # creates the FastAPI app: middleware, error handlers, routers, model warm-up
│   ├── config.py           # settings from .env + fixed limits (10 MB uploads)
│   ├── security.py         # X-API-Key dependency
│   ├── errors.py           # ErrorCode, AppError, handlers → {error_code, message}
│   ├── middleware.py       # request-size limit (keeps in-memory uploads bounded)
│   ├── logging_setup.py    # JSON logs, request ids, access log
│   ├── rate_limit.py       # sliding-window limiter for /try-on
│   ├── schemas.py          # Pydantic models = the API contract
│   ├── routers/            # health, analyze, sizing, tryon (thin: validate → call services)
│   ├── services/
│   │   ├── image_io.py     # upload validation + decode (EXIF rotation, resize)
│   │   ├── pose.py         # MediaPipe PoseLandmarker + person checks
│   │   ├── measurements.py # landmarks + mask + height → cm
│   │   ├── face.py         # head crop + MediaPipe FaceLandmarker
│   │   ├── skin_tone.py    # face → tone (ITA), undertone, hex
│   │   ├── colors.py       # tone + undertone → 8 named colours
│   │   ├── sizing.py       # measurements + size chart → scores and notes
│   │   ├── body_analysis.py # the /analyze pipeline
│   │   └── debug_image.py  # draws what was measured (debug=true)
│   └── tryon/
│       ├── base.py         # TryOnProvider interface
│       ├── replicate_idm.py # IDM-VTON on Replicate (timeout, cancel, retry once)
│       └── mock.py         # free local placeholder
├── models/                 # MediaPipe model files (gitignored; scripts/download_models.py)
├── scripts/download_models.py
├── tests/                  # pytest; images, a synthetic "paper doll" person and a fake face are generated in memory
├── Dockerfile, .dockerignore
├── .env.example
├── requirements.txt        # runtime packages
├── requirements-dev.txt    # + pytest, httpx2
└── pytest.ini
```

---

## Implementation notes

- **Photos never touch disk.** Starlette normally writes uploads over 1 MB to a temp file;
  `keep_uploads_in_memory()` raises that threshold above the max request size, and
  `BodySizeLimitMiddleware` caps requests at 21 MB, so uploads always stay in RAM (tested).
- **File type is checked from the bytes**, not the `Content-Type` header or file name.
- **MediaPipe crash workaround:** MediaPipe 1.0.1 aborts the whole process when reading the
  segmentation mask of an image whose width isn't a multiple of 4. `pose.py` pads the image
  by up to 3 columns first (tested).
- **OpenCV package:** `mediapipe` depends on `opencv-contrib-python`. Don't also install
  `opencv-python-headless`: both write to the same `cv2` folder and break each other.
- **`/try-on`** requires exactly one of `garment_image` / `garment_image_url`.
- **Validation errors** all become `422 INVALID_INPUT` with a `"<field>: <reason>"` message;
  unknown routes return `404` in the same error shape.

## Decisions made (were open questions)

1. **FACE_NOT_FOUND** → 200 with `skin_tone: null`, `color_suggestions: []` + warning.
2. **Tone labels** → `fair`, `light`, `medium`, `tan`, `brown`, `deep` (ITA bands).
3. **Size chart fields** → `chest`, `waist`, `shoulder`, `length`, `inseam`; others ignored.
4. **Pose model** → heavy.
5. **Try-on model** → IDM-VTON on Replicate (non-commercial licence!), behind a provider interface.
6. **Rate limiting** → global per-minute cap here; per-user limit in the store.

## What still needs you

1. **Calibrate measurements** against a tape measure: scan yourself with `debug=true`, compare,
   then tune `SHOULDER_WIDTH_FACTOR`, `CHEST_LEVEL` / `WAIST_LEVEL`, `DIP_THRESHOLD`
   (`measurements.py`) and `LENGTH_PER_TORSO` (`sizing.py`).
2. **Check skin tone** on 2–3 daylight photos of yourself; tune `UNDERTONE_COOL_BELOW` /
   `UNDERTONE_WARM_ABOVE` (`skin_tone.py`) if the undertone looks off.
3. **One real try-on:** set `REPLICATE_API_TOKEN`, `TRYON_MOCK=false`, and call `/try-on` with a
   shirt image (costs ~$0.02). The Replicate code is tested only against a fake client.
4. **Check the CI `docker` job is green** after pushing (Actions tab). It builds the image and
   smoke-tests it; its first run failed on two missing Linux libraries, now fixed.

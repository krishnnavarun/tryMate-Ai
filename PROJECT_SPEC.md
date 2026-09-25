# AI Service — Project Spec (Repo 2 of 2)

> Put this file in the root of the `ai-service` repo as `PROJECT_SPEC.md`.
> Claude: read this whole file before doing anything. Follow the phases in order.
> Wait for my approval after the R&D + plan step and after every phase.

---

## 1. About me and how to work with me

- I'm a MERN (JavaScript) developer. I'm **learning Python** with this project.
- Write clean, readable Python with comments explaining any non-obvious logic,
  especially the measurement math and color logic.
- After each phase: tell me what you built, how to run it, and how to test it
  (curl commands). Then stop and wait for my go-ahead.
- Don't add features outside this spec without asking.
- If something in this spec is technically wrong or outdated, tell me before building.

---

## 2. The product (big picture)

An AI-powered online clothing store that reduces wrong-size orders and returns.
There are **2 repos**:

| Repo | Stack | Responsibility |
|---|---|---|
| `store` | React + Express + MongoDB (JS) | Store, users, products, size charts, cart, UI |
| `ai-service` (this repo) | Python + FastAPI | All AI: body measurements, skin tone, size recommendation, virtual try-on |

Flow: `React client → Express server → this AI service`.
The browser never calls this service directly. Only the Express server calls it,
using a shared secret header.

---

## 3. What this service does

Given a user's full-body photo and their height:

1. **Body measurements** — estimate shoulder width, chest, waist, torso length,
   arm length, and leg length in cm from pose landmarks. Height (entered by the
   user) converts pixels to cm. A single photo has no scale, so height is required.
2. **Skin tone** — detect tone and undertone (warm / cool / neutral) from the
   face region and return clothing colors that suit it.
3. **Size recommendation** — given measurements and a product's size chart (sent by
   the caller), return the best size and a fit note for every size.
4. **Virtual try-on** — given a person image and a garment image, return an image
   of the person wearing the garment.

Important honesty rule: try-on shows **how it looks**, not **how it fits**.
Fit comes from step 3. Never claim the try-on image proves the fit.

---

## 4. Tech decisions

- Python 3.11+
- FastAPI + Uvicorn, Pydantic models for every request and response
- MediaPipe (Python, Tasks API): PoseLandmarker + FaceLandmarker or FaceDetector, IMAGE mode
- OpenCV + NumPy for image handling and color analysis (LAB / HSV)
- Virtual try-on: IDM-VTON through the Replicate Python client, behind a provider
  interface (`app/tryon/base.py` + `app/tryon/replicate_idm.py`) so it can be swapped
  for CatVTON or a self-hosted model later without changing the API
- `python-multipart` for file uploads
- `pytest` + FastAPI TestClient for tests
- `.env` via `pydantic-settings`; provide `.env.example`
- Dockerfile + `requirements.txt`

### Hard rules
- **Stateless.** No database.
- **Never write user photos to disk.** Process in memory only. Don't log image bytes.
- Every endpoint except `/health` requires the header `X-API-Key: <SERVICE_API_KEY>`.
- Max upload size: 10 MB. Accept JPEG / PNG / WEBP only.
- CORS: not needed (server-to-server), keep it closed.

### Environment variables
```
SERVICE_API_KEY=change-me
REPLICATE_API_TOKEN=
TRYON_PROVIDER=replicate_idm      # later: catvton, mock
TRYON_MOCK=false                  # true = return a placeholder image, no Replicate cost
LOG_LEVEL=info
PORT=8000
```

---

## 5. API contract (FIXED — the store repo is built against this exact contract)

Do not change field names or shapes without telling me, because the other repo depends on them.

### `GET /health`
```json
{ "status": "ok", "version": "0.1.0" }
```

### `POST /analyze`
Request: `multipart/form-data`
- `image` (file, required): full-body front photo
- `height_cm` (number, required, 120–230)
- `weight_kg` (number, optional)
- `debug` (bool, optional): if true, also return an image with landmarks drawn

Response `200`:
```json
{
  "measurements": {
    "shoulder_cm": 44.1,
    "chest_cm": 96.5,
    "waist_cm": 84.0,
    "torso_cm": 62.3,
    "arm_cm": 60.2,
    "leg_cm": 81.7
  },
  "skin_tone": { "tone": "medium", "undertone": "warm", "hex": "#C68E6A" },
  "color_suggestions": [
    { "name": "Olive", "hex": "#708238" },
    { "name": "Mustard", "hex": "#D4A017" }
  ],
  "confidence": 0.82,
  "warnings": ["Arms close to body — shoulder width may be less accurate"],
  "debug_image_base64": null
}
```

### `POST /recommend-size`
Request: `application/json`
```json
{
  "measurements": { "shoulder_cm": 44.1, "chest_cm": 96.5, "waist_cm": 84.0, "torso_cm": 62.3, "arm_cm": 60.2, "leg_cm": 81.7 },
  "size_chart": {
    "S": { "chest": [86, 92], "waist": [74, 80], "length": [68, 70], "shoulder": [41, 43] },
    "M": { "chest": [92, 98], "waist": [80, 86], "length": [70, 72], "shoulder": [43, 45] },
    "L": { "chest": [98, 104], "waist": [86, 92], "length": [72, 74], "shoulder": [45, 47] }
  },
  "category": "upper_body",
  "fit_preference": "regular"
}
```
- `size_chart` keys are size labels; each field is `[min, max]` in cm; any field may be missing.
- `fit_preference`: `"slim" | "regular" | "loose"`

Response `200`:
```json
{
  "recommended_size": "M",
  "per_size": {
    "S": { "score": 0.41, "note": "Tight at chest and shoulders" },
    "M": { "score": 0.93, "note": "Good fit" },
    "L": { "score": 0.62, "note": "Loose at shoulders, slightly long" }
  }
}
```

### `POST /try-on`
Request: `multipart/form-data`
- `person_image` (file, required)
- `garment_image` (file) **or** `garment_image_url` (string): one is required
- `category` (`"upper_body" | "lower_body" | "dresses"`, required)
- `garment_description` (string, optional, e.g. "navy blue cotton polo shirt")

Response `200`:
```json
{ "result_image_url": "https://...", "result_image_base64": null, "latency_ms": 23400, "provider": "replicate_idm" }
```
(Exactly one of `result_image_url` / `result_image_base64` is non-null.)

### Errors (all endpoints)
HTTP 4xx/5xx with:
```json
{ "error_code": "NO_PERSON_DETECTED", "message": "No person found in the photo. Use a clear full-body photo." }
```
Error codes:
| Code | HTTP | When |
|---|---|---|
| `UNAUTHORIZED` | 401 | Missing or wrong X-API-Key |
| `INVALID_INPUT` | 422 | Bad fields, wrong file type, too large |
| `NO_PERSON_DETECTED` | 422 | No pose found |
| `MULTIPLE_PEOPLE` | 422 | More than one person |
| `PARTIAL_BODY` | 422 | Key landmarks (shoulders, hips, ankles) missing or low visibility |
| `FACE_NOT_FOUND` | 422 | Face not found (measurements can still be returned with a warning if pose is fine; decide and explain) |
| `TRYON_FAILED` | 502 | Provider error |
| `TRYON_TIMEOUT` | 504 | Provider took more than 120 s |
| `INTERNAL_ERROR` | 500 | Anything else |

---

## 6. Suggested folder structure (improve it if you have a better reason)

```
ai-service/
├── app/
│   ├── main.py              # FastAPI app, routers, error handlers
│   ├── config.py            # settings from .env
│   ├── security.py          # X-API-Key dependency
│   ├── errors.py            # AppError + error codes
│   ├── schemas.py           # Pydantic models (the contract)
│   ├── routers/
│   │   ├── analyze.py
│   │   ├── sizing.py
│   │   └── tryon.py
│   ├── services/
│   │   ├── image_io.py      # load/validate/resize images in memory
│   │   ├── pose.py          # MediaPipe pose → landmarks
│   │   ├── measurements.py  # landmarks + height → cm
│   │   ├── skin_tone.py     # face region → tone, undertone, hex
│   │   ├── colors.py        # tone/undertone → color suggestions
│   │   └── sizing.py        # measurements + size chart → scores
│   └── tryon/
│       ├── base.py          # TryOnProvider interface
│       ├── replicate_idm.py
│       └── mock.py
├── models/                  # downloaded .task model files (gitignored, downloaded by a script)
├── scripts/download_models.py
├── tests/
│   ├── images/              # a few test photos (my own, with permission)
│   └── test_*.py
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 7. Step 0: R&D + plan (do this FIRST, no code yet)

Research and report back on:

1. **Try-on model:** the current Replicate IDM-VTON model. Exact inputs, output format,
   typical latency, approximate cost per run in USD, and license (I believe it's
   non-commercial; confirm). Is there a better or cheaper option right now (CatVTON,
   others)? Recommend one.
2. **MediaPipe Python Tasks API:** current PoseLandmarker usage in IMAGE mode, which model
   file to use (lite / full / heavy) and why, and the landmark indices I need
   (shoulders 11/12, hips 23/24, wrists, ankles, nose, eyes). How to estimate the top of
   the head, since pose has no head-top landmark.
3. **Measurement math:**
   - Pixel-to-cm scale from height (head-top to heel pixel distance vs `height_cm`).
   - A front photo gives **widths**, not **circumferences**. Propose a width →
     circumference approximation (e.g. ellipse approximation or anthropometric
     multipliers) and cite where the numbers come from.
   - Expected error range. Be honest.
   - What photo rules improve accuracy (standing straight, arms slightly away, fitted
     clothes, full body visible, camera at chest height).
4. **Skin tone:** how to sample skin pixels from the face (cheeks/forehead, avoid
   eyes, lips, and hair), how to handle lighting, how to decide undertone
   (e.g. LAB b* / a* values), and a simple explainable tone+undertone → color palette mapping.
5. **Size scoring:** an algorithm that scores each size given measurement ranges,
   category (which measurements matter for upper body vs lower body vs dresses),
   and fit_preference. Include how notes like "tight at chest" are generated.

Then give me:
- R&D summary (findings, risks, what's realistic, what isn't)
- Final folder structure
- The phase plan below, adjusted if needed
- Anything in this spec you disagree with

**Stop and wait for my approval.**

---

## 8. Build phases

Each phase ends with: working code, tests passing, README updated, curl examples, and a stop for my review.

### Phase 1: Skeleton
- `git init`, `.gitignore` (venv, .env, models/, __pycache__), virtual env setup instructions
- FastAPI app, config, `/health`, X-API-Key auth, error format + handlers
- `schemas.py` with ALL contract models already defined
- Stub endpoints for `/analyze`, `/recommend-size`, `/try-on` returning valid fake data that matches the contract
- `scripts/download_models.py`
- Tests: health, auth rejected, auth accepted, stub responses match schema
- **Done when:** `/docs` shows all endpoints with correct request and response models.

### Phase 2: Pose + measurements (`/analyze` part 1)
- In-memory image loading and validation (type, size, EXIF rotation fix, resize)
- PoseLandmarker → landmarks with visibility scores
- Errors: NO_PERSON_DETECTED, MULTIPLE_PEOPLE, PARTIAL_BODY
- measurements.py with the math explained in comments
- `confidence` from landmark visibility + pose quality; `warnings` for things like arms close to body
- `debug=true` returns an image with landmarks and measured lines drawn
- Tests with sample images
- **Done when:** I upload my own photo with my real height and the measurements are in a believable range. I'll compare them with a measuring tape.

### Phase 3: Skin tone + colors (`/analyze` part 2)
- Face detection → skin sample → tone, undertone, hex
- Color suggestions (6–10 colors with names and hex)
- Handle FACE_NOT_FOUND per the decision from R&D
- **Done when:** results are stable across 2–3 photos of the same person.

### Phase 4: `/recommend-size`
- Scoring algorithm + notes, category-aware, fit_preference-aware
- Unit tests with hand-made size charts (edge cases: between sizes, missing fields, out of all ranges)
- **Done when:** results make sense for 5+ hand-checked cases.

### Phase 5: `/try-on`
- Provider interface + Replicate IDM-VTON implementation + mock provider
- Timeout (120 s), retry once on transient errors, TRYON_FAILED / TRYON_TIMEOUT
- `TRYON_MOCK=true` skips Replicate (saves money during development)
- **Done when:** one real try-on works end to end with a shirt image.

### Phase 6: Production-ready
- Dockerfile (models downloaded at build time), README with setup, env vars, curl examples for every endpoint
- Structured logging (no image data in logs), request IDs
- Basic rate limiting on `/try-on`
- Deployment notes (e.g. Render / Railway / a small VM; mention memory needs for MediaPipe)
- **Done when:** `docker build` + `docker run` works and all tests pass.

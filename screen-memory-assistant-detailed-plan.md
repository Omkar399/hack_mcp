# Screen‑Memory Assistant – Comprehensive Engineering Blueprint
### Light‑stack version (Postgres + EnrichMCP + Martian) – July 2025

---

## Table of Contents
1. Product Overview  
2. System Architecture  
3. Technology Matrix  
4. Data Design  
5. API Surface (EnrichMCP schema)  
6. Capture & Processing Pipeline  
7. Operational Concerns  
8. Local Development Setup  
9. End‑to‑End Test Plan  
10. 48‑Hour Hackathon Milestones  
11. Stretch Enhancements  

---

## 1 · Product Overview
**Goal:** Persist a searchable, privacy‑preserving memory of everything shown on screen, runnable entirely on a local MacBook (no cloud data storage).  
**Primary modes**

| Mode | Trigger | Typical use |
|------|---------|-------------|
| **Background** | auto‑capture every *N* seconds (default 2 s) | Continuous journaling |
| **On‑demand** | hotkey `⌃⇧S`, CLI `/snap`, or MCP mutation | Low‑overhead single snapshot |

Key user questions supported:
* “What Docker command did I run yesterday?”  
* “Show calendar events I created at 3 pm.”  
* “List error pop‑ups in the last 30 minutes.”

---

## 2 · System Architecture

```mermaid
flowchart LR
  subgraph Desktop App
    S[PyAutoGUI<br>Screenshot]
    S --> O[Tesseract OCR<br>(resident)]
    S --> D[Diff detector<br>(SSIM)]
    D -->|scene change| C[CLIP ViT‑B/32<br>embedding]
    O --> E{conf < 0.8?}
    E -- yes --> V[GPT‑4o Vision<br>(via Martian)]
    V --> J[Event JSON]
    E -- no --> J
    C --> J
    J --> P[(Postgres)]
  end

  P -->|SQLAlchemy| M[EnrichMCP API<br>(FastAPI+FastMCP)]
  M -. ctx.ask_llm .-> R[Martian Router]

  subgraph Clients
    U1[CLI `ask`]
    U2[Electron / Tray UI]
    U1 & U2 --> M
  end
```

---

## 3 · Technology Matrix

| Concern | Choice | Justification |
|---------|--------|---------------|
| Language | Python 3.11 | FastAPI, tesserocr bindings |
| Database | Postgres 16 (Docker) | Reliable, pgvector extension |
| ORM | SQLAlchemy 2 | Async engine, native typing |
| OCR | Tesseract 5 (`tesserocr`) | ~150 ms @1080p |
| Vision fallback | GPT‑4o‑Vision via Martian | High accuracy, limited use |
| Scene embedding | CLIP ViT‑B/32 ONNX + Metal EP | 25 ms, 200 MB VRAM |
| LLM Q&A | Mixtral‑8x7B (default) | Cheap ~$0.002 / 1k |
| Router | Martian router docker image | Budget + fail‑over |
| Snapshot hotkey | `keyboard` lib (macOS accessibility) | Simple global keys |
| Diff metric | SSIM (`skimage.metrics`) | Fast & perceptually relevant |
| Vector search | pgvector `vector` type + KNN | Local, no extra infra |
| Packaging | Poetry 1.7 | Reproducible env |
| Testing | Pytest + HTTPX + Postgres test‑container | End‑to‑end coverage |

---

## 4 · Data Design

### 4.1 Tables

```sql
-- core event log
CREATE TABLE screen_events (
  id          BIGSERIAL PRIMARY KEY,
  ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
  window      TEXT,
  full_text   TEXT,
  ocr_conf    SMALLINT CHECK (ocr_conf BETWEEN 0 AND 100),
  clip_vec    VECTOR(512)  -- optional
);

-- derived commands
CREATE TABLE commands (
  id          BIGSERIAL PRIMARY KEY,
  ts          TIMESTAMPTZ NOT NULL,
  cmd         TEXT,
  exit_code   SMALLINT
);

-- calendar entries
CREATE TABLE calendar_entries (
  id          BIGSERIAL PRIMARY KEY,
  ts          TIMESTAMPTZ NOT NULL,
  title       TEXT,
  event_time  TIMESTAMPTZ,
  source_app  TEXT
);
```

### 4.2 Indexes

```
CREATE INDEX idx_screen_ts ON screen_events (ts DESC);
CREATE INDEX idx_screen_text_gin ON screen_events USING gin(to_tsvector('english', full_text));
CREATE INDEX ON commands (ts DESC);
CREATE INDEX ON calendar_entries (event_time);
-- vector index
CREATE INDEX idx_clip_vec ON screen_events USING hnsw (clip_vec);
```

---

## 5 · API Surface (MCP)

| Tool name | Args | Returns | Purpose |
|-----------|------|---------|---------|
| `capture_now()` | — | `ScreenEvent` | One off snapshot (on‑demand) |
| `find(pattern:str, since_min:int=60)` | — | `[ScreenEvent]` | Full‑text search |
| `search_semantic(query:str,k:int=5)` | — | `[ScreenEvent]` | KNN on `clip_vec` |
| `recent_errors(window_min:int=30)` | — | `[ScreenEvent]` | Regex filter on `contains_error` |
| `last_docker()` | — | `Command` | Last docker command |
| `calendar_between(start_ts,end_ts)` | — | `[CalendarEntry]` | Calendar recall |

All schema objects inherit Pydantic models, enabling autovalidation and discoverability via `explore_data_model()`.

---

## 6 · Capture & Processing

```python
# capture_daemon.py
loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)

tess = tesserocr.PyTessBaseAPI(psm=6)
clip_sess = ort.InferenceSession("clip.onnx",
            providers=["CoreMLExecutionProvider","CPUExecutionProvider"])
prev_img = None; last_clip_ts = 0

async def main():
    while True:
        raw = pyautogui.screenshot()
        img = cv2.resize(np.array(raw), (1280,720))
        tess.SetImage(img); text = tess.GetUTF8Text(); conf = tess.MeanTextConf()
        if conf < 80:
            text = call_gpt_vision(img)   # Martian decides cost
        now = time.time()
        changed = prev_img is None or ssim(prev_img,img,multichannel=True)<0.95
        clip_vec=None
        if changed and now - last_clip_ts > 15:
            clip_vec = run_clip(img); last_clip_ts=now
        prev_img = img
        await insert_event(text, conf, clip_vec)
        await asyncio.sleep(2)

loop.run_until_complete(main())
```

Regex worker parses `cmd_line` and `cal_event` patterns asynchronously, inserting rows into derived tables.

---

## 7 · Operational Concerns

| Concern | Implementation |
|---------|----------------|
| **CPU/RAM** | Resident OCR threads on one efficiency core; < 5 GB RAM total. |
| **Disk** | JSON text only: ≈ 30 MB/hr → rotate older than 30 days. |
| **Cost** | Martian `daily_usd: 1.00`; log spend per call. |
| **Privacy** | Option to AES‑encrypt DB, hotkey pause, auto‑skip “Password” windows. |
| **Metrics** | Prom‑style counters: frames/sec, OCR ms, vision tokens, spend. |
| **Logging** | Structlog to file; vision fallback events flagged. |
| **Hot reload** | Watchdog restarts daemon when file changes (dev convenience). |

---

## 8 · Local Development

```bash
brew services start postgresql@16    # or docker compose up db
poetry install                       # installs deps
docker run -p5333:5333 ghcr.io/withmartian/router:latest
export OPENAI_API_BASE=http://localhost:5333/v1
export OPENAI_API_KEY=localrouter
poetry run python capture_daemon.py &
poetry run uvicorn screen_api:app --reload --port 5003
```

*Make sure Postgres has `CREATE EXTENSION pgvector;`.*

---

## 9 · Test Plan

| Layer | Test | Tool |
|-------|------|------|
| OCR | assert OCR conf > 80 on sample PNG | Pytest fixtures |
| Vision fallback | mock low conf; assert GPT route | pytest‑mock |
| DB | alembic migrations up/down | pytest‑docker |
| API | `httpx` call `find()` returns expected row | pytest‑asyncio |
| End‑to‑end | run capture_loop 10 s, query last_docker | tox matrix |

CI on GitHub Actions with macos‑13 runner, Postgres service, and matrix {Python 3.11}.

---

## 10 · 48‑Hour Hackathon Milestones

| Hour | Deliverable |
|------|-------------|
| 0‑4 | Postgres schema, capture one frame, manual insert. |
| 4‑8 | Resident OCR loop + SQLAlchemy insert. |
| 8‑12 | Diff + CLIP, vector column, KNN query. |
| 12‑18 | EnrichMCP endpoints (`find`, `capture_now`). |
| 18‑24 | Martian router, GPT‑4o Vision fallback integrated. |
| 24‑30 | CLI `ask` + hotkey `/snap`. |
| 30‑36 | Tray UI (Electron or tkinter), cost meter. |
| 36‑42 | Tests, logging, README. |
| 42‑48 | Record demo video, polish, submit. |

---

## 11 · Stretch Enhancements

* **Whisper local** for voice queries.  
* **Llama 3‑Instruct‑8B GGUF** for offline Q&A when no internet.  
* **Timeline web app** with React + Supabase auth.  
* **Federated sync**: encrypted WAL‑ship to phone for on‑the‑go recall.  
* **Plugin SDK** to parse additional event types (JIRA ticket IDs, error traces).

---

### Final Note
This blueprint removes Featureform to slim the footprint—**Postgres + EnrichMCP is all you need** for a hackathon‑ready, local‑first screen memory assistant. Martian ensures LLM costs stay pocket‑change while delivering GPT‑4o Vision quality when needed.

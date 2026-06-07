# Pinterest Agent — Phase 1

A production-ready, LangGraph-powered Pinterest scraper. This is **Phase 1** of a
multi-agent pipeline that will eventually include Image Analysis, Ranking, Quality
Evaluation, and Instagram Publishing agents.

## Architecture

```
START → PinterestAgent → END
```

Future phases extend the graph:
```
START → PinterestAgent → ImageAnalysisAgent → RankingAgent
      → QualityEvaluationAgent → InstagramPublisherAgent → END
```

### Project structure

```
pinterest-agent/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py         # Abstract base for all agents
│   ├── models.py             # Pydantic data models (PinData, PinterestScrapingResult)
│   ├── pinterest_agent.py    # Phase 1: Playwright scraper + LangGraph node
│   └── state.py              # LangGraph AgentState TypedDict
├── config/
│   ├── __init__.py
│   └── settings.py           # Pydantic-settings config (loaded from .env)
├── utils/
│   ├── __init__.py
│   ├── helpers.py            # File I/O, sanitization, formatting
│   ├── logger.py             # structlog setup
│   └── retry.py              # tenacity retry decorators
├── data/                     # Scraped results (gitignored, .gitkeep tracks dir)
├── logs/                     # Log files (gitignored, .gitkeep tracks dir)
├── graph.py                  # LangGraph StateGraph definition
├── main.py                   # CLI entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Clone / create the project directory

```bash
cd C:\Users\<you>
# project already created if you're reading this
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright browsers

```bash
playwright install chromium
```

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY for future phases
```

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Phase 2+ LLM agents |
| `PINTEREST_EMAIL` | — | Optional: improves scraping quality |
| `PINTEREST_PASSWORD` | — | Optional: used with PINTEREST_EMAIL |
| `MAX_PINS_PER_QUERY` | `50` | Number of pins to scrape per run |
| `HEADLESS_BROWSER` | `true` | Set `false` to watch the browser |
| `LOG_FORMAT` | `json` | `json` or `console` |

---

## Usage

```bash
# Basic
python main.py "home decor ideas"

# Custom pin count
python main.py "minimalist living room" --max-pins 100

# Save to specific file
python main.py "boho bedroom" --output-file data/boho_bedroom.json

# Console-friendly logging + visible browser
HEADLESS_BROWSER=false python main.py "kitchen design" --log-level DEBUG

# Skip saving results
python main.py "art deco office" --no-save
```

### Output

Results are written to `data/results_<query>_<timestamp>.json`:

```json
{
  "query": "home decor ideas",
  "max_pins_requested": 50,
  "result": {
    "query": "home decor ideas",
    "pins": [
      {
        "pin_id": "123456789",
        "title": "Cozy living room inspo",
        "image_url": "https://i.pinimg.com/736x/...",
        "original_image_url": "https://i.pinimg.com/originals/...",
        "pin_url": "https://www.pinterest.com/pin/123456789/",
        "scraped_at": "2026-06-06T12:00:00+00:00"
      }
    ],
    "total_scraped": 50,
    "total_attempted": 50,
    "scraping_duration_seconds": 42.3
  }
}
```

---

## Adding a New Agent (Phase 2+)

1. **Create** `agents/<name>_agent.py` extending `BaseAgent`
2. **Add** output fields to `AgentState` in `agents/state.py`
3. **Add** a Pydantic result model to `agents/models.py`
4. **Add** a node function in `graph.py` and uncomment the relevant `add_node` / `add_edge` calls

All TODO comments in `graph.py`, `agents/state.py`, and `agents/models.py` mark the
exact extension points.

---

## Tech Stack

| Library | Role |
|---|---|
| `langgraph` | Agent pipeline orchestration |
| `langchain-anthropic` | Anthropic LLM integration (Phase 2+) |
| `playwright` | Async browser automation |
| `pydantic` + `pydantic-settings` | Data models + config |
| `structlog` | Structured JSON logging |
| `tenacity` | Retry with exponential backoff |
| `python-dotenv` | `.env` file loading |

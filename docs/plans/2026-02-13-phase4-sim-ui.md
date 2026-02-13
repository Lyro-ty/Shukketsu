# Phase 4: Sim Web UI — Lightweight Sim Page + Chat Integration

## Overview

Hybrid approach: a dedicated `/sim` page for character setup and one-click sim runs, plus full chat integration so the Analyst agent can iterate on results conversationally.

## Architecture

```
web/routers/sim.py              # FastAPI router
web/templates/sim/
├── index.html                  # main sim page layout
├── partials/
│   ├── gear_table.html         # gear overview (HTMX fragment)
│   ├── slot_dropdown.html      # item search dropdown per slot
│   ├── results_panel.html      # DPS results + charts
│   ├── stat_weights.html       # stat weight table
│   ├── ability_breakdown.html  # pie chart + table
│   ├── buff_toggles.html       # preset selector + individual toggles
│   └── import_form.html        # character import textarea
web/static/js/
├── sim-charts.js               # Chart.js renderers
└── sim-import.js               # import paste handling
```

## Routes

### Page Routes

```python
@router.get("/sim/")
async def sim_page(request: Request):
    """Render the sim page with empty state."""

@router.post("/sim/import")
async def sim_import(request: Request, import_string: str = Form()):
    """Parse import, return gear_table partial with populated data."""

@router.post("/sim/run")
async def sim_run(request: Request, config: SimConfig):
    """Run sim, return results_panel partial."""

@router.post("/sim/swap/{slot}")
async def sim_swap(request: Request, slot: str, item_id: int = Form()):
    """Swap item in slot, return updated gear_table row."""
```

### API Routes (JSON, for agent use)

```python
@router.post("/api/sim/run")
async def api_sim_run(config: SimRunInput) -> SimResult:
    """JSON API for Analyst agent tool."""

@router.post("/api/sim/compare")
async def api_sim_compare(input: SimCompareInput) -> CompareResult:
    """JSON API for Analyst agent tool."""

@router.post("/api/sim/optimize")
async def api_sim_optimize(input: SimOptimizeInput) -> OptimizeResult:
    """JSON API for Analyst agent tool."""
```

### Data Routes

```python
@router.get("/api/sim/items/{slot}")
async def items_for_slot(slot: str, q: str = "", phase: int = 5):
    """Search items for a gear slot. Used by swap dropdowns."""

@router.get("/api/sim/presets")
async def list_presets():
    """Available raid buff presets."""
```

## Page Layout

### Section 1: Import Bar

Full-width textarea for pasting import strings. Auto-detects format on submit. Success → populates gear table + talent string + spec selector.

```html
<form hx-post="/sim/import" hx-target="#gear-section" hx-swap="innerHTML">
  <textarea name="import_string" placeholder="Paste /simc, 70u, or WoWSims export..."></textarea>
  <button type="submit">Import</button>
</form>
```

### Section 2: Character Header

Spec dropdown, race dropdown, talent string input. Auto-filled from import, editable.

### Section 3: Gear Table

17 rows (one per slot). Each row: slot name, item name + ilvl, [Swap] button that opens a search dropdown (HTMX powered, searches `items_for_slot` API).

The swap dropdown is a `<select>` with `hx-get="/api/sim/items/{slot}?q=..."` live search. Selecting an item fires `hx-post="/sim/swap/{slot}"` to update the row.

### Section 4: Buff Configuration

Preset selector (Full 25-Man, Karazhan 10-Man, Solo, Custom). Below it, individual buff checkboxes auto-populated from the preset. Toggling any checkbox switches preset to "Custom".

Boss config: armor input, debuff checkboxes (Sunder, FF, CoR), fight length, iterations.

### Section 5: Run Button + Results

"Run Simulation" button triggers `hx-post="/sim/run"` with the full config as form data. While running, shows a progress spinner with estimated time.

Results panel (returned as HTMX partial):

**DPS Summary**: Mean +/- StdDev, Median, Min, Max

**Ability Breakdown**: Pie chart (Chart.js) + sortable table with columns: Name, DPS, %, Casts, Hit%, Crit%, Miss%

**Stat Weights**: Bar chart + table: Stat, EP Value, DPS/point, Capped flag. Hit/expertise highlighted if below cap.

**DPS Distribution**: Histogram (Chart.js) showing per-iteration DPS spread.

**Proc Uptimes**: Table: Proc name, Source, Uptime%, Procs/fight

**Resource Stats**: Energy/sec, waste, CP/sec, GCD utilization

## Chat Integration

### Sim Context in Chat

When a user has an active sim profile (from the `/sim` page or a previous chat sim), the chat handler passes it as context to the Analyst agent. The agent can reference and modify it:

```
User: "swap in DST for my current setup"
Agent: [reads current sim profile from session]
       [calls sim_compare with current trinket vs DST]
       "Swapping Bloodlust Brooch for Dragonspine Trophy gains 47 DPS (+2.5%)..."
```

### Results in Chat

Sim results rendered inline using structured chat messages:
- DPS summary as a formatted block
- Ability breakdown as a markdown table
- Charts rendered client-side from JSON data in the message

### Session State

Current sim profile stored in the WebSocket session. Updated when:
- User imports a character on `/sim` page
- User swaps an item via chat ("use DST instead")
- Agent runs a sim (latest config becomes current profile)

## Technology Choices

- **HTMX**: All sim page interactions are HTMX partials (consistent with existing wiki/chat pages)
- **Chart.js**: Lightweight, no build step, renders from JSON data. Three chart types: pie (breakdown), bar (stat weights), histogram (DPS distribution)
- **WebSocket streaming**: Long sim runs (stat weights = ~45s) stream progress updates via existing WebSocket infrastructure. Progress: "Running iteration 5000/10000..."

## Dependencies

No new dependencies. Uses existing: FastAPI, Jinja2, HTMX, Chart.js (add to static assets).

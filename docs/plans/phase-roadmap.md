# Phase Roadmap

> Lightweight outline. Detailed specs will be written at the start of each
> phase, once we know what we learned from the previous one.

## Phase 2: Multi-Agent + Knowledge Building

**Goal**: Specialist agents cooperate on complex tasks. Wiki articles written
and verified. Agentic RAG produces high-quality retrievals.

**Key capabilities**:
- Orchestrator decomposes complex queries into sub-tasks
- Researcher, Analyst, Writer, Editor agents with distinct roles
- Async message bus for inter-agent communication
- Agentic RAG: query decomposition, iterative retrieval, self-RAG, corrective RAG
- Warcraft Logs API integration (OAuth + GraphQL)
- Blizzard Battle.net API integration (OAuth + REST)
- WoWSims and CMaNGOS data importers
- Source trust scoring + conflict resolution
- Content freshness checking
- WoW-specific chunker (tooltip/talent table protection)
- Wiki browser UI with confidence badges
- Search UI
- WoW-style item tooltips
- Automated SQLite backups

**Gate**: RAG faithfulness > 0.8. Agent trajectory precision > 0.7. Domain
accuracy > 70%. Wiki has articles covering all three specs.

---

## Phase 3: Simulation Engine

**Goal**: DPS simulation produces results within 2% of WoWSims. Agent uses
sim to verify claims and settle disputes.

**Key capabilities**:
- TBC rogue combat mechanics (hit table, crit, armor, dual wield)
- Rogue abilities, talents, items, buffs modeled
- Combat, Assassination, Subtlety rotation logic
- Discrete event simulation loop
- Sim runner tool integrated with analyst agent
- Combat log parsing and analysis
- Log upload UI with improvement suggestions
- Validation against WoWSims + real Warcraft Logs data

**Gate**: P1 BiS combat swords on Patchwerk-style fight within 2% of WoWSims.

---

## Phase 4: Evaluation + Observability Polish

**Goal**: Automated eval pipeline tracks quality over time. Human feedback
collected and incorporated.

**Key capabilities**:
- Ragas RAG quality metrics (faithfulness, relevancy, precision, recall)
- Agent trajectory evaluation
- Domain accuracy evaluation (50+ curated questions)
- Eval dashboard UI with historical charts
- Agent trace viewer UI
- Human feedback mechanism (thumbs up/down)
- Fine-tuning dataset curation (Q&A pairs, tool-calling examples)

**Gate**: Full eval suite runs automatically. Results tracked over time.

---

## Phase 5: Polish + Growth (Ongoing)

**Key capabilities**:
- Interactive talent tree UI
- Sim builder visual interface
- DPS breakdown charts (Chart.js)
- Gear comparison tool
- PvP/Arena section build-out
- Proactive research (agent discovers gaps and fills them)
- Static site export
- LoRA fine-tuning execution
- Phase 2-5 TBC content updates as raids release

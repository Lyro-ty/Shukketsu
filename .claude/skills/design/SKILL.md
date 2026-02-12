---
name: design
description: Start a new design session for the next phase/step. Analyzes project state, leads collaborative brainstorming, and produces a validated design document.
user-invocable: true
---

# Collaborative Design Session

Lead a design session for the next implementation step. Analyzes where the project stands,
brainstorms the design collaboratively, and produces a validated design document.

Announce: "Using the **design** skill to start a design session."

## Workflow

### 1. Analyze Project State

Before asking any questions, automatically gather context:

- Read `CLAUDE.md` for current phase, step status, and architecture
- Read the active phase plan (e.g., `docs/plans/phase-2-multi-agent-rag.md`) to identify the next step
- Read the previous step's design doc (if one exists) for continuity
- Check recent git commits to understand what was just built
- Note the current test count from CLAUDE.md

Present a brief summary:

> "We're on Phase X, Step Y. The previous step built [components]. Next up is Step Z:
> [description from phase plan]. Current state: N tests passing."

Ask the user to confirm this is what they want to design, or if they have something else in mind.

### 2. Collaborative Design

Explore the design space through natural conversation. Follow these principles:

- **One question at a time** — never stack multiple questions in one message
- **Multiple choice preferred** — easier to answer; open-ended only when choices aren't clear
- **Lead with your recommendation** — propose 2-3 approaches, explain trade-offs, say which you'd pick and why
- **YAGNI actively** — when a feature comes up, ask: "Is this needed now, or is that a later phase concern?"
- **Flow into design naturally** — as understanding builds, start presenting design sections (200-300 words each)
- **Validate incrementally** — after each design section, check: "Does this look right so far?"
- **Surface risks early** — flag technical unknowns, external dependencies, or tricky edge cases as they arise
- **Be ready to backtrack** — if something doesn't fit, revise earlier sections openly

There is no hard boundary between brainstorming and writing. Questions flow into design sections
as understanding develops. You might design one component while still brainstorming another.

### 3. Produce Design Document

Once the design is validated through conversation, write the full document to:
`docs/plans/YYYY-MM-DD-<topic>-design.md`

The document MUST include these sections in this order:

```
# [Feature/Step Name]

> Design document for Phase X, Step Y. Validated through brainstorming session YYYY-MM-DD.

## Overview
High-level summary (2-4 paragraphs). What this adds, why it matters, how it fits.

## Design Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
Key architectural choices made during brainstorming, with reasoning.

## File Changes
### New Files
| File | Purpose |
### Modified Files
| File | Change |
### Unchanged Files
List of related files that won't be touched (prevents scope creep during implementation).

## Component N: [Name]
Detailed design per component. Each should include:
- Class/function signatures (not full implementations)
- Constructor dependencies
- Execute/processing flow (numbered steps)
- Output format with concrete examples
- Error handling strategy
- Edge cases and defensive validation

## Risks and Unknowns
| Risk | Impact | Mitigation |
|------|--------|------------|
Technical risks, unknowns needing investigation, external dependencies that could break.

## What This Does NOT Include
Explicit YAGNI declarations — features consciously deferred to later phases/steps.
Every design session MUST produce this section.

## Dependencies
What must exist before this can be implemented:
- Completed prior steps and their outputs
- Existing modules/functions this builds on
- External services or data required

## Test Plan
Specific test file breakdown with named test cases and estimated counts.
Starting count: N tests -> Estimated after this step: ~M tests.
```

### 4. Commit and Update

After writing the design document:

1. **Commit** the design doc to git with message: `docs: add Phase X Step Y design doc`
2. **Update `CLAUDE.md`**:
   - Add the new design doc to the plans table
   - Update the step status line (e.g., mark as "design complete" or similar)
3. **Handoff**: Ask "Ready to create the implementation plan?" and hand off to the
   `writing-plans` skill if the user wants to continue.

## Key Principles

- **Context first** — always analyze project state before asking questions
- **Continuity** — reference the previous step's design to avoid contradictions or re-invention
- **Scope discipline** — every design session MUST produce a "What This Does NOT Include" section
- **Concrete over abstract** — include examples, sample queries, output format illustrations
- **Design != implementation** — signatures and interface contracts yes, complete code no
- **Risks surfaced early** — unknowns identified now save hours during implementation
- **One question at a time** — never overwhelm; break complex topics into multiple questions

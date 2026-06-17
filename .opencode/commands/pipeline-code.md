---
description: Code Atlas — build a Source Insight-style symbol map of the codebase (no LLM/API key needed)
agent: build
---
Build the Code Atlas symbol map for this project, then give me a tour.

Run the analyzer below. It is deterministic static analysis — no API key, no network.
Pass a target folder as an argument, or leave empty to analyze the current project.
(If `python` is not found, the fallback tries `python3`.)

!`python tools/pipeline_code.py $ARGUMENTS || python3 tools/pipeline_code.py $ARGUMENTS`

Now read `codemap/codemap.md` and summarize for me:
- the overall structure and main modules,
- the most-referenced symbols (the project's core abstractions),
- the key internal dependencies and any notable class hierarchies or call paths.

The full machine-readable database is in `codemap/symbols.json`, and an interactive
graph is in `codemap/codemap.html`.

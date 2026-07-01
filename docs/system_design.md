# System design

This is the "how it fits together and why" document. For how to run or deploy it,
see the README and `deployment.md`.

## The shape of it

RapidHire is a pipeline with four stages. Each stage is a CrewAI agent; the stages
run in sequence and hand typed objects to each other. The whole thing is driven from
`run()` in `crew.py`, which scores a batch of resumes and ranks them.

```
resume file
   │  extract_text()            (tools/parsing.py, runs before the crew)
   ▼
resume text ──► Intake agent ──► CandidateProfile
                                     │
job description ─────────────────────┼──► Matching agent ──► MatchResult
                                     │         (vectorstore.py: embed + retrieve)
                                     ▼
                              Screening agent ──► ScoreCard        ◄── the RAG step
                                     │            (tools/scoring.py: LangChain + retrieval)
                                     ▼
                              Summary agent  ──► rationale + advisory call
                                     │
                                     ▼
                    reconcile in Python: recompute overall from
                    categories, re-derive interview/hold/reject
                                     │
                     ┌───────────────┴───────────────┐
                     │  borderline (hold) + panel on? │──► AutoGen panel (panel.py)
                     └───────────────────────────────┘
                                     │
                                     ▼
                            ranked list of ScoreCards ──► Streamlit / CLI
```

The state passed between stages is always a Pydantic model from `models.py`, never a
loose dict. A malformed hand-off fails at the boundary instead of surfacing as a
`KeyError` two stages downstream.

## Where each framework lives, and why only there

The design rule is one framework per seam. Blending them is what makes multi-framework
projects impossible to reason about later.

| Concern | Lives in | Why there |
|---|---|---|
| Orchestration | CrewAI (`crew.py`, `agents/`) | Sequential multi-agent flow is exactly what it's for. |
| LLM plumbing for scoring | LangChain (`tools/scoring.py`) | Prompt template + structured output + LCEL, in the one spot that needs it. |
| Embeddings + vector search | Plain Python (`vectorstore.py`) | No framework earns its complexity here; it's sentence-transformers and ChromaDB. |
| Borderline debate | AutoGen (`panel.py`) | Group-chat deliberation is its strength; kept optional and isolated. |

Because the layers are separated, you can replace one without touching the rest.
Swapping AutoGen for LangGraph, or ChromaDB for another store, is a change to a single
file.

## The four agents

**Intake** (`agents/intake.py`). Input: resume text. Output: `CandidateProfile`.
The extraction is done with a raw OpenAI SDK call using a forced tool call against a
hand-written strict JSON schema (`strict: True`), which is the most reliable way to
get well-formed structured output. Only redacted text is ever logged.

**Matching** (`agents/matching.py`, `tools/matching.py`). Input: the profile and the
job description. Output: `MatchResult`, a role-similarity cosine plus the supporting
evidence chunks. No LLM is involved; this is pure vector math. This is the fast path.

**Screening** (`agents/screening.py`, `tools/scoring.py`). Input: profile, match,
job description. Output: `ScoreCard`. This is the RAG step and the *only* place
retrieval feeds the model: role criteria are pulled from ChromaDB, filtered by a
cosine floor, and dropped into the prompt. The model scores each rubric category and
justifies it; the weighted overall is computed afterward in Python.

**Summary** (`agents/orchestrator.py`). Input: the scorecard and the match. Output: a
short rationale plus the model's own read of the call. That read is advisory; see
below.

## The one deliberate redundancy

Combining CrewAI, LangChain, and a structured-output LLM produces a real piece of
redundancy, and it's worth naming rather than hiding.

The Intake and Screening steps are CrewAI *agents* whose actual work is done by a
*tool* that itself calls an LLM. So there's an agent turn (the model deciding to call
the tool) wrapped around a tool that makes its own model call. The agent turn adds
latency and tokens without adding much: the tool is deterministic about what it does.

It's structured this way so all three frameworks are genuinely exercised behind their
seams, which is the point of the exercise. In production you'd collapse it. Two clean
options:

- **CrewAI Flow** instead of a Crew. A Flow lets you call the scoring function
  directly as a step, dropping the extra agent turn while keeping the orchestration.
  This is the leaner path and the one to reach for if the redundancy bothers you.
- **Skip the agent entirely** for the deterministic stages and just call the tool
  functions (`extract_profile`, `score_against_rubric`) in sequence. You lose the
  "agent" framing but keep every behavior.

The current Crew is chosen for clarity of demonstration, not for minimal token count.

## Determinism vs. judgment

The split between what the model decides and what the code decides is intentional:

- **Code does**: parsing, embedding, similarity, the weighted average, the
  interview/hold/reject thresholding (`recommend()` in `models.py`), and ranking.
- **The model does**: extract structure from a resume, and score each soft rubric
  category with a justification.

After the crew finishes, `analyze_candidate()` reconciles: it takes the authoritative
category scores from the scoring task, recomputes the overall (`weighted_check`), and
re-derives the label. The summary agent's own recommendation is discarded in favor of
the recomputed one. A language model producing a weighted average is a bug waiting to
happen, so it never gets the chance.

## Retrieval boundary

There are two collections in the store (`vectorstore.py`): `hiring_criteria` (the
rubric context, seeded from `data/criteria/` on first run) and `candidate_pool`
(optional, for nearest-peer context). Retrieval happens in exactly one place: the
scoring tool queries `hiring_criteria` once per candidate. The cosine relevance floor
(`relevance_threshold`, default 0.3) drops weak matches so the prompt isn't padded
with irrelevant context. Matching also reads the store for its similarity signal, but
that's vector math feeding a number, not retrieval feeding the model.

## The panel

When enabled, borderline candidates (overall score in the `[borderline_low,
borderline_high)` band) get a second opinion. Three AutoGen `AssistantAgent`s (a CEO,
a CTO, and an HR lead) deliberate in a `RoundRobinGroupChat` until one posts a line
in the form `CONSENSUS: interview` (or hold/reject), or a hard message cap stops them.
The consensus, if reached, overrides the recommendation for that candidate.

AutoGen is async; the rest of the app is sync. `deliberate()` bridges the two: if
no event loop is running it uses `asyncio.run()`, and if one already owns the thread
it runs the coroutine on its own loop in a worker thread. The OpenAI client is always
closed in a `finally`.

## Model, latency, and cost

- Two models by default: `gpt-4o` for the careful work (scoring, summary) and
  `gpt-4o-mini` for the cheap work (extraction, the lightweight agent turns). Both are
  config values; swapping either is one line.
- The matching layer is the only thing that's genuinely fast. A warm MiniLM encode
  plus a ChromaDB ANN lookup is sub-100ms. Everything with "LLM" in it is seconds.
- Per candidate you pay for: one extraction call, the agent turns, one scoring call,
  one summary call, and (if borderline and enabled) several panel messages. The panel
  is the most expensive path, which is why it's gated to the candidates where a
  second opinion actually changes something.

Levers if cost or latency matters: point `reasoning_model` at a smaller model, lower
`top_k` and raise `relevance_threshold` to shrink prompts, keep the panel off or
narrow the borderline band, and (in production) collapse the redundant agent turns per
the Flow note above.

## Honest limitations

The fairness, legal, and "not run end-to-end in CI" caveats are real and are spelled
out in the README's Limitations section. The short version: this is a technical
demonstration of an architecture, not a system that's safe to make hiring decisions
with. Keep a human in the loop.

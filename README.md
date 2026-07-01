# RapidHire

Paste in a job description and a stack of resumes, get back a ranked, explainable
shortlist. Each candidate comes with per-category scores, a short rationale, and an
interview / hold / reject call you can trace back to the rubric.

It's a demo of a multi-agent screening pipeline built on three frameworks that each
do one job, plus an optional fourth for the close calls. It is **not** a production
hiring system. Read the [Limitations](#limitations) before you get ideas.

## What it does

1. **Intake** turns a raw resume (PDF, DOCX, or text) into a structured profile.
2. **Matching** embeds the resume and the role, pulls the nearest hiring-criteria
   snippets out of a vector store, and computes a similarity signal.
3. **Screening** scores the candidate against a weighted rubric, using the retrieved
   criteria as context. This is the one place retrieval feeds the model.
4. **Summary** writes the recruiter-facing rationale and, for borderline scores,
   optionally convenes a small debate before settling the call.

The overall score and the interview/hold/reject label are computed in plain Python
from the category scores. The language model scores categories and writes prose. It
never does the arithmetic or picks the final label, and that line is deliberate.

## Why it's built this way

Each framework sits behind one seam instead of being blended together:

- **CrewAI** is the orchestrator: four agents in a sequential crew (`src/rapidhire/crew.py`).
- **LangChain** is the plumbing inside the scoring step, a prompt template bound to a
  structured-output schema via LCEL (`src/rapidhire/tools/scoring.py`).
- **The embedding and vector layer is plain Python**: sentence-transformers and
  ChromaDB behind a small class (`src/rapidhire/vectorstore.py`), exposed to the
  agents as tools.
- **AutoGen** is an isolated, optional module, a three-person panel that only runs
  for borderline candidates and only when you turn it on (`src/rapidhire/panel.py`).

Keeping them separated means you can rip out any one layer without unpicking the
others. `docs/system_design.md` goes into the tradeoffs, including the one real piece
of redundancy this stack forces and how a CrewAI Flow would remove it.

## Quickstart

You'll need Python 3.10 to 3.12, an OpenAI API key, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install everything (app + dev tools) into a managed virtualenv.
uv sync

# 2. Set your key.
cp .env.example .env
# edit .env and set RAPIDHIRE_OPENAI_API_KEY, or just: export OPENAI_API_KEY=sk-...

# 3a. Run the app.
uv run streamlit run src/rapidhire/app.py

# 3b. ...or use the CLI on the bundled samples.
uv run rapidhire data/sample_jobs/backend_engineer.md data/sample_resumes/*.txt
```

On first run the criteria in `data/criteria/` are embedded into a local ChromaDB
store under `data/chroma/`. That directory is disposable. Delete it and it rebuilds.

The AutoGen panel is **off by default**. The whole pipeline works without it. Turn it
on with `RAPIDHIRE_ENABLE_PANEL=true` (or the sidebar toggle) to have borderline
candidates debated before the call is made.

### Docker

```bash
docker compose up --build
```

This brings up the app plus a standalone ChromaDB server. The app talks to it over
HTTP, since the single-file client doesn't allow concurrent writers. Pass your key
through the environment; it isn't baked into the image. Heads up: the image is large,
because sentence-transformers pulls in PyTorch.

## Project structure

```
src/rapidhire/
  config.py         Settings (env-driven); nothing else reads os.environ
  logging.py        logging setup + a redact() helper so PII stays out of logs
  models.py         the typed objects that move between stages
  vectorstore.py    embeddings + ChromaDB; the whole vector layer
  crew.py           the four-agent crew and the run() entry point
  panel.py          optional AutoGen debate for borderline scores
  cli.py            rapidhire job.md resume.pdf ...
  app.py            Streamlit UI
  tools/            parsing, matching, scoring (exposed to agents as CrewAI tools)
  agents/           one builder per agent: intake, matching, screening, orchestrator
data/
  sample_resumes/   four candidates, mixed formats and quality
  sample_jobs/      example job descriptions
  criteria/         rubric context, seeded into the vector store
tests/              run without an API key; the LLM is mocked
docs/               system design and deployment notes
```

## Testing

```bash
uv run pytest
```

The suite runs with no API key. Every test that would otherwise hit the model injects
a fake. The vector store, by contrast, is tested for real against local embeddings
(those tests skip if the heavy dependencies aren't installed). If you want to lint the
way CI does:

```bash
uv run ruff check .
uv run ruff format --check .
```

## Configuration

Everything tunable is in `config.py` and settable via `RAPIDHIRE_`-prefixed env vars
or a `.env` file. The knobs you're most likely to touch: `RAPIDHIRE_REASONING_MODEL`
(scoring model, defaults to `gpt-4o`), `RAPIDHIRE_RELEVANCE_THRESHOLD` (cosine floor
for dropping weak retrieved chunks), the `RAPIDHIRE_BORDERLINE_LOW`/`HIGH` band that
defines a "hold", and `RAPIDHIRE_ENABLE_PANEL`. See `.env.example` for the full list.

Swapping the model is a one-line config change. The code doesn't hardcode `gpt-4o`
anywhere. It does assume a chat model with tool/function-calling support, which is how
structured extraction and scoring stay reliable.

## Limitations

Read this part.

**This is not a compliance-ready hiring tool.** Automated candidate screening is a
legally regulated, high-stakes activity. The EU AI Act treats employment-related AI as
high-risk, and several US jurisdictions (New York City's Local Law 144, for one)
require bias audits and disclosures for automated employment decision tools. RapidHire
does none of that. Treat its output as a drafting aid for a human who makes the actual
decision, and talk to counsel before anything resembling real use.

**Fairness is not solved here.** An LLM scoring resumes can absorb and amplify bias
from its training data and from the criteria you feed it. The panel's HR persona is
prompted to watch for unfounded reasoning, but a prompt is not a safeguard. There is no
bias measurement, no demographic auditing, and no guarantee of consistency across runs.
Keep a human in the loop and don't outsource the judgment.

**The model is the bottleneck, not the vector search.** The matching layer is fast: a
warm MiniLM encode plus an approximate-nearest-neighbor lookup is the "sub-100ms" path.
A full candidate analysis is dominated by LLM latency and runs in seconds per resume.
Scoring a large batch is neither instant nor free.

**Embedding caveats.** MiniLM truncates inputs past roughly 256 tokens, so long resumes
are chunked and pooled rather than embedded whole. A ChromaDB collection's
dimensionality is fixed on first write, so don't switch embedding models against an
existing store without clearing it.

**AutoGen is in flux.** The classic AutoGen line is in maintenance mode. Its successor
(the Microsoft Agent Framework) reached GA in April 2026, and a community fork (AG2)
also exists. This project uses `autogen-agentchat` 0.7.x to demonstrate the panel
pattern, but for a greenfield build today you'd reach for the Agent Framework or
LangGraph. The panel is isolated precisely so that swap is easy.

**Not run end-to-end in CI.** The tests exercise the plumbing with the model mocked. A
real, full run needs an API key and the heavy dependency stack installed locally.

## License

MIT, see [LICENSE](LICENSE).

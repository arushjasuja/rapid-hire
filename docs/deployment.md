# Deployment

RapidHire is a Streamlit app with a vector store and a dependency on the OpenAI API.
That shapes every deployment choice: you need somewhere to run a long-lived Python
process, somewhere durable for the Chroma data if you want it to survive restarts, and
a way to inject the API key as a secret.

Below are three targets, cheapest-effort first.

## 1. Streamlit Community Cloud

The path of least resistance for a demo.

- Push the repo to GitHub and point Streamlit Cloud at `src/rapidhire/app.py`.
- Set the API key under the app's **Secrets** (Streamlit exposes secrets as env vars):

  ```toml
  RAPIDHIRE_OPENAI_API_KEY = "sk-..."
  ```
- Leave Chroma in its default `persistent` mode. The catch: Community Cloud storage is
  ephemeral, so the `data/chroma/` store is rebuilt from `data/criteria/` on each cold
  start. That's fine here (seeding is quick and idempotent), but don't rely on it to
  persist a candidate pool between restarts.

Good for showing the thing off. Not where you'd run anything real.

## 2. Docker (single host: Fly.io, Render, a VM)

The repo ships a `Dockerfile` and a `docker-compose.yml`. Compose is the intended way
to run it, because it also stands up a proper ChromaDB server:

```bash
export RAPIDHIRE_OPENAI_API_KEY=sk-...
docker compose up --build
```

What compose gives you that the bare Dockerfile doesn't:

- A standalone **ChromaDB server** container. The app is configured (via
  `RAPIDHIRE_CHROMA_MODE=http`) to talk to it over HTTP. This matters because the
  single-file persistent client can't handle concurrent writers, and the moment you might
  run more than one app process, you want the server.
- A **named volume** (`chroma-data`) so the vector store survives container restarts.
- A healthcheck gate so the app waits for Chroma to be ready before starting.

Deploying the image to Fly.io or Render: build and push it, set the key as a secret in
the platform, and attach a persistent volume for Chroma (or run Chroma as its own
service and point `RAPIDHIRE_CHROMA_HOST` at it). Expose port 8501. Streamlit's health
endpoint is `/_stcore/health` if the platform wants a health path.

Be aware the image is large, because sentence-transformers pulls in PyTorch. First build and
first cold start are slow; budget for it.

## 3. Managed containers (Cloud Run, ECS/Fargate)

Workable, with two things to plan for.

- **Statefulness.** Streamlit holds session state in memory and these platforms scale
  by spinning up interchangeable instances. Pin the service to a single instance, or
  put a session-affinity load balancer in front, or accept that a user's in-progress
  results don't follow them across instances. For a screening tool used by a handful of
  recruiters, a single instance is usually fine.
- **The vector store.** Don't use the local persistent client across autoscaled
  instances. Run ChromaDB as its own always-on service and set the app to `http` mode
  pointing at it. On Cloud Run specifically, note that the local filesystem is
  ephemeral, which is another reason to externalize Chroma.

Secrets go through the platform's secret manager (Google Secret Manager, AWS Secrets
Manager) injected as the `RAPIDHIRE_OPENAI_API_KEY` env var. Cloud Run wants the
container listening on `$PORT`; map it to 8501 or have Streamlit read it.

## Secrets, wherever you deploy

- The app reads `RAPIDHIRE_OPENAI_API_KEY`, falling back to a bare `OPENAI_API_KEY`.
- Never bake the key into the image. `docker-compose.yml` passes it through from the
  host environment; the Dockerfile doesn't copy `.env` (`.dockerignore` excludes it).
- `.env` is gitignored. `.env.example` is the template to copy from.

## A note on scale

None of this is built for high throughput. Scoring is dominated by sequential LLM
calls (see `system_design.md`), so the way you "scale" is by processing batches in the
background rather than by adding app instances. If you genuinely needed volume, the
move would be to pull screening out of the request path into a queue and worker setup,
which is a different application than this one.

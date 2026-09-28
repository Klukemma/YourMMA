# AI Team: a cloud multi-agent orchestrator

You chat with it from a browser or your phone. A **lead** AI splits your
request into steps and hands each one to a specialist AI. It checks their work
and keeps going, round after round, until the job is done. That can take hours
or days. You check the progress bar whenever you like.

```
you ──► lead (plans, assigns, reviews results, reports back)
          │
          ├──► researcher   web search, reading docs, notes with sources
          ├──► coder        writes and runs code, tests, git   ◄── checked by critic
          └──► critic       rigorous review of code, plans, research
```

- **Every agent is a different model.** You choose which model does which job
  in `agents.yaml`, or live from the **Team** panel in the app.
- **Everything is saved to disk.** A crash or redeploy resumes the task instead
  of starting over.
- **Spending caps.** Each task has its own cap ($5 by default, adjustable per
  task), and there's a daily cap across all tasks ($25). When the daily cap is
  hit, tasks pause and resume on their own after midnight UTC.
- **Each task gets its own workspace folder.** You can browse its files in the
  app or download them as a zip.

## Security: what stops the agents doing something nasty

The agents have full autonomy inside their workspace: they can write code, run
commands, install packages and use git. Around that:

1. **It never touches your PC.** Everything runs in a throwaway cloud
   container. The only way a file reaches your computer is if you download it.
2. **Agents run as an unprivileged user.** They have no sudo. The app's
   database, your API keys and your password live in a place that user can't
   read.
3. **Downloads only come from trusted sources.** Any URL in a command must
   belong to an official registry or code host (PyPI, npm, GitHub, crates.io,
   Debian, Hugging Face and similar). Anything else is refused and shows up as
   ⛔ in the step log. You can extend the list with `trusted_domains` in
   `agents.yaml` or the `EXTRA_TRUSTED_DOMAINS` variable.
4. **Known malware moves are blocked:** `curl … | sh` from unknown sites,
   reverse shells, crypto miners, base64-decoded payloads, wiping disks,
   setuid binaries, cloud metadata endpoints, and reading other processes'
   memory or environment.
5. **Web reading is text only.** Agents can read any public web page as text,
   but they can't reach internal addresses, and binary files aren't fetched
   that way.
6. **Commands have time limits,** and the whole process group is killed on
   timeout.

No sandbox is perfect. A program an agent writes can still open network
connections at runtime. That's why the container is isolated and holds no
secrets the agents can reach. Give `AGENT_GITHUB_TOKEN` access only to the
repos you want them to touch.

## Put it in the cloud (about 10 minutes, doable from a phone)

You need:

- **An OpenRouter account and API key** (https://openrouter.ai). It's one key
  for Claude, GPT, DeepSeek and more. Add some credit.
  - Chat subscriptions such as ChatGPT Plus or Claude Pro **don't** include
    API access.
- **A host that keeps running when your PC is off.** The steps below use
  Railway (Hobby plan, about $5/month). Any Docker host works:
  - Fly.io
  - Render with a paid instance: the free tier sleeps and would stop
    multi-day tasks
  - A $5 VPS with `docker compose up -d`

### Railway

1. Sign in at https://railway.com with GitHub.
2. **New Project → Deploy from GitHub repo →** pick this repository.
   - If this code isn't on `main` yet, choose the branch that has it.
3. Open the service **Settings**:
   - **Root Directory:** `orchestrator`
   - **Networking → Generate Domain.** This gives you the URL you'll open on
     your phone.
4. Right-click the service → **Attach Volume**, with mount path **`/data`**.
   This is where tasks and files are stored. Without it, everything is wiped
   on every deploy.
5. **Variables:**
   - `APP_PASSWORD`: a long password. It's the only thing protecting the
     app, so make it strong.
   - `OPENROUTER_API_KEY`: your key.
   - Optional: `TAVILY_API_KEY` (better web search), `AGENT_GITHUB_TOKEN`,
     `TASK_BUDGET_USD`, `DAILY_BUDGET_USD`.
6. Deploy, open the domain, and log in.
   - On iPhone: Share → **Add to Home Screen**, and it behaves like an app.

Never put keys in the repo; this repository is public. Keys only go in the
host's variables.

### Any server with Docker

```bash
cd orchestrator
cp .env.example .env    # fill in APP_PASSWORD and OPENROUTER_API_KEY
docker compose up -d --build
# then open http://<server>:8000 (put it behind HTTPS, e.g. Caddy, before using it over the internet)
```

## Choosing the models

Edit `agents.yaml`, or change them in the app's **Team** panel. Model IDs are
the ones listed at https://openrouter.ai/models. The defaults are:

| Agent      | Default model              | Job                                   |
|------------|----------------------------|---------------------------------------|
| lead       | `anthropic/claude-opus-5.5`| plans, assigns, judges, reports       |
| coder      | `anthropic/claude-opus-5.5`| code, shell, git, tests               |
| researcher | `deepseek/deepseek-chat`   | web research, notes with sources      |
| critic     | `openai/gpt-5`             | strict reviews, runs the tests itself |

**Check these IDs** on OpenRouter before your first run, because names change
as new versions ship. If an ID is wrong, the task stops with a clear "model
not found" message, and you fix it in the Team panel.

**Adding a teammate** means adding a block under `agents:`. The lead reads
each agent's `description` to decide who gets which step, so describe
strengths well. Here's an example:

```yaml
  designer:
    model: google/gemini-2.5-pro
    description: UI/UX and front-end visual design; writes HTML/CSS and critiques layouts.
    tools: [read, write, web]
    review_by: critic
```

**Tool groups:**

- `web`: search the web and read pages
- `read`: list and read files
- `write`: create and edit files
- `shell`: run commands

`review_by` makes another agent approve the work before the step counts as
done. It can send the work back up to `max_revisions` times.

## Using it

- **Describe the whole goal in one message.** For example: *"Build a FastAPI
  service that tracks my gym workouts, with a SQLite DB, tests, and a
  README. Research the best way to deploy it for free."*
- **The progress bar** shows the steps finished across all rounds so far. Tap
  a step to see its instructions, its report, and a live log of every tool
  call.
- **You can message the lead mid-task** to change direction. It picks up your
  message at its next decision.
- **When the task needs something from you, it shows "Needs you".** That
  happens when the lead has a question or the budget ran out. Answer it, or
  raise the budget in the ⋯ menu and send *"continue"*.
- **Getting the results:** in the ⋯ menu, use **Browse files**, or **Download
  all files (.zip)**. If you gave the agents a GitHub token, you can also ask
  them to push to a branch and open a PR.

## Running locally

```bash
cd orchestrator
pip install -r requirements-dev.txt
APP_PASSWORD=dev OPENROUTER_API_KEY=sk-or-... uvicorn app.main:app --reload
pytest   # the tests use a scripted fake model, so no API calls
```

On a normal machine (not running as root, no `agent` user), commands run as
your own user. The user separation only applies inside the Docker image.

## How it's put together

| File | What it does |
|------|--------------|
| `app/engine.py` | The loop: the lead plans a round, steps run in parallel where independent, critic review cycles, budgets, resuming |
| `app/tools.py`  | Web search and fetch, workspace file tools, the sandboxed shell |
| `app/safety.py` | Command denylist, trusted-domain allowlist, blocking internal addresses |
| `app/llm.py`    | Talks to any OpenAI-compatible API (OpenRouter by default), with long retries for outages |
| `app/db.py`     | SQLite storage for tasks, messages, steps, the event log and costs |
| `app/main.py`   | Password-protected web API |
| `app/static/index.html` | The chat UI (a single file, works on phones) |

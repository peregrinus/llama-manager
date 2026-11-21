# llama-manager

Single-file **Python 3 + Flask** web UI to launch, monitor and stop `llama-server`
(and `sd-server`) instances. No Node, no database, **no test suite, no CI, no
linter** — there is no build to run (it is a git repo, branch `main`,
remote `origin`). The whole app is `app.py`.

## Running / verifying

There is no build, test, lint or type-check step. Verify by running it and
hitting it in a browser:

    python3 app.py            # serves http://0.0.0.0:6522 (change PORT in app.py)

- Login: created on first run (salted PBKDF2 hash in `config.json`, no plaintext); change it on `/edit` → Credentials.
- `GET /health` is public — useful as a liveness check without logging in.
- **This checkout is on Windows, but the app targets a Linux/Ubuntu server.**
  The VRAM, GPU-stats and system widgets shell out to `llama-server --list-devices`,
  `nvtop -s` and read `/proc` — they silently return empty under Windows. That is
  expected here, not a bug. On the real box they populate.

## Architecture (single file — keep it that way)

- The UI is **not** in a templates directory. `SHARED_CSS`, `INDEX_HTML`,
  `EDIT_HTML`, `LOGIN_HTML` are Python string constants; `__CSS__` and `__ERROR__`
  are placeholders substituted at render time. **Editing the UI means editing those
  strings in `app.py`**, not adding `.html` files.
- `preview/*.html` are **static design mockups only** — never served, not wired to
  the app. Treat them as scratch reference, not source.
- State is three JSON files next to `app.py` (created/managed automatically):
  - `models.json` — the model command list, edited via `/edit`.
  - `pids.json` — live PIDs, written automatically; used to re-attach after a restart.
  - `config.json` — holds the optional `api_key` (plaintext).
- **Process model:** each model is a `subprocess.Popen(..., start_new_session=True)`
  so Stop can `killpg` the whole group (SIGTERM, then SIGKILL after 10 s).
  `recover_pids()` re-attaches to still-alive servers after a restart by matching
  `/proc/*/cmdline` against the stored command. **llama-server processes outlive the
  app by design** — do not add "kill everything on exit".
- **Auth:** `before_request` gates everything except `/login` and `/health`. Browsers
  use a session; programmatic `/api/*` callers may send `X-API-Key: <key>` instead.
  **3 consecutive failed logins calls `os._exit(1)`** (intentional lockout).

## Conventions worth keeping

- **stdlib-only** beyond Flask. Don't add runtime deps.
- Keep it one file; keep the UI in the embedded string constants.
- The per-model `--api-key` in `models.json` and the `config.json` API key are
  **deliberately plaintext** per the requirements — don't "fix" them into env vars.
  Login credentials are different: stored salted-hashed, and changed via `/api/auth`.
- **Adding a new polled metric** (VRAM / GPU / system-style widget) follows a fixed
  pattern — see the `add-stat-widget` skill before doing it ad hoc.
- Deploy/update the systemd service with the `systemd-deploy` skill.

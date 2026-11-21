# Llama Server Manager

A single-file web UI to launch, monitor and stop `llama-server` instances.
Built with **Python 3 + Flask** — no Node.js, no database.

## Files

| File | Purpose |
|---|---|
| `app.py` | The entire application (server + UI), one file |
| `models.json` | Your model commands (plain text, created on first save) |
| `pids.json` | Live process IDs (managed automatically) |
| `logs/<id>.log` | stdout/stderr of each model run |
| `llama-manager.service` | systemd unit so the manager (and boot models) start at boot |

## Setup (Ubuntu x64)

1. Copy this folder to the server:
   ```bash
   sudo cp -r llama-manager /opt/
   ```
2. Install Flask (Python 3 is already on Ubuntu):
   ```bash
   sudo apt update && sudo apt install -y python3-flask
   # or: pip3 install flask
   ```
3. Install and start the service:
   ```bash
   sudo cp /opt/llama-manager/llama-manager.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now llama-manager
   ```
4. Open **http://\<server-ip\>:6522**

> Edit `User=` in the service file if your user is not `nome`, and
> `WorkingDirectory`/`ExecStart` if you put the folder somewhere else.

## Usage

- **Main page** — one card per model. Each card shows the full command
  broken into chips (binary path highlighted), a live status dot, a VRAM
  fit indicator (green/yellow/red, estimated from model size, quant,
  context and offloaded layers — computed at startup and when a model is
  edited, not polled), and Run / Stop / log buttons. Status refreshes
  every 3 seconds.
- **Performance overlay** — while a model is running, its card shows the
  latest tokens/sec (and prompt tokens/sec) parsed from the most recent
  `llama_print_timings` line in its log. Only works for models started
  via the app (those are the ones with a log file); the app just reads
  the last ~128 KB of the log on each 3 s poll — the server process
  writes the log itself, nothing is streamed.
- **Edit commands** (`/edit`) — add, edit or delete model commands, and
  toggle **Run at boot** per model. Changes are saved to `models.json`
  immediately and appear on the main page on the next refresh.
- **Run at boot** — any model with the flag set is started automatically
  when the manager starts, i.e. at server boot (the manager itself is a
  systemd service). You can freely add *and remove* boot models from
  `/edit` — it is not append-only.
- **Stopping** — Stop sends SIGTERM to the whole process group, waits up
  to 10 s, then SIGKILL. Deleting a model also stops it if it is running.
- **Logs** — the `log` button on each card shows the last 200 lines of
  that model's output, handy for spotting why a server failed to start.

## VRAM Monitor

- A compact VRAM widget appears at the top of the main page showing each Vulkan device's usage.
- It queries `llama-server --list-devices` (cached 5 s) — one short-lived process every 5 s.
- The binary path is `/opt/services/llama.cpp/llama-server` by default — change `LLAMA_SERVER_PATH` at the top of `app.py` if yours is elsewhere.

## Authentication

- **First run:** the login page becomes a "create account" form — pick a username and password. They are stored in `config.json` as a salted PBKDF2 hash (no plaintext on disk).
- **Change credentials** anytime on the **Edit commands** page (`/edit` → Credentials panel). It asks for the current password; the old login stops working immediately.
- Every page and API endpoint requires login.
- **3 consecutive failed logins shut the app down** — the process exits. A successful login resets the counter.
- While a session is active, a **Logout** button in the header ends it.
- Note: if the app is running under systemd with `Restart=on-failure`, a lockout shutdown will be restarted by systemd after 5 s. Set `Restart=no` in the unit file if you want it to stay stopped after a lockout.
- Running `llama-server` processes are left alone when the app exits — they run independently by design.

## Notes

- The VRAM widget shows a red bar when a device is >90 % used.

- Commands are stored as full command lines **including the binary path**
  and executed exactly as written (nothing is assumed to be on `$PATH`).
- The manager listens on `0.0.0.0:6522`, plain HTTP. To change the port,
  edit `PORT` at the top of `app.py`.
- If the manager is restarted, it re-attaches to any `llama-server`
  processes that are still alive (via `pids.json`), so their status and
  Stop buttons keep working.

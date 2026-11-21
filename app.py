#!/usr/bin/env python3
"""
Llama Server Manager
====================
A single-file web UI to launch, monitor and stop llama-server instances.

- Listens on 0.0.0.0:6522 (plain HTTP)
- Stores model commands in models.json (plain text, next to this file)
- Persists PIDs in pids.json so process state survives app restarts
- Models flagged "run on boot" are started automatically when this app
  starts (the app itself is started by systemd at boot)

Only dependency: Flask  (pip install flask)
"""

import hashlib
import hmac
import html
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, request, session

BASE_DIR = Path(__file__).resolve().parent
MODELS_FILE = BASE_DIR / "models.json"
PIDS_FILE = BASE_DIR / "pids.json"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

HOST = "0.0.0.0"
PORT = 6522

# llama-server binaries to query for VRAM (--list-devices). Both the
# Vulkan and SYCL builds are checked so the widget covers whichever
# backend a model is running under.
LLAMA_SERVER_BINARIES = [
    "/home/peregrinus/llama-b70-sycl-ubuntu24-x64/llama-server",
    "/opt/services/llama.cpp/llama-server",
]

# --- authentication ---
# Credentials are created on first login and stored in config.json as a
# salted PBKDF2-SHA256 hash (no plaintext on disk). Change them on /edit.
MAX_LOGIN_ATTEMPTS = 3

# Optional API key for programmatic access to /api/*. When non-empty, a
# request may present it via the X-API-Key header instead of using a
# browser session. Empty = disabled (session login only).
# Stored in config.json next to this file; editable on the /edit page.

app = Flask(__name__)
app.secret_key = "llama-manager-insecure-secret"
lock = threading.Lock()
processes = {}   # model_id -> subprocess.Popen (processes we started)
pid_map = {}     # model_id -> pid (processes recovered after an app restart)
failed_attempts = 0  # consecutive failed logins; app exits at MAX_LOGIN_ATTEMPTS


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def load_models():
    if MODELS_FILE.exists():
        try:
            with open(MODELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("models", [])
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_models(models):
    tmp = MODELS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"models": models}, f, indent=2)
    os.replace(tmp, MODELS_FILE)


def load_pids():
    if PIDS_FILE.exists():
        try:
            with open(PIDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_pids():
    pids = {mid: p.pid for mid, p in processes.items() if p.poll() is None}
    pids.update({mid: pid for mid, pid in pid_map.items() if pid_alive(pid)})
    with open(PIDS_FILE, "w", encoding="utf-8") as f:
        json.dump(pids, f)


CONFIG_FILE = BASE_DIR / "config.json"


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(cfg):
    tmp = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_FILE)


# API key loaded from disk at startup; updated live by PUT /api/config
API_KEY = load_config().get("api_key", "")


# --------------------------------------------------------------------------
# Process management
# --------------------------------------------------------------------------

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True  # exists, just not ours to signal


def model_status(mid):
    proc = processes.get(mid)
    if proc is not None:
        if proc.poll() is None:
            return {"running": True, "pid": proc.pid}
        processes.pop(mid, None)
        return {"running": False, "pid": None}
    pid = pid_map.get(mid)
    if pid is not None:
        if pid_alive(pid):
            return {"running": True, "pid": pid}
        pid_map.pop(mid, None)
    return {"running": False, "pid": None}


def start_model(model):
    try:
        cmd = shlex.split(model["command"])
    except ValueError as e:
        raise ValueError(f"could not parse command: {e}")
    if not cmd:
        raise ValueError("empty command")

    log_path = LOGS_DIR / f"{model['id']}.log"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} started: {model['command']}\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group -> clean group kill
    )
    processes[model["id"]] = proc
    save_pids()
    return proc


def stop_model(mid):
    proc = processes.get(mid)
    if proc is not None and proc.poll() is None:
        _signal_pid(proc.pid, signal.SIGTERM)
        for _ in range(50):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is None:
            _signal_pid(proc.pid, signal.SIGKILL)
        processes.pop(mid, None)
        save_pids()
        return True

    pid = pid_map.get(mid)
    if pid is not None and pid_alive(pid):
        _signal_pid(pid, signal.SIGTERM)
        for _ in range(50):
            if not pid_alive(pid):
                break
            time.sleep(0.2)
        if pid_alive(pid):
            _signal_pid(pid, signal.SIGKILL)
        pid_map.pop(mid, None)
        save_pids()
        return True
    return False


def _signal_pid(pid, sig):
    """Send a signal to the whole process group (falls back to the pid)."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _running_processes():
    """[(pid, cmdline)] for every running process, read from /proc."""
    procs = []
    try:
        pids = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return procs
    for pid_s in pids:
        try:
            with open(f"/proc/{pid_s}/cmdline", "rb") as f:
                raw = f.read()
        except (OSError, PermissionError):
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if cmdline:
            procs.append((int(pid_s), cmdline))
    return procs


def _match_model_in_cmdline(model, cmdline, cmdline_tokens):
    """True if a process cmdline belongs to this configured model.

    Matches on the -m model file path when the command has one (robust to
    flag order / spacing differences), else on every command token being
    present. Anything that does not match is ignored by design.
    """
    try:
        tokens = shlex.split(model.get("command", ""))
    except ValueError:
        return False
    if not tokens:
        return False
    path = _flag_value(tokens, {"-m", "--model"})
    if path:
        return path in cmdline
    return all(t in cmdline_tokens for t in tokens)


def recover_pids():
    """Re-attach to llama-server processes that survived an app restart."""
    for mid, pid in load_pids().items():
        if pid_alive(pid):
            pid_map[mid] = pid

    # The app may have been killed before pids.json was saved — scan the
    # running processes and match any configured model by its command.
    # Startup only; models with no matching process are left alone.
    unmatched = [m for m in load_models() if m["id"] not in pid_map]
    if not unmatched:
        return
    procs = _running_processes()
    print(f"[recover] scanned {len(procs)} running processes")
    taken = set(pid_map.values())
    for pid, cmdline in procs:
        if pid in taken or not unmatched:
            continue
        ctokens = cmdline.split()
        for m in unmatched:
            if _match_model_in_cmdline(m, cmdline, ctokens):
                pid_map[m["id"]] = pid
                taken.add(pid)
                unmatched.remove(m)
                print(f"[recover] re-attached {m['name']} (pid {pid})")
                break
    for m in unmatched:
        print(f"[recover] no running process matches {m['name']}")
    if pid_map:
        save_pids()  # persist recovered PIDs so the next restart is easy


def boot_models():
    """Start every model flagged run_on_boot that is not already running."""
    for model in load_models():
        mid = model["id"]
        if model.get("run_on_boot") and not model_status(mid)["running"]:
            try:
                proc = start_model(model)
                print(f"[boot] started {model['name']} (pid {proc.pid})")
            except Exception as e:
                print(f"[boot] failed to start {model['name']}: {e}")


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------

PUBLIC_PATHS = {"/login", "/health"}


def _hash_password(password, salt):
    """Salted PBKDF2-SHA256 digest (stdlib only)."""
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def _get_auth():
    """Stored credentials dict (username/salt/hash) from config.json, or None."""
    return load_config().get("auth")


def _check_credentials(username, password):
    auth = _get_auth()
    if not auth:
        return False
    ok_user = hmac.compare_digest(username.encode("utf-8"),
                                  auth.get("username", "").encode("utf-8"))
    digest = _hash_password(password, bytes.fromhex(auth.get("salt", "")))
    return ok_user and hmac.compare_digest(digest.encode("utf-8"),
                                           auth.get("hash", "").encode("utf-8"))


def render_login(error=None, fatal=False, creating=False):
    err = ""
    if error:
        cls = "error fatal" if fatal else "error"
        err = f'<div class="{cls}">{html.escape(error)}</div>'
    subtitle = "First run - pick a username and password" if creating else "Sign in to continue"
    button = "Create account" if creating else "Sign in"
    confirm = ('    <label>Password again\n'
               '      <input type="password" name="confirm" autocomplete="new-password" required>\n'
               '    </label>\n') if creating else ""
    return (LOGIN_HTML.replace("__CSS__", SHARED_CSS)
                       .replace("__ERROR__", err)
                       .replace("__SUBTITLE__", subtitle)
                       .replace("__CONFIRM__", confirm)
                       .replace("__BUTTON__", button))


def _shutdown():
    time.sleep(1)  # let the final response reach the client first
    os._exit(1)


def _api_key_valid():
    """True if the request presents the configured API key (when enabled)."""
    if not API_KEY:
        return False
    provided = request.headers.get("X-API-Key", "")
    return hmac.compare_digest(provided.encode("utf-8"), API_KEY.encode("utf-8"))


@app.before_request
def require_auth():
    if request.path in PUBLIC_PATHS or session.get("authed"):
        return None
    if request.path.startswith("/api/"):
        if _api_key_valid():
            return None
        return jsonify({"error": "unauthorized"}), 401
    return redirect("/login")


@app.get("/login")
def login_page():
    if session.get("authed"):
        return redirect("/")
    return render_login()


@app.post("/login")
def login_submit():
    global failed_attempts
    data = request.get_json(silent=True) or request.form
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not _get_auth():
        # First run: create the credentials instead of checking them.
        if not username or not password:
            return render_login("Choose a username and password.", creating=True), 400
        if password != data.get("confirm"):
            return render_login("Passwords do not match.", creating=True), 400
        with lock:
            salt = secrets.token_hex(16)
            cfg = load_config()
            cfg["auth"] = {"username": username, "salt": salt,
                           "hash": _hash_password(password, bytes.fromhex(salt))}
            save_config(cfg)
        session["authed"] = True
        return redirect("/")

    with lock:
        if _check_credentials(username, password):
            failed_attempts = 0
            session["authed"] = True
            return redirect("/")
        failed_attempts += 1
        remaining = MAX_LOGIN_ATTEMPTS - failed_attempts
        give_up = failed_attempts >= MAX_LOGIN_ATTEMPTS
    if give_up:
        print("[auth] too many failed login attempts — shutting down")
        threading.Thread(target=_shutdown, daemon=True).start()
        return render_login("Too many failed attempts — shutting down.", fatal=True), 403
    noun = "attempt" if remaining == 1 else "attempts"
    return render_login(f"Invalid username or password. {remaining} {noun} remaining.")


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login")


# --------------------------------------------------------------------------
# VRAM query (cached 5 s to avoid spawning a process on every poll)
# --------------------------------------------------------------------------

_vram_cache = {"data": None, "ts": 0}


def _get_vram():
    """Query device VRAM via llama-server --list-devices (cached 5 s).

    Checks every known binary (Vulkan and SYCL builds) and merges the
    device lists, so the widget covers whichever backend is in use.
    """
    now = time.time()
    if _vram_cache["data"] is not None and now - _vram_cache["ts"] < 5:
        return _vram_cache["data"]

    devices = []
    for binary in LLAMA_SERVER_BINARIES:
        try:
            result = subprocess.run(
                [binary, "--list-devices"],
                capture_output=True, text=True, timeout=10
            )
        except Exception:
            continue  # binary missing or failed — skip it
        for line in result.stdout.splitlines():
            # Matches both Vulkan and SYCL device lines, e.g.
            #   Vulkan0: Intel(R) Arc A770 (16384 MiB, 16000 MiB free)
            #   SYCL0: Intel(R) Graphics [0xe223] (32656 MiB, 32372 MiB free)
            m = re.match(
                r"\s*(Vulkan|SYCL)(\d+):\s+(.+?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB free\)", line
            )
            if m:
                total = int(m.group(4))
                free = int(m.group(5))
                used_pct = round((1 - free / total) * 100) if total > 0 else 0
                devices.append({
                    "backend": m.group(1),
                    "index": int(m.group(2)),
                    "name": m.group(3).strip(),
                    "total": total,
                    "free": free,
                    "used_pct": used_pct,
                })

    data = {"devices": devices}
    _vram_cache["data"] = data
    _vram_cache["ts"] = now
    return data


@app.get("/api/vram")
def api_vram():
    return jsonify(_get_vram())


@app.get("/api/binaries")
def api_binaries():
    """Known llama-server binaries (for the Binary dropdown on the edit page)."""
    return jsonify(LLAMA_SERVER_BINARIES)


@app.get("/api/config")
def api_get_config():
    with lock:
        return jsonify({"api_key": API_KEY})


@app.put("/api/config")
def api_update_config():
    global API_KEY
    data = request.get_json(force=True, silent=True) or {}
    key = (data.get("api_key") or "").strip()
    with lock:
        cfg = load_config()
        cfg["api_key"] = key
        save_config(cfg)
        API_KEY = key
    return jsonify({"api_key": key})


@app.post("/api/auth")
def api_update_auth():
    """Change the login credentials (requires the current password)."""
    data = request.get_json(force=True, silent=True) or {}
    current = data.get("current_password") or ""
    new_user = (data.get("username") or "").strip()
    new_pass = data.get("password") or ""
    if not new_user or not new_pass:
        return jsonify({"error": "username and new password are required"}), 400
    with lock:
        auth = _get_auth()
        if not auth or not _check_credentials(auth["username"], current):
            return jsonify({"error": "current password is incorrect"}), 400
        salt = secrets.token_hex(16)
        cfg = load_config()
        cfg["auth"] = {"username": new_user, "salt": salt,
                       "hash": _hash_password(new_pass, bytes.fromhex(salt))}
        save_config(cfg)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# GPU stats via nvtop (cached 30 s — no need to poll faster than that)
# --------------------------------------------------------------------------

_gpu_stats_cache = {"data": None, "ts": 0}
GPU_STATS_INTERVAL = 30


def _get_gpu_stats():
    """Query per-GPU stats via `nvtop -s` (JSON output), cached 30 s.

    nvtop reports null for fields it cannot read (e.g. power_draw on the
    Arc Pro B70, mem_clock); those are passed through as None and rendered
    as a dash in the UI.
    """
    now = time.time()
    if _gpu_stats_cache["data"] is not None and now - _gpu_stats_cache["ts"] < GPU_STATS_INTERVAL:
        return _gpu_stats_cache["data"]

    devices = []
    try:
        result = subprocess.run(
            ["nvtop", "-s"],
            capture_output=True, text=True, timeout=10
        )
        for dev in json.loads(result.stdout):
            devices.append({
                "device_name": dev.get("device_name") or "Unknown",
                "temp": dev.get("temp"),
                "fan_speed": dev.get("fan_speed"),
                "power_draw": dev.get("power_draw"),
                "gpu_util": dev.get("gpu_util"),
                "mem_util": dev.get("mem_util"),
            })
    except Exception:
        pass  # nvtop missing or bad output — section will be hidden

    data = {"devices": devices}
    _gpu_stats_cache["data"] = data
    _gpu_stats_cache["ts"] = now
    return data


@app.get("/api/gpu-stats")
def api_gpu_stats():
    return jsonify(_get_gpu_stats())


# --------------------------------------------------------------------------
# System stats: CPU / RAM / disk (stdlib only, /proc based)
# --------------------------------------------------------------------------

# Disk to report on; the mount point is looked up in /proc/mounts.
TARGET_DISK = "/dev/nvme0n1p2"

_cpu_last = {"sample": None, "ts": 0}


def _cpu_times():
    """(total, idle) jiffies from the first line of /proc/stat."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    values = [int(v) for v in parts[1:]]
    idle = values[3] + values[4]  # idle + iowait
    return sum(values), idle


def _cpu_util():
    """CPU utilisation % since the previous call (None on the first sample)."""
    now = time.time()
    total, idle = _cpu_times()
    prev = _cpu_last["sample"]
    prev_ts = _cpu_last["ts"]
    _cpu_last["sample"] = (total, idle)
    _cpu_last["ts"] = now
    if prev is None or now - prev_ts <= 0:
        return None
    dtotal = total - prev[0]
    didle = idle - prev[1]
    if dtotal <= 0:
        return None
    return round(100.0 * (dtotal - didle) / dtotal)


def _ram_util():
    """RAM usage from /proc/meminfo (MiB and used %)."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields:
                info[key.strip()] = int(fields[0])  # kB
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    if total <= 0:
        return None
    return {
        "total_mb": total // 1024,
        "used_mb": (total - avail) // 1024,
        "used_pct": round((1 - avail / total) * 100),
    }


def _disk_usage():
    """Usage of the mount point for TARGET_DISK (GB and used %)."""
    mount = None
    with open("/proc/mounts") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == TARGET_DISK:
                mount = parts[1]
                break
    if mount is None:
        return None
    usage = shutil.disk_usage(mount)
    if usage.total <= 0:
        return None
    gb = 2 ** 30
    return {
        "device": TARGET_DISK,
        "mount": mount,
        "total_gb": round(usage.total / gb, 1),
        "used_gb": round(usage.used / gb, 1),
        "free_gb": round(usage.free / gb, 1),
        "used_pct": round(usage.used / usage.total * 100),
    }


@app.get("/api/system-stats")
def api_system_stats():
    """Host CPU / RAM / disk usage for the front page widget."""
    return jsonify({
        "cpu_util": _cpu_util(),
        "ram": _ram_util(),
        "disk": _disk_usage(),
    })


def _flag_value(tokens, names):
    for i, t in enumerate(tokens):
        if t in names and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            return tokens[i + 1]
    return None


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def _tail_log(log_path, n=131072):
    """Last n bytes of a log file (cheap: one seek + one read)."""
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def parse_perf(log_path):
    """Latest llama_print_timings numbers from a log file, or None.

    Only the file tail is read, on demand (when the front page polls) —
    the child process writes the log itself, the app never streams it.
    """
    text = _tail_log(log_path)
    if not text:
        return None
    # Works with both the classic "llama_print_timings: ... / N runs" format
    # and the newer "slot print_timing: ... / N tokens" format.
    # findall + [-1]: the tail holds many timings lines; we want the newest.
    prompt = re.findall(r"prompt eval time =\s+([\d.]+)\s+ms\s*/\s+(\d+)\s+tokens", text)
    evalm = re.findall(r"(?<!prompt )eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+(?:runs|tokens)", text)
    # stable-diffusion.cpp (sd-server) logs e.g.
    #   "generate_image completed in 6.32s --- image.cpp:1227"
    imgm = re.findall(r"generate_image completed in\s+([\d.]+)\s*s", text)
    perf = {}
    if prompt:
        p_ms, p_n = float(prompt[-1][0]), int(prompt[-1][1])
        if p_ms > 0:
            perf["prompt_tps"] = round(p_n / (p_ms / 1000), 1)
    if evalm:
        e_ms, e_n = float(evalm[-1][0]), int(evalm[-1][1])
        if e_ms > 0:
            perf["tps"] = round(e_n / (e_ms / 1000), 1)
    if imgm:
        perf["image_s"] = round(float(imgm[-1]), 2)
    return perf or None


@app.get("/api/models")
def api_models():
    with lock:
        out = []
        for m in load_models():
            st = model_status(m["id"])
            entry = {**m, "status": st}
            # live perf only makes sense for running, app-started models
            entry["perf"] = parse_perf(LOGS_DIR / f"{m['id']}.log") if st["running"] else None
            out.append(entry)
    return jsonify(out)


@app.post("/api/models")
def api_create():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    command = (data.get("command") or "").strip()
    if not name or not command:
        return jsonify({"error": "name and command are required"}), 400
    model = {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "command": command,
        "run_on_boot": bool(data.get("run_on_boot")),
    }
    with lock:
        models = load_models()
        models.append(model)
        save_models(models)
    return jsonify(model), 201


@app.put("/api/models/<mid>")
def api_update(mid):
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        models = load_models()
        for m in models:
            if m["id"] == mid:
                if "name" in data and (data["name"] or "").strip():
                    m["name"] = data["name"].strip()
                if "command" in data and (data["command"] or "").strip():
                    m["command"] = data["command"].strip()
                if "run_on_boot" in data:
                    m["run_on_boot"] = bool(data["run_on_boot"])
                save_models(models)
                return jsonify(m)
    return jsonify({"error": "model not found"}), 404


@app.delete("/api/models/<mid>")
def api_delete(mid):
    with lock:
        models = load_models()
        remaining = [m for m in models if m["id"] != mid]
        if len(remaining) == len(models):
            return jsonify({"error": "model not found"}), 404
        save_models(remaining)
    stop_model(mid)  # stop it if it is running
    return jsonify({"ok": True})


@app.post("/api/models/<mid>/run")
def api_run(mid):
    with lock:
        model = next((m for m in load_models() if m["id"] == mid), None)
        if model is None:
            return jsonify({"error": "model not found"}), 404
        if model_status(mid)["running"]:
            return jsonify({"error": "already running"}), 409
        try:
            proc = start_model(model)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"failed to start: {e}"}), 500
    return jsonify({"ok": True, "pid": proc.pid})


@app.post("/api/models/<mid>/stop")
def api_stop(mid):
    with lock:
        ok = stop_model(mid)
    if not ok:
        return jsonify({"error": "not running"}), 409
    return jsonify({"ok": True})


@app.get("/api/models/<mid>/log")
def api_log(mid):
    log_path = LOGS_DIR / f"{mid}.log"
    if not log_path.exists():
        return jsonify({"log": ""})
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return jsonify({"log": "\n".join(lines[-200:])})


@app.get("/health")
def health():
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# UI (embedded, no external assets)
# --------------------------------------------------------------------------

SHARED_CSS = """
  :root {
    --bg: #0e1116; --card: #161b23; --border: #262d3a;
    --text: #e6e9ef; --muted: #8b93a3; --accent: #4f8cff;
    --green: #2dd4a7; --red: #f0564f; --chip: #202734;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; align-items: center; justify-content: space-between;
           padding: 16px 28px; border-bottom: 1px solid var(--border);
           position: sticky; top: 0; background: var(--bg); z-index: 5; }
  header h1 { font-size: 17px; margin: 0; font-weight: 600; letter-spacing: .2px; }
  header h1 span { color: var(--accent); }
  nav a { color: var(--muted); text-decoration: none; font-size: 14px;
          padding: 8px 14px; border: 1px solid var(--border); border-radius: 8px; }
  nav a + a { margin-left: 8px; }
  nav a:hover { color: var(--text); border-color: var(--accent); }
  /* Flag editor */
  .flags-header { font-size: 12px; font-weight: 600; color: var(--muted); margin: 14px 0 8px;
                  text-transform: uppercase; letter-spacing: .5px; }
  .flag-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .flag-row input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--accent); flex: none; }
  .flag-name { font-family: var(--mono); font-size: 12px; color: var(--accent); min-width: 50px; flex: none; }
  .flag-val { flex: 1; background: var(--chip); border: 1px solid var(--border); border-radius: 6px;
              padding: 6px 10px; color: var(--text); font-family: var(--mono); font-size: 12px; }
  .flag-val:focus { outline: none; border-color: var(--accent); }
  .flag-val:disabled { opacity: .4; }
  .positional { font-family: var(--mono); font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .cmd-wrap { position: relative; }
  .cmd-wrap label { margin-bottom: 4px; }
  .cmd-wrap input { font-family: var(--mono); font-size: 12px; }
  .binary-wrap { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
  .binary-wrap label { margin-bottom: 0; flex: 1; }
  .binary-label { font-size: 12px; color: var(--muted); white-space: nowrap; }
  main { padding: 28px; max-width: 1200px; margin: 0 auto; }
  button { font: inherit; font-size: 13px; font-weight: 600; border-radius: 8px;
           border: 1px solid var(--border); background: var(--chip);
           color: var(--text); padding: 8px 14px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.run  { background: rgba(45,212,167,.15); border-color: rgba(45,212,167,.5); color: var(--green); }
  button.stop { background: rgba(240,86,79,.12);  border-color: rgba(240,86,79,.5);  color: var(--red); }
  button.ghost { background: transparent; font-weight: 400; color: var(--muted); }
  /* VRAM widget */
  .vram { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
          padding: 12px 18px; margin-bottom: 18px; }
  .vram-title { font-size: 12px; font-weight: 600; color: var(--muted); margin-bottom: 8px; }
  .vram-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
  .vram-row:last-child { margin-bottom: 0; }
  .vram-name { font-size: 12px; flex: 1; min-width: 0; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }
  .vram-bar { flex: 2; min-width: 100px; height: 8px; border-radius: 4px;
              background: var(--chip); position: relative; overflow: hidden; }
  .vram-fill { height: 100%; border-radius: 4px; background: var(--accent); transition: width .3s; }
  .vram-fill.warn { background: var(--red); }
  .vram-info { font-size: 11px; color: var(--muted); font-family: var(--mono); white-space: nowrap; }
  /* GPU stats */
  .gpu-stats { display: flex; flex-wrap: wrap; gap: 10px; }
  .gpu-stat { flex: 1; min-width: 230px; background: var(--chip); border: 1px solid var(--border);
              border-radius: 8px; padding: 10px 12px; }
  .gpu-stat-name { font-size: 12px; font-weight: 600; margin-bottom: 8px; white-space: nowrap;
                   overflow: hidden; text-overflow: ellipsis; }
  .gpu-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
  .gpu-stat-cell { font-size: 11px; color: var(--muted); font-family: var(--mono); white-space: nowrap; }
  .gpu-stat-cell b { color: var(--text); font-weight: 600; }
"""

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Llama Server Manager</title>
<style>
__CSS__
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 18px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
          padding: 18px; display: flex; flex-direction: column; gap: 14px; }
  .card:hover { border-color: #33405a; }
  .card-top { display: flex; align-items: center; gap: 10px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--muted); flex: none; }
  .dot.on { background: var(--green); box-shadow: 0 0 8px rgba(45,212,167,.6); }
  .card-name { font-size: 15px; font-weight: 600; flex: 1; }
  .badge { font-size: 11px; color: var(--accent); border: 1px solid rgba(79,140,255,.4);
           border-radius: 999px; padding: 2px 8px; white-space: nowrap; }
  .cmd { display: flex; flex-wrap: wrap; gap: 6px; font-family: var(--mono); font-size: 12px; }
  .chip { background: var(--chip); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
  .chip.bin { background: rgba(79,140,255,.12); border-color: rgba(79,140,255,.45); color: #9cc0ff; }
  .card-bottom { display: flex; align-items: center; gap: 10px; margin-top: auto; }
  .pid { font-family: var(--mono); font-size: 12px; color: var(--muted); margin-left: auto; }
  .perf-row { display: flex; align-items: center; }
  .perf { font-family: var(--mono); font-size: 12px; color: var(--accent); white-space: nowrap; }
  .empty { color: var(--muted); text-align: center; padding: 60px 0; font-size: 14px; }
  .empty a { color: var(--accent); }
  dialog { background: var(--card); color: var(--text); border: 1px solid var(--border);
           border-radius: 12px; width: min(860px, 92vw); padding: 0; }
  dialog::backdrop { background: rgba(0,0,0,.55); }
  .modal-head { display: flex; justify-content: space-between; align-items: center;
                padding: 14px 18px; border-bottom: 1px solid var(--border); }
  .modal-head h3 { margin: 0; font-size: 14px; }
  pre { margin: 0; padding: 16px 18px; max-height: 55vh; overflow: auto;
        font-family: var(--mono); font-size: 12px; white-space: pre-wrap; word-break: break-word; }
</style>
</head>
<body>
<header>
  <h1><span>&#9670;</span> Llama Server Manager</h1>
  <nav><a href="/edit">Edit commands</a><a href="/logout">Logout</a></nav>
</header>
<main>
  <div id="vram" class="vram" style="display:none">
    <div class="vram-title">VRAM</div>
    <div id="vram-devices"></div>
  </div>
  <div id="gpuStats" class="vram" style="display:none">
    <div class="vram-title">GPU stats</div>
    <div id="gpuStatsList" class="gpu-stats"></div>
  </div>
  <div id="systemStats" class="vram" style="display:none">
    <div class="vram-title">System</div>
    <div id="systemStatsList" class="gpu-stats"></div>
  </div>
  <div id="grid" class="grid"></div>
  <div id="empty" class="empty" hidden>No models yet. <a href="/edit">Add one</a>.</div>
</main>
<dialog id="logModal">
  <div class="modal-head">
    <h3 id="logTitle">Log</h3>
    <button class="ghost" onclick="document.getElementById('logModal').close()">Close</button>
  </div>
  <pre id="logBody"></pre>
</dialog>
<script>
const grid = document.getElementById("grid");
const empty = document.getElementById("empty");
const logModal = document.getElementById("logModal");
const vramEl = document.getElementById("vram");
const vramDiv = document.getElementById("vram-devices");
const gpuStatsEl = document.getElementById("gpuStats");
const gpuStatsList = document.getElementById("gpuStatsList");
const systemStatsEl = document.getElementById("systemStats");
const systemStatsList = document.getElementById("systemStatsList");

function tokenize(cmd) {
  const tokens = [];
  const re = /"([^"]*)"|'([^']*)'|(\\S+)/g;
  let m;
  while ((m = re.exec(cmd)) !== null) {
    tokens.push(m[1] !== undefined ? m[1] : (m[2] !== undefined ? m[2] : m[3]));
  }
  return tokens;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function fmtPerf(p) {
  const parts = [];
  if (p.image_s != null) parts.push("image " + p.image_s + " s");
  if (p.tps != null) parts.push("out " + p.tps + " t/s");
  if (p.prompt_tps != null) parts.push("prompt " + p.prompt_tps + " t/s");
  return parts.join(" \u00b7 ");
}

function render(models) {
  grid.innerHTML = "";
  empty.hidden = models.length > 0;
  for (const m of models) {
    const tokens = tokenize(m.command);
    const chips = [];
    if (tokens.length) chips.push('<span class="chip bin">' + esc(tokens[0]) + "</span>");
    for (let i = 1; i < tokens.length; i++) {
      const t = tokens[i];
      if (t.startsWith("-") && i + 1 < tokens.length && !tokens[i + 1].startsWith("-")) {
        chips.push('<span class="chip">' + esc(t + " " + tokens[i + 1]) + "</span>");
        i++;
      } else {
        chips.push('<span class="chip">' + esc(t) + "</span>");
      }
    }
    const running = m.status && m.status.running;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML =
      '<div class="card-top">' +
        '<span class="dot' + (running ? " on" : "") + '"></span>' +
        '<span class="card-name">' + esc(m.name) + "</span>" +
        (m.run_on_boot ? '<span class="badge">run at boot</span>' : "") +
      "</div>" +
      '<div class="cmd">' + chips.join("") + "</div>" +
      (m.perf ? '<div class="perf-row" title="latest llama_print_timings"><span class="perf">' + esc(fmtPerf(m.perf)) + "</span></div>" : "") +
      '<div class="card-bottom">' +
        (running
          ? '<button class="stop" data-act="stop">&#9632; Stop</button>'
          : '<button class="run" data-act="run">&#9654; Run</button>') +
        '<button class="ghost" data-act="log">log</button>' +
        '<span class="pid">' + (running ? "pid " + m.status.pid : "stopped") + "</span>" +
      "</div>";
    const runBtn = card.querySelector('[data-act="run"]');
    if (runBtn) runBtn.addEventListener("click", () => act(m.id, "run"));
    const stopBtn = card.querySelector('[data-act="stop"]');
    if (stopBtn) stopBtn.addEventListener("click", () => act(m.id, "stop"));
    card.querySelector('[data-act="log"]').addEventListener("click", () => showLog(m));
    grid.appendChild(card);
  }
}

async function act(id, action) {
  await fetch("/api/models/" + id + "/" + action, { method: "POST" });
  refresh();
}

async function showLog(m) {
  const r = await fetch("/api/models/" + m.id + "/log");
  const d = await r.json();
  document.getElementById("logTitle").textContent = m.name + " — log";
  document.getElementById("logBody").textContent = d.log || "(no output yet)";
  logModal.showModal();
}

async function refresh() {
  try {
    const r = await fetch("/api/models");
    if (r.status === 401) { window.location.href = "/login"; return; }
    render(await r.json());
  } catch (e) { /* server briefly unavailable; retry next tick */ }
}

async function refreshVram() {
  try {
    const r = await fetch("/api/vram");
    const d = await r.json();
    if (!d.devices || !d.devices.length) {
      vramEl.style.display = "none";
      return;
    }
    vramEl.style.display = "";
    vramDiv.innerHTML = d.devices.map(dev => {
      const pct = dev.used_pct || 0;
      const cls = pct > 90 ? "vram-fill warn" : "vram-fill";
      return '<div class="vram-row">' +
        '<span class="vram-name" title="' + esc(dev.backend + ' ' + dev.name) + '">' + esc(dev.backend) + ' \u00b7 ' + esc(dev.name) + '</span>' +
        '<div class="vram-bar"><div class="' + cls + '" style="width:' + pct + '%"></div></div>' +
        '<span class="vram-info">' + dev.free + '/' + dev.total + ' MiB</span>' +
        '</div>';
    }).join("");
  } catch (e) { /* ignore */ }
}

async function refreshGpuStats() {
  try {
    const r = await fetch("/api/gpu-stats");
    const d = await r.json();
    if (!d.devices || !d.devices.length) {
      gpuStatsEl.style.display = "none";
      return;
    }
    gpuStatsEl.style.display = "";
    const cell = (label, val) =>
      '<span class="gpu-stat-cell">' + label + ': <b>' + esc(val == null ? "\u2014" : val) + '</b></span>';
    gpuStatsList.innerHTML = d.devices.map(dev =>
      '<div class="gpu-stat">' +
        '<div class="gpu-stat-name">' + esc(dev.device_name) + '</div>' +
        '<div class="gpu-stat-grid">' +
          cell("temp", dev.temp) +
          cell("fan", dev.fan_speed) +
          cell("power", dev.power_draw) +
          cell("gpu", dev.gpu_util) +
        '</div>' +
      '</div>'
    ).join("");
  } catch (e) { /* ignore */ }
}

async function refreshSystemStats() {
  try {
    const r = await fetch("/api/system-stats");
    const d = await r.json();
    if ((d.cpu_util == null) && !d.ram && !d.disk) {
      systemStatsEl.style.display = "none";
      return;
    }
    systemStatsEl.style.display = "";
    const card = (title, label, val) =>
      '<div class="gpu-stat">' +
        '<div class="gpu-stat-name">' + esc(title) + '</div>' +
        '<div class="gpu-stat-grid">' +
          '<span class="gpu-stat-cell">' + label + ': <b>' + esc(val == null ? "\u2014" : val) + '</b></span>' +
        '</div>' +
      '</div>';
    const cpu = d.cpu_util == null ? null : d.cpu_util + "%";
    const ram = d.ram
      ? Math.round(d.ram.used_pct) + "% (" + (d.ram.used_mb / 1024).toFixed(1) + "/" + (d.ram.total_mb / 1024).toFixed(1) + " GB)"
      : null;
    const disk = d.disk
      ? Math.round(d.disk.used_pct) + "% (" + d.disk.used_gb + "/" + d.disk.total_gb + " GB)"
      : null;
    systemStatsList.innerHTML =
      card("CPU", "util", cpu) +
      card("RAM", "used", ram) +
      card("Disk " + (d.disk ? d.disk.mount : ""), "used", disk);
  } catch (e) { /* server briefly unavailable; retry next tick */ }
}

refresh();
refreshVram();
refreshGpuStats();
refreshSystemStats();
setInterval(refresh, 3000);
setInterval(refreshVram, 5000);
setInterval(refreshGpuStats, 30000);
setInterval(refreshSystemStats, 5000);
</script>
</body>
</html>
"""

EDIT_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edit commands — Llama Server Manager</title>
<style>
__CSS__
  .panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
           padding: 20px; margin-bottom: 18px; }
  .panel h2 { margin: 0 0 14px; font-size: 15px; }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 12px; }
  input[type=text], input[type=password] { width: 100%; margin-top: 5px; background: var(--chip); color: var(--text);
                                           border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px;
                                           font-family: var(--mono); font-size: 13px; }
  input[type=text]:focus, input[type=password]:focus { outline: none; border-color: var(--accent); }
  .hint { font-size: 12px; color: var(--muted); margin: -6px 0 12px; }
  .check { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text);
           margin-bottom: 14px; }
  .row { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
  .msg { font-size: 13px; color: var(--green); }
  .mid { color: var(--muted); font-size: 11px; font-family: var(--mono); font-weight: 400; }
  .flags-header { font-size: 12px; font-weight: 600; color: var(--muted); margin: 14px 0 8px;
                  text-transform: uppercase; letter-spacing: .5px; }
  .flag-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .flag-row input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--accent); flex: none; }
  .flag-name { font-family: var(--mono); font-size: 12px; color: var(--accent); min-width: 50px; flex: none; }
  .flag-val { flex: 1; background: var(--chip); border: 1px solid var(--border); border-radius: 6px;
              padding: 6px 10px; color: var(--text); font-family: var(--mono); font-size: 12px; }
  .flag-val:focus { outline: none; border-color: var(--accent); }
  .flag-val:disabled { opacity: .4; }
  .positional { font-family: var(--mono); font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .cmd-wrap { position: relative; }
  .cmd-wrap label { margin-bottom: 4px; }
  .cmd-wrap input { font-family: var(--mono); font-size: 12px; }
  .binary-wrap { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
  .binary-wrap label { margin-bottom: 0; flex: 1; }
  .binary-label { font-size: 12px; color: var(--muted); white-space: nowrap; }
  .binary-wrap select { flex: 2; background: var(--chip); color: var(--text);
                        border: 1px solid var(--border); border-radius: 8px;
                        padding: 9px 12px; font-family: var(--mono); font-size: 12px; }
  .binary-wrap select:focus { outline: none; border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1><span>&#9670;</span> Llama Server Manager</h1>
  <nav><a href="/">Back to models</a><a href="/logout">Logout</a></nav>
</header>
<main>
  <section class="panel">
    <h2>New model</h2>
    <form id="newForm">
      <label>Name
        <input type="text" id="newName" placeholder="Llama 3 8B" required>
      </label>
      <label>Full command
        <input type="text" id="newCmd" placeholder="/opt/llama/llama-server -m /models/llama-3-8b.gguf -c 4096 -p 8080" required>
      </label>
      <div class="hint">Paste or type the full command. Click "Add model" and the switches will be parsed for editing.</div>
      <label class="check"><input type="checkbox" id="newBoot"> Run at boot</label>
      <button type="submit" class="run">Add model</button>
    </form>
  </section>
  <section class="panel">
    <h2>Settings</h2>
    <label>API key
      <input type="text" id="apiKey" placeholder="leave empty to disable" autocomplete="off">
    </label>
    <div class="hint">External apps send it as a header: <span style="font-family:var(--mono)">X-API-Key: &lt;key&gt;</span>. Leave empty to disable key auth.</div>
    <div class="row">
      <button type="button" class="run" id="saveKey">Save</button>
      <span class="msg" id="keyMsg"></span>
    </div>
  </section>
  <section class="panel">
    <h2>Credentials</h2>
    <label>Current password
      <input type="password" id="curPass" autocomplete="current-password">
    </label>
    <label>New username
      <input type="text" id="newUser" autocomplete="off">\n    </label>
    <label>New password
      <input type="password" id="newPass" autocomplete="new-password">
    </label>
    <label>Confirm new password
      <input type="password" id="newPass2" autocomplete="new-password">
    </label>
    <div class="hint">Requires the current password. The old login stops working after the change; this session stays open.</div>
    <div class="row">
      <button type="button" class="run" id="saveCreds">Save</button>
      <span class="msg" id="credsMsg"></span>
    </div>
  </section>
  <section id="list"></section>
</main>
<script>
const list = document.getElementById("list");
let binaries = [];

async function loadBinaries() {
  try {
    const r = await fetch("/api/binaries");
    if (r.ok) binaries = await r.json();
  } catch (e) { /* ignore */ }
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function tokenize(cmd) {
  const tokens = [];
  const re = /"([^"]*)"|'([^']*)'|(\\S+)/g;
  let m;
  while ((m = re.exec(cmd)) !== null) {
    tokens.push(m[1] !== undefined ? m[1] : (m[2] !== undefined ? m[2] : m[3]));
  }
  return tokens;
}

function parseCommand(cmd) {
  const tokens = tokenize(cmd);
  if (!tokens.length) return { binary: "", flags: [], positional: [] };
  const binary = tokens[0];
  const flags = [];
  const positional = [];
  for (let i = 1; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.startsWith("-")) {
      if (i + 1 < tokens.length && !tokens[i + 1].startsWith("-")) {
        flags.push({ name: token, value: tokens[i + 1], enabled: true });
        i++;
      } else {
        flags.push({ name: token, value: "", enabled: true });
      }
    } else {
      positional.push(token);
    }
  }
  return { binary, flags, positional };
}

function reconstructCommand(binary, flags, positional) {
  const parts = [binary];
  for (const flag of flags) {
    if (flag.enabled) {
      parts.push(flag.name);
      if (flag.value) parts.push(flag.value);
    }
  }
  parts.push(...positional);
  return parts.join(" ");
}

function renderFlagEditor(form, parsed) {
  const binaryInput = form.querySelector(".f-binary");
  const cmdInput = form.querySelector(".f-cmd");
  const flagsContainer = form.querySelector(".flags-container");
  const posContainer = form.querySelector(".positional-container");

  // Populate the Binary dropdown with the known binaries; keep the
  // current value as an extra option if it is a custom path.
  if (binaryInput && binaryInput.tagName === "SELECT") {
    const opts = binaries.slice();
    if (parsed.binary && !opts.includes(parsed.binary)) opts.push(parsed.binary);
    binaryInput.innerHTML = opts.map(b =>
      '<option value="' + esc(b) + '"' + (b === parsed.binary ? " selected" : "") + ">" + esc(b) + "</option>"
    ).join("");
  }

  function renderFlags() {
    let html = "";
    for (let i = 0; i < parsed.flags.length; i++) {
      const f = parsed.flags[i];
      html += '<div class="flag-row">' +
        '<input type="checkbox" class="f-flag-en" data-idx="' + i + '"' +
          (f.enabled ? " checked" : "") + '>' +
        '<span class="flag-name">' + esc(f.name) + '</span>' +
        '<input type="text" class="f-flag-val" data-idx="' + i + '"' +
          ' value="' + esc(f.value) + '"' +
          (f.enabled ? "" : " disabled") + '>' +
        '</div>';
    }
    flagsContainer.innerHTML = html || '<div style="color:var(--muted);font-size:12px">No flags detected</div>';
  }

  function updateCmd() {
    cmdInput.value = reconstructCommand(binaryInput.value, parsed.flags, parsed.positional);
  }

  // Flag checkbox change (event delegation on container)
  flagsContainer.addEventListener("change", (e) => {
    if (e.target.classList.contains("f-flag-en")) {
      const idx = parseInt(e.target.dataset.idx);
      parsed.flags[idx].enabled = e.target.checked;
      const valInput = form.querySelector('.f-flag-val[data-idx="' + idx + '"]');
      if (valInput) valInput.disabled = !e.target.checked;
      updateCmd();
    }
  });

  // Flag value change (event delegation on container)
  flagsContainer.addEventListener("input", (e) => {
    if (e.target.classList.contains("f-flag-val")) {
      const idx = parseInt(e.target.dataset.idx);
      parsed.flags[idx].value = e.target.value;
      updateCmd();
    }
  });

  // Binary change
  binaryInput.addEventListener("change", () => {
    updateCmd();
  });

  // Command manual edit → re-parse
  cmdInput.addEventListener("input", () => {
    const reParsed = parseCommand(cmdInput.value);
    parsed.binary = reParsed.binary;
    parsed.flags = reParsed.flags;
    parsed.positional = reParsed.positional;
    if (binaryInput && reParsed.binary) {
      if (![...binaryInput.options].some(o => o.value === reParsed.binary)) {
        const o = document.createElement("option");
        o.value = reParsed.binary;
        o.textContent = reParsed.binary;
        binaryInput.appendChild(o);
      }
      binaryInput.value = reParsed.binary;
    }
    renderFlags();
  });

  // Initial render
  renderFlags();
}

function render(models) {
  list.innerHTML = "";
  if (!models.length) {
    list.innerHTML = '<div class="panel" style="color:var(--muted);font-size:14px">No models defined yet.</div>';
    return;
  }
  for (const m of models) {
    const parsed = parseCommand(m.command);
    const card = document.createElement("div");
    card.className = "panel";
    card.innerHTML =
      '<h2>' + esc(m.name) + ' <span class="mid">' + m.id + "</span></h2>" +
      '<form class="edit-form">' +
        '<label>Name <input type="text" class="f-name" value="' + esc(m.name) + '"></label>' +
        '<div class="binary-wrap">' +
          '<label class="binary-label">Binary</label>' +
          '<select class="f-binary"></select>' +
        '</div>' +
        '<div class="flags-header">Parsed switches</div>' +
        '<div class="flags-container"></div>' +
        '<div class="positional-container"></div>' +
        '<div class="cmd-wrap">' +
          '<label>Full command (auto-updates from switches above)</label>' +
          '<input type="text" class="f-cmd" value="' + esc(m.command) + '">' +
        '</div>' +
        '<label class="check"><input type="checkbox" class="f-boot"' +
          (m.run_on_boot ? " checked" : "") + '> Run at boot</label>' +
        '<div class="row">' +
          '<button type="submit">Save</button>' +
          '<button type="button" class="stop del">Delete</button>' +
          '<span class="msg"></span>' +
        "</div>" +
      "</form>";
    renderFlagEditor(card, parsed);
    const form = card.querySelector("form");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const r = await fetch("/api/models/" + m.id, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: form.querySelector(".f-name").value,
          command: form.querySelector(".f-cmd").value,
          run_on_boot: form.querySelector(".f-boot").checked
        })
      });
      const msg = form.querySelector(".msg");
      msg.textContent = r.ok ? "Saved" : "Error: " + ((await r.json()).error || r.status);
      setTimeout(refresh, 400);
    });
    card.querySelector(".del").addEventListener("click", async () => {
      if (!confirm("Delete '" + m.name + "'?")) return;
      await fetch("/api/models/" + m.id, { method: "DELETE" });
      refresh();
    });
    list.appendChild(card);
  }
}

document.getElementById("newForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const r = await fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: document.getElementById("newName").value,
      command: document.getElementById("newCmd").value,
      run_on_boot: document.getElementById("newBoot").checked
    })
  });
  if (r.ok) e.target.reset();
  refresh();
});

const apiKeyInput = document.getElementById("apiKey");
const keyMsg = document.getElementById("keyMsg");

async function loadConfig() {
  try {
    const r = await fetch("/api/config");
    if (r.ok) apiKeyInput.value = (await r.json()).api_key || "";
  } catch (e) { /* ignore */ }
}

document.getElementById("saveKey").addEventListener("click", async () => {
  const r = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKeyInput.value.trim() })
  });
  keyMsg.textContent = r.ok ? "Saved" : "Error: " + r.status;
  setTimeout(() => { keyMsg.textContent = ""; }, 2500);
});

document.getElementById("saveCreds").addEventListener("click", async () => {
  const credsMsg = document.getElementById("credsMsg");
  const newPass = document.getElementById("newPass").value;
  if (newPass !== document.getElementById("newPass2").value) {
    credsMsg.textContent = "New passwords do not match";
    return;
  }
  const r = await fetch("/api/auth", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: document.getElementById("curPass").value,
      username: document.getElementById("newUser").value.trim(),
      password: newPass
    })
  });
  const d = await r.json();
  credsMsg.textContent = r.ok ? "Saved" : "Error: " + (d.error || r.status);
  if (r.ok) ["curPass", "newUser", "newPass", "newPass2"]
    .forEach(id => document.getElementById(id).value = "");
  setTimeout(() => { credsMsg.textContent = ""; }, 2500);
});

async function refresh() {
  const r = await fetch("/api/models");
  if (r.status === 401) { window.location.href = "/login"; return; }
  render(await r.json());
}

loadConfig();
loadBinaries().then(refresh);
</script>
</body>
</html>
"""


LOGIN_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login — Llama Server Manager</title>
<style>
__CSS__
  .login-wrap { display: flex; align-items: center; justify-content: center; min-height: 70vh; }
  .login-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
                padding: 28px; width: 340px; }
  .login-card h1 { font-size: 16px; margin: 0 0 4px; }
  .login-card p.sub { color: var(--muted); font-size: 13px; margin: 0 0 18px; }
  .login-card label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 12px; }
  .login-card input { width: 100%; margin-top: 5px; background: var(--chip); color: var(--text);
                      border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px;
                      font-size: 14px; }
  .login-card input:focus { outline: none; border-color: var(--accent); }
  .login-card button { width: 100%; margin-top: 6px; }
  .error { color: var(--red); font-size: 13px; margin-bottom: 12px; }
  .error.fatal { color: var(--red); font-weight: 600; }
</style>
</head>
<body>
<div class="login-wrap">
  <form class="login-card" method="post" action="/login">
    <h1><span style="color:var(--accent)">&#9670;</span> Llama Server Manager</h1>
    <p class="sub">__SUBTITLE__</p>
    __ERROR__
    <label>Username
      <input type="text" name="username" autocomplete="username" required autofocus>
    </label>
    <label>Password
      <input type="password" name="password" autocomplete="new-password" required>
    </label>
    __CONFIRM__
    <button type="submit" class="run">__BUTTON__</button>
  </form>
</div>
</body>
</html>
"""


@app.get("/")
def index():
    return INDEX_HTML.replace("__CSS__", SHARED_CSS)


@app.get("/edit")
def edit():
    return EDIT_HTML.replace("__CSS__", SHARED_CSS)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    recover_pids()
    boot_models()
    print(f"Llama Server Manager listening on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SessionEnd hook: capture what the session learned, then keep the store healthy.

Two speeds. The cheap deterministic work runs inline, because it must never be
skipped: MEMORY.md is re-compacted so it cannot drift back over the loader's
undocumented 200-line / 25,000-byte cap and start silently dropping rules again.

Capture runs detached because it usually finishes inside the hook's window.
Everything slower lives in maintain.py under launchd, because a detached child
of a hook is killed when the hook exits and that silently stopped every other
job for nine days.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(HERE)
MEMDIR = os.path.join(ROOT, "src")
if not os.path.isdir(MEMDIR):
    # Nine days of silence came from this path resolving to a directory that
    # did not exist; Popen(cwd=...) threw, the outer handler swallowed it, and
    # the hook exited 0 having done nothing. Never let it be wrong quietly.
    sys.stderr.write("memory-session-end: cannot find src at %s\n" % MEMDIR)
    sys.exit(0)
sys.path.insert(0, MEMDIR)
from memlib import LOG_DIR as LOGDIR, STATE_ROOT as STATE  # noqa: E402
STAMP = os.path.join(STATE, "last-maintenance")

CURATE_EVERY = 20 * 3600     # once a day is plenty for contradiction sweeps
MIN_TRANSCRIPT_BYTES = 4000  # a session too short to have learned anything


def due_for_curate():
    try:
        return time.time() - os.path.getmtime(STAMP) > CURATE_EVERY
    except OSError:
        return True


def touch_stamp():
    try:
        os.makedirs(STATE, exist_ok=True)
        open(STAMP, "w").write(str(time.time()))
    except OSError:
        pass


def detach(argv, logname):
    """Spawn fully detached so the session can close immediately."""
    os.makedirs(LOGDIR, exist_ok=True)
    out = open(os.path.join(LOGDIR, logname), "a")
    env = dict(os.environ)
    env.pop("QUIPU_CHILD", None)
    subprocess.Popen(
        argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, cwd=MEMDIR, env=env,
    )


def main():
    # `claude -p` fires SessionEnd too. Without this, capture would capture
    # itself, forever.
    if os.environ.get("QUIPU_CHILD"):
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    py = sys.executable

    # Inline: the store must never be left over the loader's cap.
    try:
        subprocess.run([py, os.path.join(MEMDIR, "curate.py"), "compact"],
                       capture_output=True, timeout=30, cwd=MEMDIR)
    except Exception:
        pass

    # Fold the session just finished into the searchable history. Deterministic,
    # no model, ~8s for 200 sessions and far less for one.
    try:
        subprocess.run([py, os.path.join(MEMDIR, "sessions.py"), "index", "--since", "2"],
                       capture_output=True, timeout=60, cwd=MEMDIR)
    except Exception:
        pass

    transcript = payload.get("transcript_path") or ""
    session_id = payload.get("session_id") or ""
    big_enough = False
    try:
        big_enough = os.path.getsize(transcript) >= MIN_TRANSCRIPT_BYTES
    except OSError:
        pass

    if transcript and big_enough:
        detach([py, os.path.join(MEMDIR, "capture.py"), "session",
                "--transcript", transcript, "--session-id", session_id],
               "memory-capture.out")

    # Everything else moved to maintain.py under launchd. A detached child of a
    # hook dies when the hook exits, and for nine days every one of these jobs
    # did exactly that: enrich, verify, profile and skills each ran once,
    # on the day they were written, and never again. The hook now only does what
    # finishes inside its own window and leaves a marker so maintain runs soon.
    try:
        os.makedirs(STATE, exist_ok=True)
        open(os.path.join(STATE, "session-ended"), "w").write(str(time.time()))
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)

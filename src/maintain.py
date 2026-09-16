#!/usr/bin/env python3
"""
maintain.py - the maintenance driver. Runs from launchd, never from a hook.

WHY THIS EXISTS. For nine days every background job was dead and nobody noticed.
The session-end hook launched them as detached children, and Claude Code kills
the hook's process group when the hook exits, so enrich, verify, profile and
skill detection each ran exactly once, on the day they were built. Meanwhile
100 new memories went unenriched and the store looked healthy in every
`mem stats`. Capture survived only because it usually finishes inside the
hook's 45-second window.

The fix is to stop asking a hook to outlive itself. launchd runs this every
few minutes as a real daemon, and this decides what is due:

  every run     enrich anything new, refresh session search
  daily         verify claims, curate contradictions, rebuild the profile
  weekly        skill detection

Each job is guarded by a stamp so it cannot run twice in its window, and the
whole thing takes a lock so two launchd fires cannot overlap. Everything here
is idempotent: running it ten times in a row does nothing the tenth time.

Usage:
  maintain.py run [--force-all]
  maintain.py status
"""

import argparse
import fcntl
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memlib import STATE_ROOT, log_line  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STAMPS = os.path.join(STATE_ROOT, "maintain")
LOCK = os.path.join(STATE_ROOT, "maintain.lock")

HOUR, DAY = 3600, 86400
JOBS = [
    # (name, argv, minimum seconds between runs, timeout seconds)
    ("sessions", ["sessions.py", "index", "--since", "3"],        10 * 60,   120),
    # Meaning vectors for new conversation turns and changed memories. Static
    # embeddings: 71,062 turns took 34s at nice 15 and peaked at 102 MB, so the
    # incremental run every ten minutes is a fraction of a second.
    ("embed",    ["sessions.py", "embed"],                         10 * 60,   300),
    ("embeddoc", ["recall.py", "embed"],                           10 * 60,   120),
    ("enrich",   ["enrich.py", "run", "--batch", "8", "--workers", "3"], 20 * 60, 20 * 60),
    ("verify",   ["verify.py", "run"],                             DAY,       15 * 60),
    ("curate",   ["curate.py", "run", "--limit", "200"],           DAY,       30 * 60),
    ("profile",  ["profile.py", "build"],                          DAY,       10 * 60),
    ("skills",   ["skills.py", "detect", "--sessions", "60"],      7 * DAY,   20 * 60),
]


def stamp_path(name):
    return os.path.join(STAMPS, name)


def last_ran(name):
    try:
        return os.path.getmtime(stamp_path(name))
    except OSError:
        return 0.0


def touch(name):
    os.makedirs(STAMPS, exist_ok=True)
    with open(stamp_path(name), "w") as fh:
        fh.write(str(int(time.time())))


def run_job(name, argv, timeout):
    env = dict(os.environ)
    env["QUIPU_CHILD"] = "1"          # its claude -p calls must not recurse
    t0 = time.time()
    try:
        target = os.path.join(HERE, argv[0])
        runner = [PY, target]
        p = subprocess.run(runner + argv[1:],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=HERE, env=env)
        ok = p.returncode == 0
        tail = (p.stdout or p.stderr or "").strip().splitlines()
        note = tail[-1][:120] if tail else ""
    except subprocess.TimeoutExpired:
        ok, note = False, "timed out after %ds" % timeout
    except Exception as exc:
        ok, note = False, str(exc)[:120]
    log_line("memory-maintain.log", name, "ok" if ok else "FAIL",
             "%.0fs" % (time.time() - t0), note)
    return ok, note


def run(force_all=False, verbose=True):
    os.makedirs(STATE_ROOT, exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        if verbose:
            print("another maintain run is in progress")
        return 0

    now = time.time()
    ran = []
    for name, argv, every, timeout in JOBS:
        if not force_all and now - last_ran(name) < every:
            continue
        ok, note = run_job(name, argv, timeout)
        touch(name)                         # even on failure: no hot retry loop
        ran.append((name, ok, note))
        if verbose:
            print("  %-9s %s  %s" % (name, "ok " if ok else "FAIL", note))

    # Enrichment and re-indexing change what recall sees; fold it in once.
    if any(n in ("enrich", "curate") for n, _, _ in ran):
        run_job("index", ["recall.py", "index"], 120)

    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    if verbose and not ran:
        print("nothing due")
    return 0


def status():
    now = time.time()
    print("  %-9s %-18s %s" % ("job", "last ran", "next due"))
    for name, _argv, every, _t in JOBS:
        last = last_ran(name)
        when = time.strftime("%m-%d %H:%M", time.localtime(last)) if last else "never"
        due = "now" if now - last >= every else "in %.0fm" % ((every - (now - last)) / 60)
        print("  %-9s %-18s %s" % (name, when, due))


def main():
    ap = argparse.ArgumentParser(prog="maintain.py")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("--force-all", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "run":
        return run(force_all=args.force_all)
    if args.cmd == "status":
        status()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

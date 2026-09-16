#!/usr/bin/env python3
"""
PreCompact hook: capture the part of a long session that is about to be deleted.

`capture.py` runs at SessionEnd and reads the transcript tail. That is fine for
a short session and wrong for a long one, because the window gets compacted
first: the middle of a three-hour session is summarised away and the detail is
gone before anything ever looked at it for memories. The hardest-won facts tend
to live exactly there, in the middle, after the false starts and before the
writeup.

This fires on the boundary instead. `matcher: "auto"` catches automatic
compaction, which is the one nobody notices; `manual` catches /compact.

It is deliberately fire-and-forget: compaction must not wait for a model. The
capture runs detached and its proposals land in the review queue like any other.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
LOGDIR = os.path.expanduser(os.environ.get("QUIPU_LOGS", "~/.claude/quipu/logs"))

# A window worth mining. Below this there is nothing in the middle to lose.
MIN_BYTES = 200_000


def main():
    if os.environ.get("QUIPU_CHILD"):
        return 0
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    transcript = payload.get("transcript_path") or ""
    session_id = payload.get("session_id") or ""
    try:
        if os.path.getsize(transcript) < MIN_BYTES:
            return 0
    except OSError:
        return 0

    os.makedirs(LOGDIR, exist_ok=True)
    out = open(os.path.join(LOGDIR, "memory-precompact.out"), "a")
    env = dict(os.environ)
    env.pop("QUIPU_CHILD", None)
    try:
        subprocess.Popen(
            [sys.executable, os.path.join(SRC, "capture.py"), "session",
             "--transcript", transcript,
             "--session-id", (session_id or "precompact") + "-precompact"],
            stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=SRC, env=env)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)

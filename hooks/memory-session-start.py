#!/usr/bin/env python3
"""
SessionStart hook: put the user profile in context before anything is typed.

This is the only part of the memory system that is always on. Everything else
waits to be asked. It is deliberately ~350 tokens, because a line here costs
tokens on every turn of every session forever, and the index file next to it
already costs 6,000.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


def main():
    if os.environ.get("QUIPU_CHILD"):
        return 0
    try:
        json.load(sys.stdin)
    except Exception:
        pass
    parts = []
    try:
        import profile
        parts.append(profile.context_block())
    except Exception:
        pass

    # A self-writing agent that writes silently is how you end up with fifty
    # skills nobody approved. Every inert draft is announced every session until
    # the user promotes or rejects it.
    try:
        import skills
        waiting = skills.pending()
        if waiting:
            lines = ["<skill-drafts-awaiting-review>",
                     "A procedure that keeps recurring was drafted as a skill. Drafts are "
                     "INERT: they affect nothing until the user promotes them. Mention "
                     "them to the user at a natural moment.", ""]
            for n in waiting:
                lines.append("- %s  (promote: `mem skill promote %s`  reject: `mem skill reject %s`)"
                             % (n, n, n))
            lines.append("</skill-drafts-awaiting-review>")
            parts.append("\n".join(lines))
    except Exception:
        pass

    block = "\n\n".join(x for x in parts if x)
    if not block:
        return 0
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "SessionStart", "additionalContext": block}}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)

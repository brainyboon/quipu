#!/usr/bin/env python3
"""
skills.py - notice a procedure that keeps recurring, and write it down.

READ THIS BEFORE CHANGING THE GATE. The measured result on self-authored skills
is that they make an agent WORSE. On SkillsBench (87 tasks, 11 domains,
deterministic verifiers) agent-self-generated skills scored BELOW the no-skills
baseline on every harness tested (-8.1pp on Claude Code + Opus, -11.3pp
elsewhere), while human-curated skills scored +16.6pp. And an LLM asked to judge
which of two skills is better picks the WORSE one 84.2% of the time when there
is a real gap between them (arXiv 2602.12670, 2605.23899).

So the naive version of this feature is a machine for getting slowly worse, and
the part that matters is not detection or authoring. Both are cheap and close to
solved. The part that matters is the ADMISSION GATE.

The design that follows from the evidence:

  1. DETECT from repetition, not from vibes. A skill is proposed when the same
     procedure shows up across three or more separate sessions. We have 67,704
     indexed turns to mine, which most systems in this literature do not.
  2. WRITE INERT. Every draft ships with `disable-model-invocation: true`, so its
     description never enters anyone's context and it cannot influence a single
     turn until it is admitted.
  3. KEEP IT SMALL. Compact skills measured +21.5pp; comprehensive ones +0.7pp,
     which is nothing. The body is capped and the cap is enforced at write time.
  4. STRUCTURAL ADMISSION, NOT PROSE JUDGEMENT. Because LLM judges pick the worse
     skill most of the time, nothing here asks a model "is this good". A draft
     must name a concrete failure it prevents and cite real paths or commands
     that exist on this machine. Those are checkable.
  5. A HUMAN ADMITS IT. Human-curated is the only condition measured positive, so
     promotion is the user's, never the agent's. They are told about every draft.
  6. BOUNDED LIFETIME EDITS. A skill that wants a fifth revision is a skill that
     should be split or retired, not patched again.
  7. NEVER SHADOW A BUILT-IN. Anthropic's own guidance: a new project skill
     silently shadows a same-named built-in. Name collisions are refused.

Usage:
  skills.py detect [--sessions N] [--model haiku]
  skills.py pending                 drafts waiting for the user
  skills.py show <name>
  skills.py promote <name>          the user admits it: the inert flag comes off
  skills.py reject <name> [--why]
  skills.py stats
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sessions as sessions_mod  # noqa: E402
from memlib import (  # noqa: E402
    LLMError, MemoryLock, audit, claude_json, connect, log_line,
)

DRAFT_DIR = os.path.expanduser("~/.claude/skills/_drafts")
LIVE_DIR = os.path.expanduser("~/.claude/skills")
MAX_BODY_CHARS = 7000          # ~1,900 tokens: the measured sweet spot
MIN_SESSIONS = 3               # a procedure is not a pattern until it recurs
MAX_EDITS = 4

# Names that already mean something. A same-named skill silently shadows them.
RESERVED = {
    "memory", "verify", "review", "security-review", "init", "loop", "schedule",
    "simplify", "compact", "config", "help", "clear", "model", "agents", "hooks",
    "mcp", "permissions", "plugin", "resume", "status", "doctor", "export", "cost",
    "context", "skills", "run",
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["skills"],
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "description", "prevents", "evidence",
                             "body", "sessions_seen"],
                "properties": {
                    "name": {"type": "string", "description": "short-kebab-case"},
                    "description": {"type": "string", "description": "under 200 chars, says WHEN to use it"},
                    "prevents": {"type": "string", "description": "the concrete failure this stops. Not 'saves time'."},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "real paths, commands, flags or URLs the procedure touches, verbatim from the transcripts",
                    },
                    "body": {"type": "string", "description": "the procedure. Steps and commands. No preamble."},
                    "sessions_seen": {"type": "integer", "description": "how many DISTINCT sessions you saw this in"},
                },
            },
        }
    },
}

SYSTEM = (
    "You find procedures an engineer has repeated across many sessions and turn "
    "them into short, checkable skills. You are extremely reluctant: most weeks "
    "produce no skill at all. You output JSON only."
)

PROMPT = """Below are excerpts from %d separate work sessions.

Find PROCEDURES that were repeated across at least %d DIFFERENT sessions and are
worth writing down once so nobody rediscovers them.

A procedure qualifies only if ALL of these hold:
- it recurred across %d or more distinct sessions
- getting it wrong has a concrete cost: a broken deploy, lost data, wasted hours,
  a wrong answer stated confidently
- it involves specific, nameable things: real commands, flags, file paths, URLs
- it is NOT already obvious to a competent engineer reading the code

DO NOT propose a skill for any of these:
- a one-off task, however involved
- anything already recorded in the memory store or a README
- general engineering advice with no local specifics
- a thing that changes often enough that a written procedure would go stale
- restating what a tool's own help output says

For each qualifying procedure return:
  name          short-kebab-case, must not collide with a common command name
  description   under 200 chars, saying WHEN to reach for it
  prevents      the specific failure this stops. "Saves time" is not an answer.
                Name the thing that goes wrong without it.
  evidence      real paths / commands / flags / URLs, VERBATIM from the sessions.
                This is checked against the filesystem. Invented paths are rejected.
  body          the procedure. Steps, commands, gotchas. Under %d characters.
                Short and sharp beats thorough: measured, comprehensive skill
                documentation is worth almost nothing while compact is worth a lot.
  sessions_seen how many distinct sessions you actually saw this in

Return {"skills": []} if nothing qualifies. That is the common and correct answer.

SESSIONS:
%s
"""


def slug_ok(name):
    return bool(re.match(r"^[a-z][a-z0-9-]{2,40}$", name or ""))


def evidence_is_real(items):
    """A path or command the draft cites must actually exist here.

    This replaces asking a model whether a skill is good, which measures worse
    than chance. Whether `~/project/src/recall.py` exists is not a matter of
    opinion.
    """
    checked, real = 0, 0
    for item in items or []:
        item = str(item).strip().strip("`")
        if item.startswith("/") and " " not in item:
            checked += 1
            real += 1 if os.path.exists(item) else 0
        elif re.match(r"^[a-z][a-z0-9_.-]*$", item.split()[0] if item.split() else ""):
            cmd = item.split()[0]
            checked += 1
            real += 1 if subprocess.run(["which", cmd], capture_output=True).returncode == 0 else 0
    if not checked:
        return False, "no checkable path or command in the evidence"
    ratio = real / float(checked)
    return ratio >= 0.5, "%d of %d cited paths/commands exist" % (real, checked)


def gate(draft):
    """Structural admission checks. Every one is objectively decidable."""
    fails = []
    if not slug_ok(draft.get("name")):
        fails.append("name is not short-kebab-case")
    if draft.get("name") in RESERVED:
        fails.append("name collides with an existing skill and would shadow it")
    if os.path.isdir(os.path.join(LIVE_DIR, draft.get("name", ""))):
        fails.append("a live skill already owns this name")
    if len(draft.get("description", "")) > 200:
        fails.append("description over 200 chars")
    if len(draft.get("body", "")) > MAX_BODY_CHARS:
        fails.append("body over %d chars: compact skills measure far better" % MAX_BODY_CHARS)
    if int(draft.get("sessions_seen") or 0) < MIN_SESSIONS:
        fails.append("seen in fewer than %d sessions, so it is not a pattern yet" % MIN_SESSIONS)
    prevents = draft.get("prevents", "")
    if len(prevents) < 25 or re.search(r"(?i)saves? time|faster|convenien|easier", prevents):
        fails.append("'prevents' does not name a concrete failure")
    ok, note = evidence_is_real(draft.get("evidence"))
    if not ok:
        fails.append("evidence does not check out: %s" % note)
    return fails, note


def write_draft(draft, note):
    """Write it INERT. Its description never enters context until promoted."""
    name = draft["name"]
    d = os.path.join(DRAFT_DIR, name)
    os.makedirs(d, exist_ok=True)
    body = draft["body"].strip()[:MAX_BODY_CHARS]
    text = (
        "---\n"
        "name: %s\n"
        "description: %s\n"
        "disable-model-invocation: true\n"
        "metadata:\n"
        "  source: autonomous-draft\n"
        "  drafted: %s\n"
        "  sessions_seen: %d\n"
        "  edits: 0\n"
        "---\n\n"
        "<!-- DRAFT. Inert: `disable-model-invocation: true` keeps this out of\n"
        "     context entirely until the user promotes it. Self-authored skills measure\n"
        "     BELOW no-skills at all until a human admits them, so this waits.\n"
        "     Promote: skills.py promote %s      Reject: skills.py reject %s -->\n\n"
        "**Prevents:** %s\n\n"
        "**Evidence checked:** %s\n\n"
        "%s\n"
        % (name, draft["description"].replace("\n", " ")[:200],
           time.strftime("%Y-%m-%d"), int(draft.get("sessions_seen") or 0),
           name, name, draft.get("prevents", ""), note, body)
    )
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return os.path.join(d, "SKILL.md")


def session_material(n=40, per_session=1800):
    con = connect()
    con.executescript(sessions_mod.SCHEMA)
    rows = con.execute(
        "SELECT id, opening FROM session WHERE turns > 12 ORDER BY ended DESC LIMIT ?",
        (n,)).fetchall()
    blocks = []
    for sid, opening in rows:
        turns = con.execute(
            "SELECT role, text FROM turn WHERE sid=? ORDER BY seq LIMIT 40", (sid,)).fetchall()
        body = " ".join("%s: %s" % (r[:1].upper(), t[:400]) for r, t in turns)
        blocks.append("--- SESSION %s ---\n%s\n" % (sid[:8], body[:per_session]))
    con.close()
    return "\n".join(blocks), len(rows)


def detect(n_sessions=40, model="haiku", verbose=True):
    material, n = session_material(n_sessions)
    if not n:
        print("no indexed sessions to mine. Run: sessions.py index")
        return []
    if verbose:
        print("mining %d sessions for repeated procedures" % n)
    try:
        data = claude_json(PROMPT % (n, MIN_SESSIONS, MIN_SESSIONS, MAX_BODY_CHARS, material),
                           model=model, timeout=600, system=SYSTEM)
    except LLMError as exc:
        print("detect failed: %s" % exc)
        return []

    drafts = (data or {}).get("skills") or []
    if not drafts:
        if verbose:
            print("nothing recurred enough to be worth a skill. This is the normal answer.")
        log_line("memory-skills.log", "DETECT", "sessions=%d" % n, "drafts=0")
        return []

    written, rejected = [], []
    with MemoryLock():
        con = connect()
        for d in drafts:
            fails, note = gate(d)
            if fails:
                rejected.append((d.get("name", "?"), fails))
                audit(con, "skill-gated", d.get("name", "?"), "; ".join(fails)[:200], "skills")
                continue
            path = write_draft(d, note)
            written.append((d["name"], path, d))
            audit(con, "skill-drafted", d["name"],
                  "%s | seen in %s sessions | %s" % (d["description"][:80],
                                                     d.get("sessions_seen"), note), "skills")
            con.execute(
                "INSERT INTO review(ts, kind, rels, detail, status) VALUES (?,?,?,?, 'open')",
                (time.time(), "skill-draft", d["name"],
                 "INERT draft. prevents: %s | %s | promote with: skills.py promote %s"
                 % (d.get("prevents", "")[:120], note, d["name"])))
        con.commit()
        con.close()

    log_line("memory-skills.log", "DETECT", "sessions=%d" % n,
             "drafted=%d" % len(written), "gated=%d" % len(rejected))
    if verbose:
        for name, fails in rejected:
            print("  gated   %-28s %s" % (name, fails[0]))
        for name, path, d in written:
            print("  DRAFTED %-28s seen in %s sessions" % (name, d.get("sessions_seen")))
            print("          prevents: %s" % d.get("prevents", "")[:90])
            print("          %s" % path)
    return written


def pending():
    if not os.path.isdir(DRAFT_DIR):
        return []
    return sorted(d for d in os.listdir(DRAFT_DIR)
                  if os.path.exists(os.path.join(DRAFT_DIR, d, "SKILL.md")))


def promote(name):
    src = os.path.join(DRAFT_DIR, name, "SKILL.md")
    if not os.path.exists(src):
        print("no draft called %s" % name)
        return 1
    if name in RESERVED or os.path.isdir(os.path.join(LIVE_DIR, name)):
        print("refusing: %s would shadow an existing skill" % name)
        return 1
    dst_dir = os.path.join(LIVE_DIR, name)
    os.makedirs(dst_dir, exist_ok=True)
    text = open(src, encoding="utf-8").read()
    text = re.sub(r"^disable-model-invocation: true\n", "", text, flags=re.M)
    text = re.sub(r"(?s)<!-- DRAFT\..*?-->\n\n", "", text)
    text = text.replace("source: autonomous-draft", "source: autonomous-promoted-by-user")
    with open(os.path.join(dst_dir, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    import shutil
    shutil.rmtree(os.path.join(DRAFT_DIR, name), ignore_errors=True)
    with MemoryLock():
        con = connect()
        audit(con, "skill-promoted", name, "admitted by the user; inert flag removed", "user")
        con.execute("UPDATE review SET status='resolved' WHERE kind='skill-draft' AND rels=?", (name,))
        con.commit()
        con.close()
    print("promoted: %s is now live at %s" % (name, dst_dir))
    return 0


def reject(name, why=""):
    import shutil
    d = os.path.join(DRAFT_DIR, name)
    if not os.path.isdir(d):
        print("no draft called %s" % name)
        return 1
    shutil.rmtree(d, ignore_errors=True)
    with MemoryLock():
        con = connect()
        audit(con, "skill-rejected", name, why or "rejected by the user", "user")
        con.execute("UPDATE review SET status='dismissed' WHERE kind='skill-draft' AND rels=?", (name,))
        con.commit()
        con.close()
    print("rejected and removed: %s" % name)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="skills.py")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("detect")
    p.add_argument("--sessions", type=int, default=40)
    p.add_argument("--model", default="haiku")
    sub.add_parser("pending")
    p = sub.add_parser("show"); p.add_argument("name")
    p = sub.add_parser("promote"); p.add_argument("name")
    p = sub.add_parser("reject"); p.add_argument("name"); p.add_argument("--why", default="")
    sub.add_parser("stats")
    args = ap.parse_args()

    if args.cmd == "detect":
        detect(n_sessions=args.sessions, model=args.model)
        return 0
    if args.cmd == "pending":
        names = pending()
        if not names:
            print("no drafts waiting")
        for n in names:
            head = open(os.path.join(DRAFT_DIR, n, "SKILL.md"), encoding="utf-8").read()
            desc = re.search(r"^description: (.+)$", head, re.M)
            print("  %-28s %s" % (n, (desc.group(1) if desc else "")[:80]))
        return 0
    if args.cmd == "show":
        p = os.path.join(DRAFT_DIR, args.name, "SKILL.md")
        if not os.path.exists(p):
            p = os.path.join(LIVE_DIR, args.name, "SKILL.md")
        if not os.path.exists(p):
            print("not found: %s" % args.name)
            return 1
        print(open(p, encoding="utf-8").read())
        return 0
    if args.cmd == "promote":
        return promote(args.name)
    if args.cmd == "reject":
        return reject(args.name, args.why)
    if args.cmd == "stats":
        con = connect()
        for action in ("skill-drafted", "skill-gated", "skill-promoted", "skill-rejected"):
            n = con.execute("SELECT count(*) FROM audit WHERE action=?", (action,)).fetchone()[0]
            print("  %-16s %d" % (action.replace("skill-", ""), n))
        con.close()
        print("  %-16s %d waiting for the user" % ("pending", len(pending())))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

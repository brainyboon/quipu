#!/usr/bin/env python3
"""
capture.py - turn a finished session into memory, without being asked.

Until now a memory existed only if the model decided mid-task to stop and write
one. That is the weakest link in the whole system: the best material shows up at
the end of a long debugging session, exactly when nobody stops to write notes.
This runs at SessionEnd, reads what actually happened, and proposes what is
worth keeping.

Conservative on purpose:
  - at most 3 new memories per session
  - it is shown the existing memories on the same subject first, so it updates
    the index instead of writing a fourth note about the same fact
  - an update to an existing memory is NEVER applied automatically, it goes to
    the review queue; only genuinely new files are written
  - every captured file records the session it came from, so a bad memory can
    be traced back
  - runs under the store lock, because several sessions can end at once

Usage:
  capture.py session --transcript <path> [--session-id X] [--dry-run]
  capture.py pending          show what capture has proposed for review
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recall  # noqa: E402
from memlib import (  # noqa: E402
    MEM_ROOT, LLMError, MemoryLock, audit, claude_json, connect, log_line,
    read_doc, walk_memory,
)

MAX_NEW = 3
MIN_TURNS = 4
MIN_CHARS = 1200
SESSION_CHARS = 14000

SYSTEM = (
    "You maintain a long-term memory store for one user. You record durable "
    "facts about people and organisations, and above all when those facts change. "
    "You are extremely selective about everything else: most sessions produce no "
    "memory at all, and you never restate what code or git history already says. "
    "You output JSON only."
)
MAX_FACTS = 5

# What a fact may be about. "work" (the default) keeps to what a coding assistant
# needs. "personal" adds family, health and money, which MemConflict asks about;
# the benchmark adapter turns it on, and nothing else should without consent.
FACT_SCOPE = os.environ.get("QUIPU_FACTS_SCOPE", "work").lower()
SCOPES = {
    "work": {
        "@@FACT_WHO@@": "clients, teammates, companies and vendors",
        "@@FACT_ATTR@@": "team",
        "@@FACT_SCOPE@@": (
            "Worth recording: where someone is based and their time zone, employer,\n"
            "job title, team and role, what they use or own for work, standing\n"
            "preferences and dislikes about how work is done, plans and commitments\n"
            "with dates, budgets, prices, deadlines, who is responsible for what.\n\n"
            "Never record health, relationships, family, religion, politics, income\n"
            "or other private life details, even when the conversation mentions them."),
    },
    "personal": {
        "@@FACT_WHO@@": "clients,\nteammates, family, friends, companies and vendors",
        "@@FACT_ATTR@@": "marital-status",
        "@@FACT_SCOPE@@": (
            "Worth recording: where someone lives or is based, employer, job title,\n"
            "industry, income or savings band, relationship and family status, children,\n"
            "health status, what they use or own, standing preferences and dislikes, plans\n"
            "and commitments with dates, budgets, prices, deadlines, who is responsible for\n"
            "what."),
    },
}


def scoped_prompt():
    text = PROMPT
    for key, value in SCOPES.get(FACT_SCOPE, SCOPES["work"]).items():
        text = text.replace(key, value)
    return text

PROMPT = """Below is a conversation, the facts the store already holds, and the
memories that already exist on related subjects.

Return ONE JSON object with two arrays, "facts" and "memories". Either may be
empty, and "memories" usually is.

== FACTS ==
A fact is a durable thing about a PERSON or an ORGANISATION that someone is
likely to ask about later. The user counts (the person the assistant works
for; call them "user" if the conversation never names them), and so do
@@FACT_WHO@@.

Record a fact when this conversation STATES it for the first time, or CHANGES a
known one. A change is the most valuable thing you can record: someone moved,
changed job or role, a budget or price moved, a plan was dropped, a preference
reversed, a role or status changed.

@@FACT_SCOPE@@

Not a fact, ever: an event, incident, finding, bug, outage or security issue
(those are MEMORIES if they qualify at all); anything about credentials, tokens,
keys or passwords, including their names or where they leaked; a passing mood;
small talk; advice the assistant gave; anything true only inside this
conversation.

Each fact object:
  "entity"     who it is about, short and stable: "user", "alex", "acme-corp".
               If KNOWN FACTS already has this subject, reuse its key EXACTLY.
  "attribute"  what about them, short and stable: "residence", "employer",
               "job-title", "@@FACT_ATTR@@". Reuse a KNOWN FACTS key EXACTLY
               when it is the same attribute.
  "value"      the value as it stands NOW: a short noun phrase of at most 12
               words, a name, place, number, status or date. "Melbourne,
               Australia", "Senior designer at Acme", "3,000 USD". Never a
               sentence of explanation, never "moved" or "changed".
               If it only holds under a condition, the condition is PART of the
               value: "tea in the evening, coffee before noon", "remote except
               on client days". A preference recorded without its condition is
               wrong, not shorter.
  "change"     true only when this replaces a different KNOWN FACTS value.
  "evidence"   the shortest exact quote that states it, under 160 characters.
One fact per attribute. Never split a fact into extra attributes for its date,
reason, source or a detail of it ("residence-change-date", "job-start-reason"):
those belong to the fact they describe, not beside it. At most %d facts.

== MEMORIES ==
A memory is written ONLY if it is:
  - a non-obvious fact that cost real effort to learn (a trap, a root cause, a
    counterintuitive behaviour of a tool or service)
  - a decision and the reason behind it
  - a standing preference or rule the user stated
  - a durable pointer (where a thing lives, what a credential is called)

NOT worth writing:
  - anything the code, the README or git history already records
  - what was done in this session as a narrative
  - anything already covered by the existing memories shown below
  - anything true only inside this conversation
  - anything that belongs in FACTS instead

At most %d memories. Each object:
  "action"   "new" or "update"
  "target"   for "update": the exact existing filename to change. omit for "new"
  "name"     short-kebab-case slug (for "new")
  "type"     one of: user, feedback, project, reference
  "description"  ONE line stating the operative fact, under 190 characters.
                 Lead with what is surprising. This line is what a future search
                 has to match, so use the words someone would actually type.
  "body"     2-6 sentences. State the fact, then why it matters, then how to act
             on it. Name the real files, commands and services. No narrative.
  "why"      one sentence on why this deserves to be permanent

Rules:
- Never invent a fact that is not in the conversation.
- Prefer "update" over "new" when an existing memory covers the same subject.
- Output the JSON object and nothing else.

CONVERSATION DATE: %s

CONVERSATION:
%s

KNOWN FACTS (entity / attribute = current value, as of):
%s

EXISTING MEMORIES ON RELATED SUBJECTS:
%s
"""

SLUG_RE = re.compile(r"[^a-z0-9-]+")


def read_session(transcript_path):
    """User turns and the assistant's prose. Tool output is deliberately dropped."""
    if not transcript_path or not os.path.exists(transcript_path):
        return "", 0
    # Transcripts reach hundreds of MB once tool results are in them. Only the
    # tail is worth reading, and reading all of it costs a minute per session.
    TAIL_BYTES = 6 * 1024 * 1024
    try:
        size = os.path.getsize(transcript_path)
        with open(transcript_path, encoding="utf-8", errors="ignore") as fh:
            if size > TAIL_BYTES:
                fh.seek(size - TAIL_BYTES)
                fh.readline()          # discard the partial line
            lines = fh.readlines()
    except OSError:
        return "", 0

    turns = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        kind = row.get("type")
        if kind not in ("user", "assistant"):
            continue
        msg = row.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        if not isinstance(content, str):
            continue
        content = re.sub(r"<[a-z-]+>.*?</[a-z-]+>", " ", content, flags=re.S)
        content = re.sub(r"\s+", " ", content).strip()
        if len(content) < 15:
            continue
        turns.append("%s: %s" % ("USER" if kind == "user" else "ASSISTANT", content[:2500]))

    text = "\n\n".join(turns)
    if len(text) > SESSION_CHARS:
        # Keep the start (what was asked) and the end (what was learned).
        text = text[: SESSION_CHARS // 3] + "\n\n[...]\n\n" + text[-(2 * SESSION_CHARS // 3):]
    return text, len(turns)


def related_memories(session_text, n=12):
    """What the store already knows about this session's subjects."""
    probe = " ".join(session_text.split()[:400])
    hits = recall.search(probe, limit=n, floor=2.0)
    if not hits:
        return "(none found)"
    con = connect()
    out = []
    for h in hits:
        row = con.execute(
            "SELECT COALESCE(e.summary,'') FROM doc d "
            "LEFT JOIN enrich e ON e.hash=d.hash WHERE d.rel=?", (h["rel"],)
        ).fetchone()
        summary = (row[0] if row else "") or h.get("descr") or ""
        out.append("- %s :: %s" % (h["rel"], summary[:220]))
    con.close()
    return "\n".join(out)


def slugify(name, kind):
    slug = SLUG_RE.sub("-", (name or "").lower().strip()).strip("-")[:60]
    if not slug:
        slug = "memory-%d" % int(time.time())
    prefix = {"user": "user", "feedback": "feedback", "reference": "reference"}.get(kind, "project")
    fname = slug.replace("-", "_")
    if not fname.startswith(prefix + "_"):
        fname = "%s_%s" % (prefix, fname)
    return slug, fname + ".md"


def already_have(prop):
    """A memory on the same subject, if one exists.

    Three sessions in two days each captured the same lesson and none saw the
    others, because the related-memories list is built by recall over the
    SESSION text and a fresh capture is not indexed yet. This checks the
    proposed description against what is on disk right now, which does not
    depend on the index.
    """
    words = set(re.findall(r"[a-z]{4,}", (prop.get("description", "") + " "
                                          + prop.get("name", "")).lower()))
    if len(words) < 4:
        return None
    best, best_score = None, 0.0
    for path, rel, _m in walk_memory():
        try:
            meta, _b, _s, _r = read_doc(path)
        except OSError:
            continue
        theirs = set(re.findall(r"[a-z]{4,}", (meta.get("description", "") + " "
                                               + meta.get("name", "")).lower()))
        if not theirs:
            continue
        score = len(words & theirs) / float(len(words | theirs))
        if score > best_score:
            best, best_score = rel, score
    if best_score >= 0.45:
        return best
    # Word overlap only catches a near-identical restatement. It missed a
    # restated rule twice in one day: the canonical note and the new capture
    # shared five words out of forty-eight, so Jaccard read 0.12 and two more
    # copies went in. Ask retrieval, which knows the aliases, as a second opinion.
    covered = recall.covered_by(
        (prop.get("description", "") + " " + " ".join(prop.get("body", "").split()[:60])).strip())
    return covered[0] if covered else None


def write_memory(prop, session_id):
    slug, fname = slugify(prop.get("name"), prop.get("type"))
    path = os.path.join(MEM_ROOT, fname)
    if os.path.exists(path):
        return None, "exists"
    twin = already_have(prop)
    if twin:
        return None, "duplicate of %s" % twin

    body = (prop.get("body") or "").strip()
    stamp = time.strftime("%Y-%m-%d")
    body += (
        "\n\nCaptured automatically from a session on %s (session %s). "
        "Nothing here was verified after the session ended; check paths and "
        "flags before acting on it." % (stamp, (session_id or "?")[:8])
    )
    text = (
        "---\n"
        "name: %s\n"
        "description: %s\n"
        "metadata:\n"
        "  type: %s\n"
        "  source: capture\n"
        "---\n\n%s\n" % (
            slug,
            (prop.get("description") or "").replace("\n", " ").strip()[:250],
            prop.get("type") or "project",
            body,
        )
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return fname, "written"


# --- facts --------------------------------------------------------------------
# WHY FACTS ARE A SEPARATE KIND. Until 2026-09-14 capture could only write NEW
# notes. When it noticed that something had changed it proposed an "update", and
# every update was routed to a review queue that is easy to ignore. So the store
# could learn that a person or company exists and never learn that it changed.
# On MemConflict, where most questions are exactly "what changed", the distilled
# layer wrote nothing at all.
#
# A fact note is low-risk to update automatically in a way a rule is not: it
# never loses information. The new value goes on top, the old one moves into a
# dated history underneath, and git keeps every version besides. Rules,
# decisions and feedback keep the reviewed path, because rewriting those
# automatically is what damaged the store four different ways in September.

FACT_PREFIX = "fact_"
SECRETISH = re.compile(
    r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY|\b[A-Za-z0-9+/]{40,}={0,2}\b|\b[0-9a-f]{32,}\b)")


def _key(text, cap=40):
    return (SLUG_RE.sub("-", (text or "").lower().strip()).strip("-") or "x")[:cap]


def fact_path(entity, attribute):
    return os.path.join(MEM_ROOT, "%s%s__%s.md" % (FACT_PREFIX, _key(entity).replace("-", "_"),
                                                   _key(attribute).replace("-", "_")))


def _read_fact(path):
    """(current value, as_of, history list, evidence) from a fact note."""
    cur = as_of = evidence = ""
    history = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("Current: "):
                m = re.match(r"Current: (.*?)(?: \(as of ([^)]*)\))?$", line)
                if m:
                    cur, as_of = m.group(1).strip(), (m.group(2) or "").strip()
            elif line.startswith("Evidence: "):
                evidence = line[len("Evidence: "):].strip().strip('"')
            elif line.startswith("- ") and ": " in line:
                d, v = line[2:].split(": ", 1)
                history.append((d.strip(), v.strip()))
    except OSError:
        pass
    return cur, as_of, history, evidence


def known_facts(limit=80):
    """Every fact note's current value, so the model reuses keys instead of
    inventing a second name for the same attribute."""
    rows = []
    for path, rel, mtime in walk_memory():
        base = os.path.basename(rel)
        if not base.startswith(FACT_PREFIX):
            continue
        try:
            meta, _b, _s, _r = read_doc(path)
        except OSError:
            continue
        md = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        entity = meta.get("entity") or md.get("entity") or ""
        attribute = meta.get("attribute") or md.get("attribute") or ""
        cur, as_of, _h, _e = _read_fact(path)
        if entity and attribute and cur:
            rows.append((mtime, "- %s / %s = %s%s" % (entity, attribute, cur,
                                                     (" (as of %s)" % as_of) if as_of else "")))
    rows.sort(reverse=True)
    return "\n".join(r[1] for r in rows[:limit]) or "(none yet)"


def _same(a, b):
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()
    return norm(a) == norm(b)


def write_fact(fact, session_date="", session_id=""):
    """Create or update one fact note. Returns (filename, status)."""
    entity = (fact.get("entity") or "").strip()
    attribute = (fact.get("attribute") or "").strip()
    value = re.sub(r"\s+", " ", str(fact.get("value") or "")).strip()
    if not entity or not attribute or not value:
        return None, "incomplete"
    # A fact value is a noun phrase. Anything longer is a narrative that belongs
    # in a memory, and nothing about credentials is ever a fact.
    if len(value) > 120 or len(value.split()) > 16:
        return None, "value too long for a fact"
    if re.search(r"(?i)\b(token|api[ _-]?key|password|secret|credential|private key)s?\b",
                 entity + " " + attribute + " " + value):
        return None, "refused: credentials are never facts"
    if SECRETISH.search(value):
        return None, "refused: looks like a secret"
    evidence = re.sub(r"\s+", " ", str(fact.get("evidence") or "")).strip()[:200]
    if SECRETISH.search(evidence):
        evidence = ""
    when = (session_date or time.strftime("%Y-%m-%d"))[:10]
    path = fact_path(entity, attribute)
    fname = os.path.basename(path)
    cur, as_of, history, old_evidence = _read_fact(path) if os.path.exists(path) else ("", "", [], "")

    if cur and _same(cur, value):
        # A confirmation writes nothing. Rewriting the file only to move a
        # "last confirmed" date changed its content, which re-triggered every
        # downstream pass that keys on content, for no new information.
        return fname, "confirmed"
    else:
        status = "changed" if cur else "new"
        history = [(when, value)] + (history or ([(as_of, cur)] if cur else []))

    label = "%s %s" % (entity, attribute.replace("-", " ").replace("_", " "))
    # Dates are what make a fact answerable: "did it change recently?" needs the
    # day the current value started, and "what was it before?" needs the old
    # value with the window it held. History is newest first and a confirmation
    # adds no row, so the top row is the day the current value began.
    since = history[0][0] if history and _same(history[0][1], value) else when
    descr = "%s: %s (since %s)." % (label, value, since)
    older = [(d, v) for d, v in history if not _same(v, value)]
    if older:
        p_from, p_val = older[0]
        descr += " Previously %s (from %s until %s)." % (p_val, p_from, since)
    lines = ["---",
             "name: fact-%s-%s" % (_key(entity), _key(attribute)),
             "description: %s" % json.dumps(descr[:260]),
             "metadata:",
             "  type: fact",
             "  entity: %s" % _key(entity),
             "  attribute: %s" % _key(attribute),
             "  source: capture",
             "---", "",
             "Current: %s (as of %s)" % (value, when)]
    if evidence:
        lines.append('Evidence: "%s"' % evidence.replace('"', "'"))
    lines += ["", "History, newest first:"]
    seen = set()
    for d, v in history:
        k = (d, v.lower())
        if k in seen:
            continue
        seen.add(k)
        lines.append("- %s: %s" % (d or "unknown", v))
    lines += ["", "Recorded automatically from conversations. A new value replaces the "
                  "current one and the old value stays in the history above, so nothing "
                  "here is ever lost. Session %s." % ((session_id or "?")[:8])]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return fname, status


def propose_all(text, n_turns, model="haiku", session_date=""):
    """(memories, facts) worth keeping from session text already in hand.

    Split out of run() so a caller that already has the dialogue (the MemConflict
    adapter) does not have to write a fake transcript file to be read back.
    """
    recall.build_index()
    prompt = scoped_prompt() % (MAX_FACTS, MAX_NEW, session_date or time.strftime("%Y-%m-%d"),
                       text, known_facts(), related_memories(text))
    data = claude_json(prompt, model=model, timeout=420, system=SYSTEM)
    memories, facts = [], []
    if isinstance(data, dict):
        memories = data.get("memories") or data.get("results") or []
        facts = data.get("facts") or []
    elif isinstance(data, list):
        # an older-style bare array: treat objects with an entity as facts
        facts = [d for d in data if isinstance(d, dict) and d.get("entity")]
        memories = [d for d in data if isinstance(d, dict) and not d.get("entity")]
    memories = [m for m in memories if isinstance(m, dict)][:MAX_NEW]
    facts = [f for f in facts if isinstance(f, dict)][:MAX_FACTS]
    return memories, facts


def propose(text, n_turns, model="haiku"):
    """Memories only, for callers written before facts existed."""
    return propose_all(text, n_turns, model=model)[0]


def run(transcript_path, session_id="", model="haiku", dry=False, verbose=True):
    text, n_turns = read_session(transcript_path)
    if n_turns < MIN_TURNS or len(text) < MIN_CHARS:
        if verbose:
            print("session too small to be worth capturing (%d turns)" % n_turns)
        return []

    try:
        props, facts = propose_all(text, n_turns, model=model)
    except LLMError as exc:
        log_line("memory-capture.log", "ERROR", (session_id or "?")[:8], str(exc)[:200])
        if verbose:
            print("capture failed: %s" % exc)
        return []

    fact_log = []
    if facts and not dry:
        with MemoryLock():
            con = connect()
            for f in facts:
                fname, status = write_fact(f, session_id=session_id)
                if fname and status in ("new", "changed", "confirmed"):
                    audit(con, "fact-" + status, fname,
                          ("%s = %s" % (f.get("attribute"), f.get("value")))[:200], "capture")
                fact_log.append("%s:%s" % (status, fname or f.get("attribute")))
            con.commit()
            con.close()
        log_line("memory-capture.log", "FACTS", (session_id or "?")[:8], "|".join(fact_log))
        if verbose:
            for x in fact_log:
                print("  fact        %s" % x)
    elif facts and dry:
        fact_log = ["would-record:%s/%s=%s" % (f.get("entity"), f.get("attribute"), f.get("value"))
                    for f in facts]
        if verbose:
            for x in fact_log:
                print("  %s" % x)

    if not props:
        log_line("memory-capture.log", "NONE", (session_id or "?")[:8], "%d turns" % n_turns)
        if verbose:
            print("nothing worth keeping from this session")
        return []

    written = []
    with MemoryLock():
        con = connect()
        for p in props:
            action = str(p.get("action", "new")).lower()
            if action == "update":
                target = str(p.get("target", "")).strip()
                detail = "%s\n      proposed: %s" % (
                    p.get("why", ""), (p.get("description") or "")[:200])
                if not dry:
                    con.execute(
                        "INSERT INTO review(ts, kind, rels, detail, status) "
                        "VALUES (?,?,?,?, 'open')",
                        (time.time(), "update", target or "(unknown)", detail),
                    )
                written.append(("review", target or "(unknown)"))
                continue

            if dry:
                written.append(("would-write", p.get("name")))
                continue
            fname, status = write_memory(p, session_id)
            if fname:
                audit(con, "capture", fname, (p.get("why") or "")[:200], "capture")
                written.append(("new", fname))
            elif status.startswith("duplicate of"):
                # Same subject already on disk: route as an update for review
                # rather than writing a fourth file about it.
                con.execute(
                    "INSERT INTO review(ts, kind, rels, detail, status) VALUES (?,?,?,?, 'open')",
                    (time.time(), "update", status[len("duplicate of "):],
                     "new capture on the same subject: %s" % (p.get("description") or "")[:200]))
                written.append(("dup->review", status[len("duplicate of "):]))
            else:
                written.append((status, p.get("name")))
        con.commit()
        con.close()

    log_line("memory-capture.log", "RUN", (session_id or "?")[:8],
             "turns=%d" % n_turns, "|".join("%s:%s" % w for w in written))
    if verbose:
        for kind, name in written:
            print("  %-11s %s" % (kind, name))
    return written


def main():
    ap = argparse.ArgumentParser(prog="capture.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("session")
    p.add_argument("--transcript", required=True)
    p.add_argument("--session-id", default="")
    p.add_argument("--model", default="haiku")
    p.add_argument("--dry-run", action="store_true")

    sub.add_parser("pending")
    args = ap.parse_args()

    if args.cmd == "session":
        run(args.transcript, args.session_id, model=args.model, dry=args.dry_run)
        return 0

    if args.cmd == "pending":
        con = connect()
        rows = con.execute(
            "SELECT id, ts, kind, rels, detail FROM review "
            "WHERE status='open' ORDER BY ts DESC LIMIT 40").fetchall()
        con.close()
        if not rows:
            print("nothing pending")
        for rid, ts, kind, rels, detail in rows:
            print("#%-4d %s  %-8s %s\n      %s" % (
                rid, time.strftime("%Y-%m-%d", time.localtime(ts)), kind, rels, detail))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
profile.py - USER.md: the small, always-loaded model of who the user is.

Everything else in this system is retrieval: it waits to be asked. This file is
the opposite. It is in the prompt before a word is typed, so it has to earn its
place in a way a retrieved memory does not, and it has to stay small.

WHY THE OVERFLOW RULE IS INVERTED FROM HERMES. Hermes hard-caps its user profile
and makes the write ERROR when full, on the theory that this forces the agent to
consolidate. In a real deployment it does not: their issue #32064 reports the
profile "repeatedly hits the user-profile cap, causing failed memory.add calls
and repeated loss of operator corrections/preferences". A failed write does not
produce consolidation, it produces silence, and the thing that gets lost is the
correction the user just made. So here a write NEVER fails. On overflow the
weakest existing line is demoted into the searchable store, where recall can
still surface it, and the demotion is logged for review. The file stays bounded;
nothing is destroyed.

PROVENANCE IS A FIRST-CLASS FIELD. Anthropic's own auto-memory prompt draws the
line hard: "the lesson must be something the user told you or corrected you on,
not a finding of your own about the code, the tools, or your own mistake." That
distinction is worth keeping rather than flattening, so every line is tagged:

  [told]      The user stated or corrected this directly. Highest confidence.
  [observed]  Derived from how they actually work. Useful, weaker, decays first.

When the file is full, [observed] lines are demoted before [told] lines, always.

Usage:
  profile.py build [--model haiku]     compose USER.md from the store
  profile.py show
  profile.py add "line" [--told|--observed]
  profile.py context                   what the SessionStart hook injects
  profile.py stats
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recall  # noqa: E402
from memlib import (  # noqa: E402
    MEM_ROOT, LLMError, MemoryLock, audit, claude, claude_json, connect,
    log_line, read_doc, walk_memory, write_atomic,
)

USER_MD = os.path.join(MEM_ROOT, "USER.md")
# Lines that survive every rebuild verbatim. The composer is a model and it is
# not deterministic: in testing two of four rebuilds dropped the rule the user had
# corrected most often.
# Anything in this file is written first and never demoted.
PINNED_MD = os.path.join(MEM_ROOT, "USER.pinned.md")

# Hermes uses 1,375 chars. A little more, because a user's standing rules are the
# load-bearing part and cutting them costs more than the tokens save. Still an
# order of magnitude under the 22KB index file.
MAX_CHARS = 2000
MAX_LINES = 22

HEADER = """<!-- USER.md - always in context. Bounded at %d chars.
     [told] = the user said it. [observed] = inferred from how they work.
     Never edit the cap upward without reading profile.py's docstring. -->
"""


def parse(text=None):
    """Return [{tag, text}] for the profile's live lines."""
    if text is None:
        try:
            text = open(USER_MD, encoding="utf-8").read()
        except OSError:
            return []
    out = []
    for line in text.splitlines():
        m = re.match(r"^- \[(told|observed)\]\s+(.+)$", line.strip())
        if m:
            out.append({"tag": m.group(1), "text": m.group(2).strip()})
    return out


def render(entries):
    body = "\n".join("- [%s] %s" % (e["tag"], e["text"]) for e in entries)
    return (HEADER % MAX_CHARS) + "\n" + body + "\n"


def size_of(entries):
    return len(render(entries))


def demote(entry, reason):
    """Move a line out of the always-on file and into the searchable store.

    This is the whole reason a write never has to fail. The fact stays findable
    by recall; it just stops costing tokens on every single prompt.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", entry["text"].lower())[:48].strip("-")
    fname = "user_demoted_%s.md" % slug.replace("-", "_")
    path = os.path.join(MEM_ROOT, fname)
    if os.path.exists(path):
        return fname

    # The filename comes from the wording, and the composer rewords a line a
    # little on every rebuild, so the same rule demoted twice lands in two
    # different files. That ran 39 times, with several rules copied three times,
    # each copy competing with the real one for a slot in the four memories a
    # prompt gets. Ask retrieval whether the store already carries this fact
    # before writing it down again.
    covered = recall.covered_by(entry["text"])
    if covered:
        rel, score = covered
        try:
            with MemoryLock():
                con = connect()
                con.execute(
                    "INSERT INTO review(ts, kind, rels, detail, status) "
                    "VALUES (?,?,?,?, 'open')",
                    (time.time(), "update", rel,
                     "USER.md line dropped, not written to a new file: %s "
                     "already carries it (score %s). Line was: %s"
                     % (rel, score, entry["text"][:150])))
                con.commit(); con.close()
        except Exception:
            pass
        return rel
    write_atomic(path, (
        "---\nname: %s\ndescription: %s\nmetadata:\n  type: user\n  source: profile-demotion\n---\n\n"
        "%s\n\nDemoted from USER.md on %s because %s. It was not deleted: it is "
        "still retrievable, it just no longer costs tokens on every prompt.\n"
        % (slug or "demoted-preference",
           entry["text"][:190].replace("\n", " "),
           entry["text"], time.strftime("%Y-%m-%d"), reason)
    ))
    return fname


def fit(entries, verbose=True, dry=False):
    """Bring the profile inside its bounds by demoting, never by erroring."""
    demoted = []
    # [observed] goes before [told]; within a tag, the oldest at the bottom goes
    # first, because the composer writes in descending importance.
    while size_of(entries) > MAX_CHARS or len(entries) > MAX_LINES:
        idx = None
        for i in range(len(entries) - 1, -1, -1):
            if entries[i]["tag"] == "observed" and not entries[i].get("pinned"):
                idx = i
                break
        if idx is None:
            for i in range(len(entries) - 1, -1, -1):
                if not entries[i].get("pinned"):
                    idx = i
                    break
        if idx is None:
            break
        if len(entries) <= 1:
            break
        victim = entries.pop(idx)
        # dry=True is for tests. A direct call to fit() wrote 16 filler files
        # into the real store once; demotion must never be a side effect of
        # checking whether demotion works.
        fname = ("(dry)" if dry else
                 demote(victim, "USER.md was over its %d-char bound" % MAX_CHARS))
        demoted.append((victim, fname))
        if verbose:
            print("  demoted [%s] %s -> %s" % (victim["tag"], victim["text"][:60], fname))
    return entries, demoted


def write(entries, verbose=True):
    entries, demoted = fit(entries, verbose=verbose)
    with MemoryLock():
        write_atomic(USER_MD, render(entries))
        con = connect()
        for victim, fname in demoted:
            audit(con, "profile-demote", fname, victim["text"][:180], "profile")
            # Recorded in the audit trail, NOT the review queue. Demotion is
            # something the system did correctly and reversibly; putting it in
            # front of the user trains them to ignore a queue that should only ever
            # hold real decisions.
        con.commit()
        con.close()
    log_line("memory-profile.log", "WRITE", "%d lines" % len(entries),
             "%d chars" % size_of(entries), "%d demoted" % len(demoted))
    return entries, demoted


def add(text, tag="told", verbose=True):
    """Add a line. This can never fail; something else moves aside if needed."""
    entries = parse()
    if any(e["text"].lower() == text.lower() for e in entries):
        if verbose:
            print("already in the profile")
        return entries
    # [told] goes to the top: it is what the user actually said.
    entries.insert(0 if tag == "told" else len(entries), {"tag": tag, "text": text})
    entries, _ = write(entries, verbose=verbose)
    if verbose:
        print("added [%s] %s" % (tag, text[:70]))
        print("profile now %d chars / %d, %d lines" % (size_of(entries), MAX_CHARS, len(entries)))
    return entries


SYSTEM = (
    "You write a tiny, high-signal user profile that will sit in an AI "
    "assistant's context on every single turn. Every character costs tokens "
    "forever. You output JSON only."
)

PROMPT = """Below are the standing rules and user facts from an engineer's memory store.

Compose the ALWAYS-LOADED profile of this person. It goes in the system prompt on
every turn of every session, so it must be brutally short: at most %d lines and
%d characters TOTAL.

Return JSON: {"entries": [{"tag": "told"|"observed", "text": "..."}]}

What earns a line:
- how they want to be worked with, where getting it wrong wastes their time
- standing rules that change behaviour on most tasks
- who they are and what they are building, in the fewest words that are still useful
- the corrections they have had to repeat

What does NOT earn a line:
- anything specific to one project, service or incident. Retrieval handles those.
- anything the assistant would do anyway
- background, history, or nuance. This is not documentation, it is a lens.

Tagging:
- "told" if the source shows they stated or corrected it themselves
- "observed" if it is inferred from how they work

Rules:
- Order by how often it changes behaviour, most first.
- One clause per line where possible. No line over 130 characters.
- Write it as instructions to the assistant, not description of the user.
- Fewer, sharper lines beat more lines. Ten excellent lines beat twenty adequate ones.

SOURCE MATERIAL:
%s
"""


def source_material(limit=60):
    """The user and feedback memories: the rules, in their own words."""
    rows = []
    for path, rel, mtime in walk_memory():
        try:
            meta, body, _sha, _raw = read_doc(path)
        except OSError:
            continue
        kind = (meta.get("metadata_type") or meta.get("type") or "")
        base = os.path.basename(rel)
        if kind not in ("user", "feedback") and not base.startswith(("user_", "feedback_")):
            continue
        rows.append((mtime, "### %s\n%s\n%s" % (
            meta.get("name") or base, meta.get("description", ""), body[:700])))
    rows.sort(reverse=True)
    return "\n\n".join(r[1] for r in rows[:limit])


def pinned():
    """[told] lines that a rebuild may never lose."""
    try:
        return [{"tag": "told", "text": l.strip().lstrip("-").strip(), "pinned": True}
                for l in open(PINNED_MD, encoding="utf-8")
                if l.strip() and not l.strip().startswith("#")]
    except OSError:
        return []


def build(model="haiku", verbose=True):
    src = source_material()
    if not src:
        print("no user/feedback memories to build from")
        return []
    data = claude_json(PROMPT % (MAX_LINES, MAX_CHARS, src), model=model,
                       timeout=420, system=SYSTEM)
    entries = data.get("entries") if isinstance(data, dict) else data
    clean = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        text = str(e.get("text", "")).strip().lstrip("-").strip()
        if not text:
            continue
        tag = "told" if str(e.get("tag", "told")).lower().startswith("t") else "observed"
        clean.append({"tag": tag, "text": text[:200]})
    if not clean:
        print("model returned nothing usable")
        return []
    pins = pinned()
    pinned_lower = {p["text"].lower()[:40] for p in pins}
    clean = [e for e in clean if e["text"].lower()[:40] not in pinned_lower]
    entries, demoted = write(pins + clean, verbose=verbose)
    if verbose:
        print("\nUSER.md: %d lines, %d/%d chars (~%d tokens every turn)"
              % (len(entries), size_of(entries), MAX_CHARS, size_of(entries) // 3.7))
    return entries


def context_block():
    """What the SessionStart hook injects. Empty string if there is no profile."""
    entries = parse()
    if not entries:
        return ""
    lines = ["<user-profile>",
             "Who you are working with and how. This is always true, unlike a "
             "retrieved memory. [told] means they said it themselves.", ""]
    lines += ["- [%s] %s" % (e["tag"], e["text"]) for e in entries]
    lines.append("</user-profile>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="profile.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("build")
    p.add_argument("--model", default="haiku")

    sub.add_parser("show")
    sub.add_parser("context")
    sub.add_parser("stats")

    p = sub.add_parser("add")
    p.add_argument("text")
    p.add_argument("--observed", action="store_true")

    args = ap.parse_args()

    if args.cmd == "build":
        try:
            build(model=args.model)
        except LLMError as exc:
            print("build failed: %s" % exc)
            return 1
        return 0

    if args.cmd == "show":
        entries = parse()
        if not entries:
            print("no profile yet. Run: profile.py build")
            return 1
        for e in entries:
            print("  [%-8s] %s" % (e["tag"], e["text"]))
        print("\n  %d lines, %d/%d chars" % (len(entries), size_of(entries), MAX_CHARS))
        return 0

    if args.cmd == "context":
        print(context_block())
        return 0

    if args.cmd == "add":
        add(args.text, tag="observed" if args.observed else "told")
        return 0

    if args.cmd == "stats":
        entries = parse()
        told = sum(1 for e in entries if e["tag"] == "told")
        print("lines:     %d (%d told, %d observed)" % (len(entries), told, len(entries) - told))
        print("size:      %d / %d chars (~%d tokens per turn)"
              % (size_of(entries), MAX_CHARS, size_of(entries) / 3.7))
        con = connect()
        n = con.execute("SELECT count(*) FROM audit WHERE action='profile-demote'").fetchone()[0]
        con.close()
        print("demoted:   %d line(s) moved to the searchable store over time" % n)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
curate.py - the control plane. The part that makes the store fix itself.

The recall plane (find the right memory) is the well-studied half. The control
plane (supersede, flag, retire) is where memory systems actually rot, and the
research is blunt about the split: deterministic rules handle lexical and
temporal cases but cannot canonicalize, an LLM canonicalizes but cannot be
trusted with deletion intent, and running BOTH at different points beats either
alone by a wide margin. So this file does exactly that.

Deterministic passes (cheap, exact, run every time):
  duplicates   byte-identical files
  staleness    a memory that names an absolute path which no longer exists
  orphans      enrichment rows for content that no longer exists

Model pass (offline, batched, through `claude -p`):
  contradictions   two memories that cannot both be true; the older one is
                   marked superseded, never deleted, and the decision is
                   written to the audit table with the reason

Nothing is ever deleted. Supersession is a ranking penalty plus a visible
[SUPERSEDED by ...] label on recall, so a wrong call is obvious and reversible.
Anything the model is not confident about goes to the review queue instead.

Usage:
  curate.py run [--limit N] [--model haiku] [--dry-run]
  curate.py stale                 deterministic passes only
  curate.py review [--all]        what is waiting for a human
  curate.py audit [--n 40]        what the curator has done
  curate.py undo <rel>            clear a supersession or stale flag
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recall  # noqa: E402
import verify  # noqa: E402
from memlib import (  # noqa: E402
    MEMORY_MD, MEMORY_MD_MAX_BYTES, MEMORY_MD_MAX_LINES, MEM_ROOT,
    LLMError, MemoryLock, audit, claude_json, connect, log_line, read_doc,
    walk_memory, write_atomic,
)

SYSTEM = (
    "You audit an engineering memory store for contradictions. You are "
    "conservative: two notes about different things are not a contradiction, "
    "and a note that adds detail to another is not a contradiction. You output "
    "JSON only."
)

PROMPT = """Below are %d PAIRS of notes from one engineer's memory store.

For each pair decide whether the two notes CONTRADICT: they make claims that
cannot both be true right now about the same subject.

NOT a contradiction:
- different subjects that share vocabulary
- one note adding detail, scope or nuance to the other
- a general rule and a specific exception to it
- the same fact written twice in different words (that is a DUPLICATE)

IS a contradiction:
- opposite values for the same setting, path, owner, price, or status
- one says a thing is live and the other says it is dead or blocked
- one supersedes the other because the world changed

For each pair return an object with:
  "i"        the pair number as given
  "verdict"  one of "contradict", "duplicate", "independent"
  "current"  when "contradict" or "duplicate": which file is the one to keep,
             "a" or "b". Prefer the note that is dated later or describes the
             later state of the world. If you genuinely cannot tell, use "unsure".
  "scope"    "full" if the keeper covers EVERYTHING the other note claims, so
             nothing is lost by retiring it. "partial" if the other note also
             carries claims the keeper does not address. A note listing several
             open items, where the keeper resolves only one of them, is
             "partial". When in doubt say "partial".
  "why"      one sentence, under 160 characters, naming the specific conflict
  "confidence" 0.0 to 1.0

Output a JSON array, one object per pair, nothing else.

PAIRS:
"""

def subject_overlap(con, a, b):
    """How much two memories are actually about the same thing.

    A shared technology is not a shared subject. An early model pass retired
    a note about one service in favour of a release note about a different one,
    at 0.85 confidence, on the strength of one shared technology name.
    Real pairs share two or more topics AND several entities; that false one
    shared exactly one of each.
    """
    def sets(rel):
        row = con.execute(
            "SELECT COALESCE(e.topics,''), COALESCE(e.entities,'') FROM doc d "
            "LEFT JOIN enrich e ON e.hash=d.hash WHERE d.rel=?", (rel,)).fetchone()
        if not row:
            return set(), set()
        return set(row[0].lower().split()), set(row[1].lower().split())
    ta, ea = sets(a)
    tb, eb = sets(b)
    return len(ta & tb), len(ea & eb)


PATH_RE = re.compile(r"`((?:/Users|/home)/[^/`\s]+/[^`\s]{4,160})`")
MIN_SHARED_TOPICS = 2
MIN_SHARED_ENTITIES = 2
# A rule is not retired by an event. The first unguarded run retired standing
# rules in favour of status notes that happened to share a topic. Standing
# guidance and status reports are different kinds of claim and one never
# supersedes the other; those pairs go to the user.
RULE_KINDS = {"feedback", "user"}
# An `_index/*.md` file is a pointer map, not a claim. It neither supersedes a
# memory nor gets superseded by one: it restates whatever it points at, so a
# model reading the two side by side always sees "the same fact twice". The
# first guarded run let an index file retire the project note it points at,
# on exactly that reasoning.
NEVER_SUPERSEDE_KINDS = {"index"}

# USER.md and USER.pinned.md are COMPOSED from the store. They restate the rules
# on purpose, which is their whole job, so a duplicate finding against them is
# structural and will be true forever. Left in, the curator re-files the same
# three items every single run and the queue fills with noise nobody can ever
# close. Do not compare them against anything.
COMPOSED_FILES = {"USER.md", "USER.pinned.md"}
# `repo/.worktrees/<feature>-<date>` is a template, not a path that
# went missing. Anything with a placeholder or a glob in it is documentation.
TEMPLATE_CHARS = set("<>{}*?$")
# Paths that are SUPPOSED to disappear. A memory naming the worktree it was
# written in, or a build artefact, is not rotten just because that thing is
# gone. Flagging them buries the real signal: on the first run 9 of 13 flags
# were paths like these.
EPHEMERAL = re.compile(
    r"/\.worktrees?[-/]|/\.worktrees$|/worktrees?/|/node_modules/|"
    r"/dist/|/build/|/\.next/|/scratchpad/|\.dmg$|\.zip$|\.log$"
)
MIN_CONFIDENCE = 0.75


# ---------------------------------------------------------------- deterministic


def file_review(con, kind, rels, detail):
    """File a review unless the identical one is already open.

    The curator re-evaluates the whole store on every run, so a pair it cannot
    resolve on its own gets re-filed every single time. Three copies of the same
    USER.md finding were sitting in the queue. A queue that repeats itself is a
    queue nobody reads.
    """
    a, b = (rels.split("|", 1) + [""])[:2]
    if a.strip() in COMPOSED_FILES or b.strip() in COMPOSED_FILES:
        return
    dup = con.execute(
        "SELECT 1 FROM review WHERE status='open' AND kind=? AND rels=? LIMIT 1",
        (kind, rels)).fetchone()
    if dup:
        return
    con.execute(
        "INSERT INTO review(ts, kind, rels, detail, status) VALUES (?,?,?,?, 'open')",
        (time.time(), kind, rels, detail))


def pass_duplicates(con, dry=False):
    """Byte-identical memory files. One is redundant by definition."""
    found = []
    rows = con.execute(
        "SELECT hash, group_concat(rel, '|') , count(*) c FROM doc "
        "GROUP BY hash HAVING c > 1"
    ).fetchall()
    for sha, rels, _c in rows:
        parts = rels.split("|")
        keep, drop = parts[0], parts[1:]
        found.append((keep, drop))
        if dry:
            continue
        for d in drop:
            con.execute(
                "UPDATE doc SET superseded_by=? WHERE rel=? AND superseded_by=''",
                (keep, d),
            )
            audit(con, "duplicate", d, "byte-identical to %s" % keep, "deterministic")
    return found


def pass_staleness(con, dry=False):
    """A memory that points at an absolute path which no longer exists.

    This is the cheapest true signal of rot in the store: the note said "the
    fix lives in <file>" and the file has since moved or been deleted.
    """
    flagged, cleared = [], []
    for path, rel, _mtime in walk_memory():
        try:
            _meta, body, _sha, raw = read_doc(path)
        except OSError:
            continue
        missing = []
        for m in set(PATH_RE.findall(raw)):
            target = m.rstrip("/.,)")
            if TEMPLATE_CHARS & set(target) or EPHEMERAL.search(target):
                continue
            if not os.path.exists(target):
                missing.append(target)
        row = con.execute("SELECT stale_note FROM doc WHERE rel=?", (rel,)).fetchone()
        existing = row[0] if row else ""
        if missing:
            note = "%d referenced path(s) gone: %s" % (
                len(missing), ", ".join(os.path.basename(p) for p in missing[:3]))
            if note != existing:
                flagged.append((rel, note))
                if not dry:
                    con.execute("UPDATE doc SET stale_note=? WHERE rel=?", (note, rel))
                    audit(con, "stale", rel, note, "deterministic")
        elif existing:
            cleared.append(rel)
            if not dry:
                con.execute("UPDATE doc SET stale_note='' WHERE rel=?", (rel,))
                audit(con, "stale-cleared", rel, "paths exist again", "deterministic")
    return flagged, cleared


def pass_orphans(con, dry=False):
    """Enrichment rows whose content hash is no longer in the store."""
    n = con.execute(
        "SELECT count(*) FROM enrich WHERE hash NOT IN (SELECT hash FROM doc)"
    ).fetchone()[0]
    if n and not dry:
        con.execute("DELETE FROM enrich WHERE hash NOT IN (SELECT hash FROM doc)")
    return n


# ---------------------------------------------------------------- model pass


def candidate_pairs(con, limit=None):
    """Memories close enough to each other to be worth a contradiction check.

    Uses the index itself: each memory searches for its own subject and the
    nearest others become candidates. Cheap, and it only ever proposes pairs
    the retriever would actually return together.
    """
    docs = con.execute(
        "SELECT d.rel, d.name, d.kind, d.mtime, "
        "       COALESCE(e.summary,''), COALESCE(e.topics,'') "
        "FROM doc d LEFT JOIN enrich e ON e.hash = d.hash "
        "WHERE d.superseded_by='' ORDER BY d.mtime DESC"
    ).fetchall()
    by_rel = {r[0]: r for r in docs}

    # A pair the user has already ruled on stays ruled on. Without this the sweep
    # re-proposes every dismissed pair on every run, and the review queue fills
    # up with questions that were answered days ago.
    settled = set()
    for (rels,) in con.execute(
            "SELECT rels FROM review WHERE status IN ('dismissed','resolved')"):
        parts = [x.strip() for x in rels.split("|") if x.strip()]
        if len(parts) == 2:
            settled.add(tuple(sorted(parts)))

    seen, pairs = set(settled), []
    for rel, name, _kind, _mtime, summary, topics in docs:
        probe = " ".join([name.replace("_", " "), summary, topics]).strip()
        if len(probe) < 20:
            continue
        for hit in recall.search(probe, limit=4, floor=2.0):
            other = hit["rel"]
            if other == rel or other not in by_rel:
                continue
            key = tuple(sorted((rel, other)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((by_rel[key[0]], by_rel[key[1]]))
            if limit and len(pairs) >= limit:
                return pairs
    return pairs


def pair_block(i, a, b):
    def side(tag, row):
        rel, name, kind, mtime, summary, _topics = row
        return "  %s file: %s\n  %s name: %s (%s, updated %s)\n  %s says: %s\n" % (
            tag, rel, tag, name, kind,
            time.strftime("%Y-%m-%d", time.localtime(mtime)), tag,
            (summary or "(no summary)")[:400],
        )
    return "\n--- PAIR %d ---\n%s%s" % (i, side("a", a), side("b", b))


def check_batch(batch, model):
    prompt = PROMPT % len(batch)
    for i, (a, b) in enumerate(batch):
        prompt += pair_block(i, a, b)
    data = claude_json(prompt, model=model, timeout=420, system=SYSTEM)
    if isinstance(data, dict):
        data = data.get("pairs") or data.get("results") or [data]
    out = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        try:
            idx = int(obj.get("i", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(batch):
            continue
        a, b = batch[idx]
        out.append({
            "a": a[0], "b": b[0],
            "a_mtime": a[3], "b_mtime": b[3],
            "scope": str(obj.get("scope", "partial")).lower().strip(),
            "verdict": str(obj.get("verdict", "")).lower().strip(),
            "current": str(obj.get("current", "")).lower().strip(),
            "why": str(obj.get("why", "")).strip()[:300],
            "confidence": float(obj.get("confidence") or 0),
        })
    return out


def apply_verdict(con, v, dry=False):
    """Return a short action string, or None when nothing is done."""
    if v["verdict"] not in ("contradict", "duplicate"):
        return None

    keep = drop = None
    if v["current"] == "a":
        keep, drop = v["a"], v["b"]
    elif v["current"] == "b":
        keep, drop = v["b"], v["a"]

    shared_topics, shared_entities = subject_overlap(con, v["a"], v["b"])
    same_subject = shared_topics >= MIN_SHARED_TOPICS or shared_entities >= MIN_SHARED_ENTITIES
    kinds = {r[0] for r in con.execute(
        "SELECT kind FROM doc WHERE rel IN (?,?)", (v["a"], v["b"]))}
    same_class = (len(kinds & RULE_KINDS) in (0, len(kinds))
                  and not (kinds & NEVER_SUPERSEDE_KINDS))
    # Partial overlap is the subtlest failure and it passed every other guard:
    # one note listed SIX open decisions, a newer note resolved exactly one of
    # them, and the model retired the whole list at high confidence. A keeper has
    # to cover everything the loser claims, or five open decisions get buried
    # under a SUPERSEDED label.
    full_scope = v.get("scope") == "full"
    low = (v["confidence"] < MIN_CONFIDENCE or not same_subject
           or not same_class or not full_scope)
    if keep is None or low:
        # Not confident enough to reorder the user's memory on our own.
        if not dry:
            file_review(
                con, v["verdict"], "%s | %s" % (v["a"], v["b"]),
                "%s (confidence %.2f, current=%s, shared topics=%d entities=%d%s)"
                % (v["why"], v["confidence"], v["current"] or "?",
                   shared_topics, shared_entities,
                   ("" if same_class else ", index map or rule-vs-event")
                   + ("" if full_scope else ", PARTIAL scope")))
        return "review  %s <-> %s" % (v["a"], v["b"])

    # Never let the model retire the newer note over the older one; if it wants
    # to, that is a judgement call for the user, not an automatic rewrite.
    keep_mtime = v["a_mtime"] if keep == v["a"] else v["b_mtime"]
    drop_mtime = v["b_mtime"] if keep == v["a"] else v["a_mtime"]
    if drop_mtime > keep_mtime + 86400:
        if not dry:
            file_review(con, v["verdict"], "%s | %s" % (keep, drop),
                        "model wants to retire the NEWER note: %s" % v["why"])
        return "review  newer-loses  %s" % drop

    if not dry:
        con.execute(
            "UPDATE doc SET superseded_by=? WHERE rel=? AND superseded_by=''",
            (keep, drop),
        )
        con.execute(
            "UPDATE doc SET supersedes=? WHERE rel=?", (drop, keep),
        )
        audit(con, v["verdict"], drop,
              "superseded by %s: %s (conf %.2f)" % (keep, v["why"], v["confidence"]),
              "model")
    return "supersede  %s -> %s" % (drop, keep)


# ---------------------------------------------------------------- MEMORY.md


DETAIL_INDEX = "_index/detail-map.md"
TARGET_BYTES = 21000          # 25,000 cap, kept with headroom to grow into
TARGET_LINES = 185            # 200 cap
LINE_CAP = 165                # a pointer line does not need to be a paragraph


def compact_memory_md(dry=False, verbose=True):
    """Keep MEMORY.md inside the loader's undocumented caps.

    Claude Code loads only the first 200 lines or 25,000 bytes of MEMORY.md,
    whichever comes first, and says nothing useful when it truncates. Past the cap the
    bottom of the file, usually the newest lines, silently never loads.

    This does nothing while the file fits. When it does not, nothing is deleted:
    nested lines and the last pointer lines move to an _index file that
    retrieval still reaches, and a trimmed pointer keeps its full text there.
    """
    if not os.path.exists(MEMORY_MD):
        return None
    with open(MEMORY_MD, encoding="utf-8") as fh:
        original = fh.read()
    lines = original.split("\n")

    before = (len(lines), len(original.encode("utf-8")))
    # Nothing is at risk while the loader reads the whole file, so nothing is touched.
    if before[0] <= MEMORY_MD_MAX_LINES and before[1] <= MEMORY_MD_MAX_BYTES:
        if verbose:
            print("MEMORY.md  %d lines / %d bytes: fits, nothing to do" % before)
        return before

    kept, moved = [], []
    for line in lines:
        if line.startswith("  - ") or line.startswith("\t- "):
            moved.append(line.strip())
            continue
        kept.append(line)

    def trim(line):
        # Only a pointer line is trimmed: its full text lives in the file it links,
        # and a copy of the whole line goes to the detail map as well.
        if not line.startswith("- [") or len(line) <= LINE_CAP:
            return line
        moved.append(line)
        cut = line[:LINE_CAP]
        space = cut.rfind(" ")
        return (cut[:space] if space > LINE_CAP * 0.6 else cut).rstrip(" ,;-") + " ..."

    kept = [trim(l) for l in kept]

    # Nested lines are gone and the file still sits at the cap, so moving them
    # cannot help twice. Enforce real headroom: while over the target, move the
    # LAST pointer line, the part the loader would have dropped anyway, into the
    # detail map. Recall reaches every moved line, so nothing becomes unfindable.
    def is_movable(i):
        return kept[i].startswith("- [")
    while len(kept) > TARGET_LINES or len("\n".join(kept).encode("utf-8")) > TARGET_BYTES:
        idx = next((i for i in range(len(kept) - 1, -1, -1) if is_movable(i)), None)
        if idx is None:
            break
        moved.append(kept.pop(idx))

    # Collapse runs of blank lines left behind by the moved bullets.
    squeezed, blank = [], False
    for line in kept:
        if not line.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        squeezed.append(line)
    kept = squeezed

    text = "\n".join(kept)
    after = (len(kept), len(text.encode("utf-8")))

    if verbose:
        print("MEMORY.md  %d lines / %d bytes  ->  %d lines / %d bytes"
              % (before[0], before[1], after[0], after[1]))
        print("  caps: %d lines / %d bytes. %s"
              % (MEMORY_MD_MAX_LINES, MEMORY_MD_MAX_BYTES,
                 "FITS" if after[0] <= MEMORY_MD_MAX_LINES and after[1] <= MEMORY_MD_MAX_BYTES
                 else "STILL OVER"))
        print("  %d detail line(s) move to %s" % (len(moved), DETAIL_INDEX))
    if dry:
        return after

    detail_path = os.path.join(MEM_ROOT, DETAIL_INDEX)
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)
    header = (
        "# Detail map\n\n"
        "Lines moved out of MEMORY.md on %s to keep it inside the length Claude "
        "Code loads (200 lines or 25,000 bytes). Nothing was lost: retrieval "
        "indexes this file and the memories it points at.\n\n"
        % time.strftime("%Y-%m-%d")
    )
    existing = ""
    if os.path.exists(detail_path):
        with open(detail_path, encoding="utf-8") as fh:
            existing = fh.read()
    seen = set(re.findall(r"^- .*$", existing, re.M))
    new_lines = [m for m in moved if m not in seen]
    body = (existing if existing else header) + "\n".join(new_lines) + "\n"
    write_atomic(detail_path, body)
    write_atomic(MEMORY_MD, text)

    con = connect()
    audit(con, "compact-index", "MEMORY.md",
          "%d->%d lines, %d->%d bytes, %d detail lines moved"
          % (before[0], after[0], before[1], after[1], len(new_lines)), "deterministic")
    con.commit()
    con.close()
    log_line("memory-curate.log", "COMPACT",
             "%d->%d lines" % (before[0], after[0]),
             "%d->%d bytes" % (before[1], after[1]))
    return after


def check_memory_md():
    """True when MEMORY.md still fits in the loader's window."""
    if not os.path.exists(MEMORY_MD):
        return True, 0, 0
    with open(MEMORY_MD, "rb") as fh:
        raw = fh.read()
    n_lines = raw.count(b"\n") + 1
    ok = n_lines <= MEMORY_MD_MAX_LINES and len(raw) <= MEMORY_MD_MAX_BYTES
    return ok, n_lines, len(raw)


INDEX_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")


def check_index_links():
    """Index lines in MEMORY.md that point at a memory which no longer exists.

    MEMORY.md is the only thing every session loads, so a pointer into nothing
    is a rule that silently stopped existing. Forty memories were deleted on
    2026-09-13 and any one of them could have been linked here.

    Only the link is checked, never the wording. A hook line is SUPPOSED to read
    differently from the note's description: the hook carries the surprising
    part and the description carries the subject. Measured on the current file,
    a word-overlap test flagged twelve lines and all twelve were correct, so
    that check would be pure noise and is deliberately absent.
    """
    dead = []
    if not os.path.exists(MEMORY_MD):
        return dead
    for n, line in enumerate(open(MEMORY_MD, encoding="utf-8", errors="replace"), 1):
        for label, target in INDEX_LINK.findall(line):
            if not os.path.exists(os.path.join(MEM_ROOT, target)):
                dead.append((n, label.strip(), target))
    return dead


REVIEW_MD = "REVIEW.md"

REVIEW_INTRO = {
    "unverified": ("Memories the world no longer agrees with",
                   "A probe checked what the memory asserts and found a path, branch, PR, "
                   "secret or project that is gone. Recall already labels these; they are "
                   "listed so you can correct or retire them."),
    "contradict": ("Two memories that disagree",
                   "The curator would not pick a winner on its own. Both are live, so both "
                   "can be injected, which is how a rule stops holding."),
    "duplicate":  ("Two memories saying the same thing",
                   "Harmless individually. Each one competes for the four slots a prompt "
                   "gets, so the store gets quieter as these close."),
    "update":     ("Something proposed and never applied",
                   "A capture wanted to change an existing memory. Nothing was written."),
}


def write_review_md(dry=False):
    """Regenerate REVIEW.md from the live queue.

    Nothing regenerated it at first, so for eleven days it described items that
    had already been closed. A queue nobody can see is a queue that only grows.
    """
    con = connect()
    rows = con.execute(
        "SELECT id, kind, rels, detail FROM review WHERE status='open' "
        "ORDER BY kind, id").fetchall()
    con.close()
    body = ["# Needs your call", "",
            "Auto-generated %s. **%d open.**" % (time.strftime("%Y-%m-%d %H:%M"), len(rows)), ""]
    if not rows:
        body += ["Nothing is waiting. Everything the system could settle by itself is closed.", ""]
    else:
        body += ["Everything the system could settle by itself has already been closed.",
                 "Close one with `mem resolve <id>` or `mem dismiss <id>`.", ""]
        by_kind = {}
        for rid, kind, rels, detail in rows:
            by_kind.setdefault(kind, []).append((rid, rels, detail))
        for kind in ("contradict", "duplicate", "update", "unverified"):
            group = by_kind.pop(kind, [])
            if not group:
                continue
            title, why = REVIEW_INTRO.get(kind, (kind, ""))
            body += ["## %s (%d)" % (title, len(group)), "", why, ""]
            for rid, rels, detail in group:
                body.append("- **#%d** `%s`  " % (rid, (rels or "").strip()))
                body.append("  %s" % (detail or "").strip().replace("\n", " ")[:400])
            body.append("")
        for kind, group in by_kind.items():
            body += ["## %s (%d)" % (kind, len(group)), ""]
            for rid, rels, detail in group:
                body.append("- **#%d** `%s`  " % (rid, (rels or "").strip()))
                body.append("  %s" % (detail or "").strip()[:400])
            body.append("")
    text = "\n".join(body)
    if not dry:
        write_atomic(os.path.join(MEM_ROOT, REVIEW_MD), text)
    return len(rows)


# ---------------------------------------------------------------- driver


def run(limit=400, model="haiku", workers=4, batch_size=6, dry=False, verbose=True):
    recall.build_index()
    con = connect()
    t0 = time.time()

    compact_memory_md(dry=dry, verbose=verbose)
    dups = pass_duplicates(con, dry)
    flagged, cleared = pass_staleness(con, dry)
    orphans = pass_orphans(con, dry)
    if not dry:
        con.commit()
    if verbose:
        print("deterministic: %d duplicate set(s), %d newly stale, %d cleared, %d orphan rows"
              % (len(dups), len(flagged), len(cleared), orphans))
        for rel, note in flagged[:6]:
            print("   stale  %s  (%s)" % (rel, note))

    # Check what the memories claim against what is actually true. Extraction
    # is incremental and hash-keyed; running the probes costs nothing.
    try:
        verify.extract(limit=60, verbose=False)
        verify.run(verbose=verbose)
    except Exception as exc:
        if verbose:
            print("verification pass skipped: %s" % str(exc)[:120])

    pairs = candidate_pairs(con, limit=limit)
    if verbose:
        print("model pass: %d candidate pair(s)" % len(pairs))
    if not pairs:
        con.close()
        return

    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    actions, checked, failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_batch, b, model): b for b in batches}
        for fut in as_completed(futures):
            try:
                verdicts = fut.result()
            except LLMError as exc:
                failed += 1
                log_line("memory-curate.log", "ERROR", str(exc)[:200])
                continue
            checked += len(verdicts)
            with MemoryLock():
                c = connect()
                for v in verdicts:
                    act = apply_verdict(c, v, dry)
                    if act:
                        actions.append(act)
                c.commit()
                c.close()

    con.close()
    # The queue is only useful if the user can see it, so REVIEW.md is
    # regenerated whenever the queue moves.
    try:
        n_open = write_review_md(dry=dry)
        if verbose:
            print("REVIEW.md: %d open item(s)" % n_open)
    except Exception as exc:
        log_line("memory-curate.log", "REVIEW-MD-FAILED", str(exc)[:160])
    log_line("memory-curate.log", "RUN", "pairs=%d" % checked,
             "actions=%d" % len(actions), "failed=%d" % failed,
             "%.0fs" % (time.time() - t0))
    if verbose:
        print("checked %d pair(s), %d action(s), %d batch failure(s), %.0fs%s"
              % (checked, len(actions), failed, time.time() - t0,
                 "  [DRY RUN]" if dry else ""))
        for a in actions[:25]:
            print("   " + a)


def main():
    ap = argparse.ArgumentParser(prog="curate.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("run")
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--model", default="haiku")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("stale")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("compact")
    p.add_argument("--dry-run", action="store_true")

    sub.add_parser("check")

    p = sub.add_parser("review")
    p.add_argument("--all", action="store_true")

    p = sub.add_parser("audit")
    p.add_argument("--n", type=int, default=40)

    p = sub.add_parser("undo")
    p.add_argument("rel")

    for verb in ("resolve", "dismiss"):
        p = sub.add_parser(verb)
        p.add_argument("ids", nargs="+", type=int)
        p.add_argument("--note", default="")

    args = ap.parse_args()

    if args.cmd == "run":
        run(limit=args.limit, model=args.model, workers=args.workers,
            batch_size=args.batch, dry=args.dry_run)
        return 0

    if args.cmd == "stale":
        con = connect()
        dups = pass_duplicates(con, args.dry_run)
        flagged, cleared = pass_staleness(con, args.dry_run)
        orphans = pass_orphans(con, args.dry_run)
        if not args.dry_run:
            con.commit()
        con.close()
        print("%d duplicate set(s), %d stale, %d cleared, %d orphans%s"
              % (len(dups), len(flagged), len(cleared), orphans,
                 "  [DRY RUN]" if args.dry_run else ""))
        for rel, note in flagged:
            print("   %s  %s" % (rel, note))
        return 0

    if args.cmd == "compact":
        with MemoryLock():
            compact_memory_md(dry=args.dry_run)
        return 0

    if args.cmd == "check":
        ok, n_lines, n_bytes = check_memory_md()
        print("MEMORY.md: %d lines, %d bytes -> %s" % (
            n_lines, n_bytes, "fits" if ok else "TRUNCATED, run `curate.py compact`"))
        dead = check_index_links()
        if dead:
            print("dead index links: %d (a rule that stopped existing)" % len(dead))
            for n, label, target in dead:
                print("   line %-4d %-38s -> %s" % (n, label[:38], target))
        else:
            print("index links: every pointer resolves")
        return 0 if (ok and not dead) else 1

    if args.cmd == "review":
        con = connect()
        q = "SELECT id, ts, kind, rels, detail, status FROM review"
        if not args.all:
            q += " WHERE status='open'"
        q += " ORDER BY ts DESC LIMIT 60"
        rows = con.execute(q).fetchall()
        con.close()
        if not rows:
            print("review queue is empty")
            return 0
        for rid, ts, kind, rels, detail, status in rows:
            print("#%-4d %s  %-11s %-8s %s" % (
                rid, time.strftime("%Y-%m-%d", time.localtime(ts)), kind, status, rels))
            print("      %s" % detail)
        return 0

    if args.cmd == "audit":
        con = connect()
        rows = con.execute(
            "SELECT ts, action, target, detail, actor FROM audit "
            "ORDER BY ts DESC LIMIT ?", (args.n,)).fetchall()
        con.close()
        for ts, action, target, detail, actor in rows:
            print("%s  %-14s %-12s %s" % (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)), action, actor, target))
            if detail:
                print("      %s" % detail)
        if not rows:
            print("no audit entries yet")
        return 0

    if args.cmd in ("resolve", "dismiss"):
        # `resolve` means the conflict was real and has been dealt with.
        # `dismiss` means it was not a conflict at all. Both close the item and
        # both leave an audit row, because a wrongly dismissed contradiction is
        # exactly the kind of thing that needs to be findable later.
        with MemoryLock():
            con = connect()
            for rid in args.ids:
                row = con.execute("SELECT rels FROM review WHERE id=?", (rid,)).fetchone()
                if not row:
                    print("#%d not found" % rid)
                    continue
                status = "resolved" if args.cmd == "resolve" else "dismissed"
                con.execute("UPDATE review SET status=? WHERE id=?", (status, rid))
                audit(con, "review-" + args.cmd, row[0], args.note, "user")
                print("#%-3d %-9s %s" % (rid, status, row[0][:70]))
            con.commit()
            con.close()
        try:
            print("REVIEW.md: %d open item(s) left" % write_review_md())
        except Exception:
            pass
        return 0

    if args.cmd == "undo":
        with MemoryLock():
            con = connect()
            cur = con.execute(
                "UPDATE doc SET superseded_by='', stale_note='' WHERE rel LIKE ?",
                ("%" + args.rel + "%",))
            audit(con, "undo", args.rel, "flags cleared by hand", "user")
            con.commit()
            n = cur.rowcount
            con.close()
        print("cleared flags on %d memory file(s)" % n)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

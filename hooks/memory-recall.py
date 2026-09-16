#!/usr/bin/env python3
"""
UserPromptSubmit hook: automatic memory recall.

v1 fired on the literal prompt text and scored about 17% precision on real
sessions while scoring 96% on a synthetic battery of clean topical queries. The
gap was the query, not the index: a real prompt reads "still broken, look at
this", carries pasted images and URLs, and task notifications come through too,
none of which is a search query. So this version:

  - reads the session transcript and searches the WORKING TOPIC, not one line
  - strips images, URLs, code fences and pasted blocks out of the query
  - refuses to fire on machine-generated turns (task notifications, slash
    command envelopes, hook output)
  - stays silent once a topic has been served this session
  - never runs a model: this hook blocks the user's prompt, and a `claude -p` round
    trip is 8+ seconds. All model work is offline, in the index.

Any failure exits 0 silently. A memory system that can break the prompt is
worse than no memory system.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
RECALL = os.path.join(SRC, "recall.py")
sys.path.insert(0, SRC)
from memlib import LOG_DIR, STATE_ROOT          # noqa: E402
STATE_DIR = os.path.join(STATE_ROOT, "sessions")

LIMIT = 4
MAX_FACTS = 2
BUDGET = 1400
# A single hook's stdout is silently replaced by a stub past ~10,000 chars with
# no error reported, so stay far under it.
HARD_BUDGET = 8000
MAX_SECONDS = 8
CONTEXT_TURNS = 6
TOPIC_SERVED_RATIO = 0.6
CONTEXT_CHARS = 700

# Turns the harness generates. None of them is the user asking for something.
MACHINE_PREFIXES = (
    "<task-notification", "<command-name", "<command-message", "<local-command",
    "<system-reminder", "<bash-input", "<bash-stdout", "Caveat:",
    "[Request interrupted", "API Error", "<user-prompt-submit-hook",
)

NOISE = [
    (re.compile(r"<task-notification>.*?</task-notification>", re.S), " "),
    (re.compile(r"<system-reminder>.*?</system-reminder>", re.S), " "),
    (re.compile(r"<local-command-[a-z]+>.*?</local-command-[a-z]+>", re.S), " "),
    (re.compile(r"<command-[a-z]+>.*?</command-[a-z]+>", re.S), " "),
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"\[Image #\d+\]"), " "),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"[│┃▎|]+"), " "),
]


def clean(text):
    for pattern, repl in NOISE:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


# Batch prompts a script feeds to `claude -p`. A scripted translation job can
# fire recall dozens of times a week on "Translate the following article...",
# which is not a person asking anything.
BATCH_SHAPE = re.compile(
    r"(?is)^(translate|summari[sz]e|classify|extract|rewrite|convert) the following"
    r"|return (only )?(a |the )?json( object)?|respond (only )?(with|in) json"
    r"|output (only )?json|no (prose|preamble|explanation)")


def is_machine_turn(prompt):
    head = prompt.lstrip()
    if head.startswith(MACHINE_PREFIXES):
        return True
    return bool(BATCH_SHAPE.search(head[:600]))


def session_context(transcript_path):
    """The last few real turns, so the query knows what we are working on."""
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    try:
        with open(transcript_path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()[-400:]
    except OSError:
        return ""

    turns = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") not in ("user", "assistant"):
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
        content = clean(content)
        if len(content) < 12 or is_machine_turn(content):
            continue
        turns.append(content[:CONTEXT_CHARS])

    return " ".join(turns[-CONTEXT_TURNS:])


def load_state(session_id):
    path = os.path.join(STATE_DIR, "%s.json" % session_id)
    try:
        with open(path) as fh:
            d = json.load(fh)
        return path, set(d.get("seen", [])), set(d.get("terms", []))
    except Exception:
        return path, set(), set()


def save_state(path, seen, terms):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"seen": sorted(seen), "terms": sorted(terms), "ts": time.time()}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def prune_state():
    try:
        cutoff = time.time() - 7 * 86400
        for fn in os.listdir(STATE_DIR):
            p = os.path.join(STATE_DIR, fn)
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
    except Exception:
        pass


def log(session_id, prompt, hits, note=""):
    try:
        d = LOG_DIR
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "memory-recall.log"), "a") as fh:
            fh.write("%s\t%s\t%s\t%s\t%s\n" % (
                time.strftime("%Y-%m-%d %H:%M:%S"), (session_id or "?")[:8],
                prompt[:90].replace("\t", " ").replace("\n", " "),
                ",".join(h["rel"] for h in hits) or "-", note))
    except Exception:
        pass


def main():
    # `claude -p` fires this hook too. Without the guard, an offline enrichment
    # or capture run would recurse into itself.
    if os.environ.get("QUIPU_CHILD"):
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    raw_prompt = (payload.get("prompt") or "").strip()
    session_id = payload.get("session_id") or "nosession"
    if not raw_prompt or is_machine_turn(raw_prompt):
        return 0

    prompt = clean(raw_prompt)
    if len(prompt) < 10:
        return 0

    context = session_context(payload.get("transcript_path"))

    try:
        proc = subprocess.run(
            [sys.executable, RECALL, "query", prompt[:1500],
             "--context", context[:4000],
             "--limit", str(LIMIT + 3), "--json"],
            capture_output=True, text=True, timeout=MAX_SECONDS,
        )
        hits = json.loads(proc.stdout or "[]")
    except Exception:
        return 0
    if not hits:
        log(session_id, prompt, [], "no-hit")
        return 0

    # Anything well below the best hit for this prompt is filler.
    top = max(h.get("score", 0) for h in hits)
    strong = [h for h in hits if h.get("score", 0) >= top * 0.72]

    state_path, seen, served_terms = load_state(session_id)

    # Topic silence. Compare the SUBJECT the user named, not the top hit's evidence:
    # "deploy the api to production" and "ok ship the api to prod, go" are
    # the same question but rank different documents first, so an exact-subset
    # test on hit evidence lets the second one through carrying the tail of the
    # ranking. Two thirds of the subject already served means covered.
    topic = set(strong[0].get("query_terms") or strong[0].get("evidence", []))
    if topic and len(topic & served_terms) / float(len(topic)) >= TOPIC_SERVED_RATIO:
        log(session_id, prompt, [], "topic-served")
        return 0

    # At most MAX_FACTS fact notes per injection. Measured on MemConflict
    # 2026-09-14: fact notes ranked freely filled every slot, and accuracy on
    # questions that need the surrounding conversation fell from 0.70 to 0.06;
    # capped at two of five it recovered to 0.61 while change questions kept
    # most of their doubling. A fact is one line; the rest of the window is for
    # the rules and context that explain it.
    fresh, n_facts = [], 0
    for h in strong:
        if h["rel"] in seen:
            continue
        if os.path.basename(h["rel"]).startswith("fact_"):
            if n_facts >= MAX_FACTS:
                continue
            n_facts += 1
        fresh.append(h)
        if len(fresh) >= LIMIT:
            break
    if not fresh:
        log(session_id, prompt, [], "all-seen")
        return 0

    lines, used = [], 0
    for h in fresh:
        note = ""
        if h.get("superseded_by"):
            note = " [SUPERSEDED by %s]" % h["superseded_by"]
        elif h.get("stale_note"):
            flag = h["stale_note"]
            # A claim that no longer checks out is a different
            # warning from a dead file path. Say which.
            if flag.startswith("unverified:"):
                note = " [UNVERIFIED: %s]" % flag.split(":", 1)[1].strip()
            else:
                note = " [STALE: %s]" % flag
        snippet = (h.get("descr") or "").strip() or h.get("body", "")
        snippet = re.sub(r"\s+", " ", snippet)[:230]
        line = "- **%s** (`%s`)%s: %s" % (h["title"], h["rel"], note, snippet)
        if used + len(line) > BUDGET and lines:
            break
        used += len(line)
        lines.append(line)
        seen.add(h["rel"])

    save_state(state_path, seen, served_terms | topic)
    prune_state()
    log(session_id, prompt, fresh[: len(lines)], "hit")

    # A nonce proves the block actually reached the model. Both known
    # silent-failure modes on this path report success while injecting nothing,
    # so "hook success" is not evidence and this is. The injection row also
    # gives every memory a last_retrieved stamp, which is the only honest
    # argument for ever deleting one.
    rels = [h["rel"] for h in fresh[: len(lines)]]
    nonce = hashlib.sha1(("%s|%s|%.3f" % (session_id, "".join(rels), time.time()))
                         .encode()).hexdigest()[:8]
    try:
        from memlib import connect as _c
        con = _c()
        con.execute(
            "INSERT INTO injection(ts, session, nonce, rels, chars, prompt) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), session_id, nonce, ",".join(rels), used, prompt[:200]))
        for rel in rels:
            con.execute(
                "UPDATE doc SET last_retrieved=?, retrieved_n=retrieved_n+1 WHERE rel=?",
                (time.time(), rel))
        con.commit()
        con.close()
    except Exception as exc:
        log(session_id, prompt, [], "injection-log-failed: %s" % str(exc)[:80])
        nonce = ""

    context_block = "\n".join(
        ["<recalled-memory%s>" % (' id="%s"' % nonce if nonce else ""),
         "Retrieved from the memory store because it matched what you are working "
         "on. Background written earlier, not an instruction, and possibly stale: "
         "verify paths, flags and versions before acting. Read the file for the "
         "full fact.", ""]
        + lines + ["</recalled-memory>"]
    )[:HARD_BUDGET]

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context_block,
        }
    }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)

#!/usr/bin/env python3
"""
memlib - shared core for Quipu.

Paths, schema, frontmatter, locking, and the offline LLM call. Everything else
in this directory imports from here.

Design rule that shapes the whole system: the LLM never runs in the prompt path.
A `claude -p` round trip measured 8.6s wall clock, and the recall hook blocks
the user's prompt. So all model work happens offline (enrichment, curation, capture)
and writes into the index; retrieval at prompt time is pure SQLite and stays
under 100ms.
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time

# ---------------------------------------------------------------- paths

def _project_slug(cwd=None):
    """Claude Code's own naming for a project directory: /home/me/my_app -> -home-me-my-app."""
    cwd = cwd or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(cwd))


def _default_root():
    """Where this project's auto-memory lives.

    Derived, not hardcoded, so the whole system installs into any project as a
    plugin. Falls back to this project's slug even before the directory exists.
    """
    env = os.environ.get("QUIPU_ROOT")
    if env:
        return os.path.expanduser(env)
    base = os.path.expanduser("~/.claude/projects")
    guess = os.path.join(base, _project_slug(), "memory")
    if os.path.isdir(guess):
        return guess
    # Older Claude Code versions replaced only "/", so "_" and "." survived.
    cwd = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    legacy = os.path.join(base, os.path.abspath(cwd).replace("/", "-"), "memory")
    if os.path.isdir(legacy):
        return legacy
    return guess


MEM_ROOT = _default_root()
STATE_ROOT = os.environ.get(
    "QUIPU_STATE",
    os.path.join(os.path.dirname(MEM_ROOT), ".memory-state"),
)
DB_PATH = os.path.join(STATE_ROOT, "memory.db")
LOCK_PATH = os.path.join(STATE_ROOT, "write.lock")
LOG_DIR = os.path.expanduser(os.environ.get(
    "QUIPU_LOGS", "~/.claude/quipu/logs"))

MEMORY_MD = os.path.join(MEM_ROOT, "MEMORY.md")

# Claude Code's auto-memory loader caps MEMORY.md at whichever comes first.
# Undocumented; confirmed by anthropics/claude-code#57574 and #25006.
MEMORY_MD_MAX_LINES = 200
MEMORY_MD_MAX_BYTES = 25000

SKIP_NAMES = {"MEMORY.md", "README.md", "AGENTS.md", "CLAUDE.md"}
SKIP_PREFIX = ("MEMORY.", ".")

# A rule does not get less true with age; a project status does. Half-life in
# days, per memory type, applied to the ranking score at read time. `None`
# means the memory never decays.
HALF_LIFE_DAYS = {
    "feedback": None,
    "user": None,
    "reference": 540.0,
    "project": 150.0,
    "index": 150.0,
    "note": 240.0,
    # A fact note carries its own history: the current value on top, every
    # earlier value below it with the date it stopped being true. Currency is
    # recorded inside the file, so the file itself must never fade.
    "fact": None,
}

# Rules outrank status reports when both match a prompt.
# A tiebreaker, not a thumb on the scale. Set too high, a weakly-matching rule
# outranks a memory that actually answers the question.
TYPE_WEIGHT = {
    "feedback": 1.15,
    "user": 1.12,
    "reference": 1.06,
    "project": 1.0,
    "index": 0.95,
    "note": 1.0,
    "fact": 1.10,
}


# ---------------------------------------------------------------- schema

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);

CREATE TABLE IF NOT EXISTS doc (
  id       INTEGER PRIMARY KEY,
  rel      TEXT NOT NULL UNIQUE,
  path     TEXT NOT NULL,
  name     TEXT NOT NULL,
  descr    TEXT NOT NULL DEFAULT '',
  kind     TEXT NOT NULL DEFAULT 'note',
  hash     TEXT NOT NULL,
  mtime    REAL NOT NULL,
  bytes    INTEGER NOT NULL DEFAULT 0,
  status   TEXT NOT NULL DEFAULT 'live',
  supersedes   TEXT NOT NULL DEFAULT '',
  superseded_by TEXT NOT NULL DEFAULT '',
  stale_note   TEXT NOT NULL DEFAULT '',
  -- Last time this memory was actually put in front of the model. A memory
  -- never retrieved across hundreds of sessions is the only honest argument
  -- for deleting it; write-time tells you nothing.
  last_retrieved REAL NOT NULL DEFAULT 0,
  retrieved_n    INTEGER NOT NULL DEFAULT 0
);

-- LLM-derived retrieval aids, keyed by content hash so a re-run is free
-- for unchanged files.
CREATE TABLE IF NOT EXISTS enrich (
  hash     TEXT PRIMARY KEY,
  rel      TEXT NOT NULL,
  topics   TEXT NOT NULL DEFAULT '',
  aliases  TEXT NOT NULL DEFAULT '',
  triggers TEXT NOT NULL DEFAULT '',
  entities TEXT NOT NULL DEFAULT '',
  summary  TEXT NOT NULL DEFAULT '',
  model    TEXT NOT NULL DEFAULT '',
  ts       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk (
  id      INTEGER PRIMARY KEY,
  doc_id  INTEGER NOT NULL,
  rel     TEXT NOT NULL,
  title   TEXT NOT NULL,
  descr   TEXT NOT NULL,
  kind    TEXT NOT NULL,
  body    TEXT NOT NULL,
  aids    TEXT NOT NULL DEFAULT '',
  mtime   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chunk_doc ON chunk(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  title, descr, aids, body,
  content='chunk', content_rowid='id',
  tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
  INSERT INTO fts(rowid, title, descr, aids, body)
  VALUES (new.id, new.title, new.descr, new.aids, new.body);
END;
CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
  INSERT INTO fts(fts, rowid, title, descr, aids, body)
  VALUES ('delete', old.id, old.title, old.descr, old.aids, old.body);
END;

-- Every control-plane mutation is recorded. A losing fact is never deleted
-- silently; the audit row is how a bad merge gets found later.
CREATE TABLE IF NOT EXISTS audit (
  id     INTEGER PRIMARY KEY,
  ts     REAL NOT NULL,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  actor  TEXT NOT NULL DEFAULT ''
);

-- Checkable assertions extracted from memories, one row per claim. `plane`
-- decides what a failure is allowed to mean: a control-plane probe (does this
-- file/branch/secret/instance exist) is authoritative, a data-plane probe
-- (does this URL respond) can prove life but never death, because unreachable
-- from this machine is not the same as down.
CREATE TABLE IF NOT EXISTS claim (
  id      INTEGER PRIMARY KEY,
  rel     TEXT NOT NULL,
  hash    TEXT NOT NULL,
  probe   TEXT NOT NULL,
  args    TEXT NOT NULL DEFAULT '{}',
  expect  TEXT NOT NULL DEFAULT '',
  quote   TEXT NOT NULL DEFAULT '',
  plane   TEXT NOT NULL DEFAULT 'control',
  status  TEXT NOT NULL DEFAULT 'unchecked',
  detail  TEXT NOT NULL DEFAULT '',
  checked REAL NOT NULL DEFAULT 0,
  created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS claim_rel  ON claim(rel);
CREATE UNIQUE INDEX IF NOT EXISTS claim_uniq ON claim(hash, probe, args);

-- Every injection, so "why did it do that" is a lookup instead of an
-- investigation. Two silent-failure modes sit on this path: plugin-delivered
-- UserPromptSubmit output can be discarded while reporting success
-- (anthropics/claude-code#12151), and a single hook's output is replaced by a
-- stub past ~10,000 chars with no error. A nonce written here and echoed in the
-- block is the only way to know the context actually landed.
CREATE TABLE IF NOT EXISTS injection (
  id      INTEGER PRIMARY KEY,
  ts      REAL NOT NULL,
  session TEXT NOT NULL,
  nonce   TEXT NOT NULL,
  rels    TEXT NOT NULL DEFAULT '',
  chars   INTEGER NOT NULL DEFAULT 0,
  prompt  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS injection_ts ON injection(ts);

-- Things the curator wants a human to look at rather than decide alone.
CREATE TABLE IF NOT EXISTS review (
  id      INTEGER PRIMARY KEY,
  ts      REAL NOT NULL,
  kind    TEXT NOT NULL,
  rels    TEXT NOT NULL,
  detail  TEXT NOT NULL,
  status  TEXT NOT NULL DEFAULT 'open'
);
"""


# CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a new column
# needs an explicit migration or it silently never appears.
MIGRATIONS = [
    ("doc", "last_retrieved", "REAL NOT NULL DEFAULT 0"),
    ("doc", "retrieved_n", "INTEGER NOT NULL DEFAULT 0"),
]


def connect():
    os.makedirs(STATE_ROOT, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=15.0)
    con.executescript(SCHEMA)
    for table, column, decl in MIGRATIONS:
        cols = [r[1] for r in con.execute("PRAGMA table_info(%s)" % table)]
        if column not in cols:
            con.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, decl))
    con.commit()
    return con


def get_meta(con, key, default=None):
    row = con.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(con, key, value):
    con.execute(
        "INSERT INTO meta(k,v) VALUES(?,?) "
        "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, str(value)),
    )


def audit(con, action, target, detail="", actor="curator"):
    con.execute(
        "INSERT INTO audit(ts, action, target, detail, actor) VALUES (?,?,?,?,?)",
        (time.time(), action, target, detail, actor),
    )


# ---------------------------------------------------------------- locking


class MemoryLock(object):
    """Advisory lock around any write to the memory store.

    Several Claude sessions can run at once and they all share this
    directory. Without this, two session-end captures can interleave a
    read-modify-write of MEMORY.md and one loses.
    """

    def __init__(self, timeout=20.0):
        self.timeout = timeout
        self.fh = None

    def __enter__(self):
        os.makedirs(STATE_ROOT, exist_ok=True)
        self.fh = open(LOCK_PATH, "w")
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except IOError:
                if time.time() > deadline:
                    self.fh.close()
                    self.fh = None
                    raise RuntimeError("memory store busy: could not lock in %.0fs" % self.timeout)
                time.sleep(0.15)

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None
        return False


# ---------------------------------------------------------------- files


def parse_frontmatter(text):
    """Return (meta, body). Tolerates malformed or absent frontmatter."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end]
    body = text[end + 4:].lstrip("\n")
    meta = {}
    for line in raw.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip().strip("\"'")
        if v:
            meta.setdefault(k.strip(), v)   # `metadata:` nests `type:`; flatten
    return meta, body


def walk_memory():
    """Yield (abspath, rel, mtime) for every indexable memory file."""
    for dirpath, dirnames, filenames in os.walk(MEM_ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "archive"]
        for fn in sorted(filenames):
            if not fn.endswith(".md") or fn in SKIP_NAMES or fn.startswith(SKIP_PREFIX):
                continue
            p = os.path.join(dirpath, fn)
            try:
                yield p, os.path.relpath(p, MEM_ROOT), os.path.getmtime(p)
            except OSError:
                continue


def read_doc(path):
    """Return (meta, body, sha1, raw) for one memory file."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    meta, body = parse_frontmatter(raw)
    return meta, body, hashlib.sha1(raw.encode("utf-8")).hexdigest(), raw


def doc_kind(meta, rel):
    kind = meta.get("metadata_type") or meta.get("type")
    if kind in HALF_LIFE_DAYS:
        return kind
    if rel.startswith("_index"):
        return "index"
    base = os.path.basename(rel)
    for prefix in ("feedback", "project", "reference", "user", "fact"):
        if base.startswith(prefix + "_"):
            return prefix
    return "note"


def split_sections(body, max_chars=2200):
    """Big files split on headings so a hit points at the relevant section."""
    if len(body) <= max_chars:
        return [("", body)]
    parts = re.split(r"\n(?=#{1,3} )", body)
    out, buf, head_of_buf = [], [], ""
    for part in parts:
        m = re.match(r"#{1,3} (.+)", part)
        head = m.group(1).strip() if m else ""
        if len(part) > max_chars:
            if buf:
                out.append((head_of_buf, "\n".join(buf)))
                buf, head_of_buf = [], ""
            for i in range(0, len(part), max_chars):
                out.append((head, part[i:i + max_chars]))
            continue
        if buf and sum(len(b) for b in buf) + len(part) > max_chars:
            out.append((head_of_buf, "\n".join(buf)))
            buf, head_of_buf = [], ""
        if not buf:
            head_of_buf = head
        buf.append(part)
    if buf:
        out.append((head_of_buf, "\n".join(buf)))
    return out or [("", body)]


def write_atomic(path, text):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    shutil.move(tmp, path)


# ---------------------------------------------------------------- the model


class LLMError(Exception):
    pass


# The benchmark's fairness contract gives every provider the SAME internal
# model (qwen3.5-4b on vllm-gen). A Claude subscription is not available to a
# container and would not be fair anyway, so setting QUIPU_LLM_BASE_URL swaps the
# offline passes onto any OpenAI-compatible endpoint. Unset, nothing changes.
LLM_BASE_URL = os.environ.get("QUIPU_LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("QUIPU_LLM_MODEL", "qwen3.5-4b")
LLM_API_KEY = os.environ.get("QUIPU_LLM_API_KEY", "EMPTY")


# 1,500 is ample: five facts and three memories are about a thousand. A small
# model at temperature 0 can fall into a repetition loop and write until the cap,
# and on the benchmark that cost 100 seconds per conversation on one persona
# (8,192 tokens at single-stream speed, every session) where others took two.
LLM_MAX_TOKENS = int(os.environ.get("QUIPU_LLM_MAX_TOKENS", "1500"))
# Offline extraction wants a JSON answer, not an essay about one. On a reasoning
# model the private think block shares the token budget, so a long think can
# truncate the JSON it was supposed to produce. Off by default for these calls;
# the benchmark's pinned answer and judge settings are separate and untouched.
LLM_THINKING = os.environ.get("QUIPU_LLM_THINKING", "0") in ("1", "true", "yes")


def _openai_chat(prompt, system=None, timeout=300, max_tokens=None):
    import json as _json
    import urllib.request as _u
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    payload = {
        "model": LLM_MODEL, "messages": msgs,
        "temperature": 0.0, "max_tokens": max_tokens or LLM_MAX_TOKENS,
    }
    if not LLM_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = _json.dumps(payload).encode()
    req = _u.Request(LLM_BASE_URL.rstrip("/") + "/chat/completions", data=body,
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer " + LLM_API_KEY})
    try:
        with _u.urlopen(req, timeout=timeout) as r:
            d = _json.loads(r.read().decode())
    except Exception as exc:
        raise LLMError("%s: %s" % (type(exc).__name__, str(exc)[:200]))
    try:
        return (d["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError):
        raise LLMError("unexpected response shape: %s" % str(d)[:200])


def claude(prompt, model="haiku", timeout=300, system=None):
    """Run a prompt through `claude -p`, billed to whatever your Claude CLI uses.

    Never call this from a hook that blocks the prompt: a round trip is
    seconds, not milliseconds. Offline paths only.

    QUIPU_CHILD guards the recursion: `claude -p` fires this machine's own
    SessionStart/UserPromptSubmit/SessionEnd hooks, so without the flag a
    capture run would spawn a capture run.
    """
    if LLM_BASE_URL:
        return _openai_chat(prompt, system=system, timeout=timeout)

    env = dict(os.environ)
    env["QUIPU_CHILD"] = "1"
    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "text",
        # No tools and no MCP servers. The prompt carries transcript text, and
        # anything pasted into a session can carry instructions. The old flags,
        # bypassPermissions plus an empty --allowed-tools, left every tool live:
        # on 2026-09-15 a test prompt through them wrote a file to disk. `--tools ""`
        # alone still left MCP tools that can run code, so both flags are needed.
        "--tools", "",
        "--strict-mcp-config",
    ]
    if system:
        cmd += ["--append-system-prompt", system]
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        raise LLMError("claude -p timed out after %ss" % timeout)
    if proc.returncode != 0:
        raise LLMError("claude -p exit %s: %s" % (proc.returncode, (proc.stderr or "")[:300]))
    return (proc.stdout or "").strip()


JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def claude_json(prompt, model="haiku", timeout=300, system=None, retries=1):
    """claude() that must return JSON. Strips fences, retries once on garbage."""
    last = None
    for attempt in range(retries + 1):
        out = claude(prompt, model=model, timeout=timeout, system=system)
        text = out
        m = JSON_BLOCK.search(out)
        if m:
            text = m.group(1)
        text = text.strip()
        start = min([i for i in (text.find("["), text.find("{")) if i != -1] or [0])
        text = text[start:]
        try:
            return json.loads(text)
        except ValueError as exc:
            last = exc
            prompt = prompt + "\n\nYour previous reply was not valid JSON. Reply with JSON only, no prose, no fences."
    raise LLMError("model did not return JSON: %s" % last)


def rebind(root, state=None):
    """Point the whole system at a different store, in-process.

    The benchmark runs 30 isolated personas in one process, so the store path
    has to move without a restart. Modules that did `from memlib import
    MEM_ROOT` captured a VALUE, not a reference, so this patches them too
    rather than only rebinding here and silently leaving half the system
    pointed at the old directory.
    """
    global MEM_ROOT, STATE_ROOT, DB_PATH, LOCK_PATH, MEMORY_MD
    MEM_ROOT = os.path.abspath(os.path.expanduser(root))
    STATE_ROOT = os.path.abspath(os.path.expanduser(
        state or os.path.join(os.path.dirname(MEM_ROOT), ".memory-state")))
    DB_PATH = os.path.join(STATE_ROOT, "memory.db")
    LOCK_PATH = os.path.join(STATE_ROOT, "write.lock")
    MEMORY_MD = os.path.join(MEM_ROOT, "MEMORY.md")
    os.makedirs(MEM_ROOT, exist_ok=True)
    os.makedirs(STATE_ROOT, exist_ok=True)

    import sys as _sys
    mine = {"recall", "enrich", "curate", "capture", "verify", "sessions",
            "profile", "skills"}
    for name, mod in list(_sys.modules.items()):
        if name.split(".")[-1] not in mine or mod is None:
            continue
        for attr, val in (("MEM_ROOT", MEM_ROOT), ("STATE_ROOT", STATE_ROOT),
                          ("DB_PATH", DB_PATH), ("MEMORY_MD", MEMORY_MD)):
            if hasattr(mod, attr):
                setattr(mod, attr, val)
        if hasattr(mod, "USER_MD"):
            mod.USER_MD = os.path.join(MEM_ROOT, "USER.md")
    return MEM_ROOT, STATE_ROOT


def log_line(name, *fields):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, name), "a") as fh:
            fh.write(
                "\t".join(
                    [time.strftime("%Y-%m-%d %H:%M:%S")]
                    + [str(f).replace("\t", " ").replace("\n", " ") for f in fields]
                )
                + "\n"
            )
    except Exception:
        pass

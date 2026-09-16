#!/usr/bin/env python3
"""
sessions.py - search everything that was ever said, not just what got written down.

The memory store holds distilled facts: ~470 notes someone decided were worth
keeping. That is the right thing to inject on every prompt, and it is a lossy
record. "What did we decide about the pricing model three weeks ago" is not a
fact anyone wrote down, it is a conversation.

There are 604 session transcripts sitting on disk, 4.7GB of them. Only 3% is
actual conversation; the rest is tool output nobody needs to search. So this
strips them down to the human text and indexes that: 4.7GB becomes ~140MB of
searchable dialogue, queried in milliseconds with no model call at all.

This is the layer Hermes calls `session_search`, and their docs are right about
why it matters: memory is for facts that must always be in context, session
search is for "did we discuss X last week". Different jobs, different costs.

Usage:
  sessions.py index [--all] [--since DAYS]
  sessions.py search "query" [--limit N] [--json]
  sessions.py show <session-id> [--grep TEXT]
  sessions.py stats
"""

import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memlib import MEM_ROOT, MemoryLock, connect, get_meta, set_meta  # noqa: E402

TRANSCRIPT_DIR = os.environ.get(
    "QUIPU_TRANSCRIPTS",
    os.path.dirname(MEM_ROOT),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
  id       TEXT PRIMARY KEY,
  path     TEXT NOT NULL,
  started  REAL NOT NULL,
  ended    REAL NOT NULL,
  turns    INTEGER NOT NULL DEFAULT 0,
  bytes    INTEGER NOT NULL DEFAULT 0,
  opening  TEXT NOT NULL DEFAULT '',
  indexed  REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turn (
  id      INTEGER PRIMARY KEY,
  sid     TEXT NOT NULL,
  seq     INTEGER NOT NULL,
  role    TEXT NOT NULL,
  ts      REAL NOT NULL,
  text    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turn_sid ON turn(sid);
CREATE VIRTUAL TABLE IF NOT EXISTS turn_fts USING fts5(
  text, content='turn', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS turn_ai AFTER INSERT ON turn BEGIN
  INSERT INTO turn_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS turn_ad AFTER DELETE ON turn BEGIN
  INSERT INTO turn_fts(turn_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""

# Machine chatter that is not conversation.
NOISE = re.compile(
    r"<(task-notification|system-reminder|local-command-[a-z]+|command-[a-z]+|"
    r"user-prompt-submit-hook|recalled-memory)>.*?</\1>", re.S)
MACHINE_START = (
    "<task-notification", "<system-reminder", "<local-command", "<command-name",
    "Caveat:", "[Request interrupted", "<user-prompt-submit-hook",
)
MIN_TURN_CHARS = 25


def clean(text):
    text = NOISE.sub(" ", text)
    text = re.sub(r"```.*?```", " [code] ", text, flags=re.S)
    return re.sub(r"\s+", " ", text).strip()


def read_turns(path):
    """Pull the human-readable dialogue out of a transcript, skipping tool output."""
    out = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
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
                        if isinstance(c, dict) and c.get("type") == "text")
                if not isinstance(content, str):
                    continue
                text = clean(content)
                if len(text) < MIN_TURN_CHARS or text.startswith(MACHINE_START):
                    continue
                ts = row.get("timestamp") or ""
                out.append((row["type"], ts, text[:6000]))
    except OSError:
        return []
    return out


def parse_ts(value, fallback):
    if not value:
        return fallback
    try:
        return time.mktime(time.strptime(value[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return fallback


def index(all_of_them=False, since_days=None, verbose=True):
    con = connect()
    con.executescript(SCHEMA)
    files = sorted(
        (p for p in os.listdir(TRANSCRIPT_DIR) if p.endswith(".jsonl")),
        key=lambda p: os.path.getmtime(os.path.join(TRANSCRIPT_DIR, p)))
    cutoff = time.time() - since_days * 86400 if since_days else 0

    known = {r[0]: r[1] for r in con.execute("SELECT id, indexed FROM session")}
    todo = []
    for fn in files:
        path = os.path.join(TRANSCRIPT_DIR, fn)
        mtime = os.path.getmtime(path)
        if mtime < cutoff:
            continue
        sid = fn[:-6]
        # Re-read a session only when it has grown since we last looked.
        if not all_of_them and known.get(sid, 0) >= mtime:
            continue
        todo.append((sid, path, mtime))

    if not todo:
        con.close()
        if verbose:
            print("session index is current")
        return 0

    if verbose:
        print("indexing %d session(s)" % len(todo))

    n_turns = 0
    t0 = time.time()
    with MemoryLock():
        for i, (sid, path, mtime) in enumerate(todo, 1):
            turns = read_turns(path)
            if not turns:
                continue
            con.execute("DELETE FROM turn WHERE sid=?", (sid,))
            first = parse_ts(turns[0][1], mtime)
            last = parse_ts(turns[-1][1], mtime)
            opening = next((t for r, _ts, t in turns if r == "user"), "")[:300]
            con.execute(
                "INSERT INTO session(id, path, started, ended, turns, bytes, opening, indexed) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                " ended=excluded.ended, turns=excluded.turns, bytes=excluded.bytes, "
                " opening=excluded.opening, indexed=excluded.indexed",
                (sid, path, first, last, len(turns), os.path.getsize(path),
                 opening, mtime))
            for seq, (role, ts, text) in enumerate(turns):
                con.execute(
                    "INSERT INTO turn(sid, seq, role, ts, text) VALUES (?,?,?,?,?)",
                    (sid, seq, role, parse_ts(ts, mtime), text))
                n_turns += 1
            if verbose and i % 50 == 0:
                print("  %d/%d sessions, %d turns (%.0fs)" % (i, len(todo), n_turns, time.time() - t0))
        set_meta(con, "sessions_indexed", time.time())
        con.commit()
    con.close()
    if verbose:
        print("indexed %d turns from %d session(s) in %.0fs"
              % (n_turns, len(todo), time.time() - t0))
    return n_turns


def fts_escape(text):
    words = re.findall(r"[A-Za-z0-9_.-]{2,}", text.lower())
    return " OR ".join('"%s"' % w.replace('"', '""') for w in words[:12])


# --- recency: MEASURED, AND IT DOES NOT WORK. DEFAULT OFF. ---------------------
# The idea was sound and the measurement killed it. bm25 answers "which turn is
# most ABOUT this" and not "which turn is still TRUE", so a freshness term looked
# like the obvious tie-breaker for a question whose answer changed.
#
# Measured 2026-09-13 on 334 dynamic-conflict questions from MemConflict, five
# personas, roughly a year of dialogue each, asking whether the turn carrying the
# answer lands in the top five:
#
#     recency weight 0.00 -> 47.9%      <- best
#     recency weight 0.35 -> 44.3%
#     recency weight 0.80 -> 37.7%
#     recency weight 1.50 -> 32.6%
#
# Monotonically worse. Every increment displaces the topically correct turn with
# a more recent, less relevant one. The current answer being recent does not make
# the turn that states it the most recent turn in the log.
#
# The same run found where the recall actually goes, and it is not staleness:
# taking the best turns overall instead of one per session gives 47.9% at depth
# five, 58.7% at ten, 71.9% at twenty and 85.0% at fifty. The evidence is in the
# index and it is below the cut. DEPTH is the lever, re-ranking on time is not.
#
# The knob stays so the arms can be re-run, and it stays at zero. Do not raise it
# without a measurement that beats 47.9% on that set.
RECENCY_WEIGHT = float(os.environ.get("QUIPU_SESSION_RECENCY_WEIGHT", "0"))
RECENCY_HALF_LIFE_DAYS = float(os.environ.get("QUIPU_SESSION_HALF_LIFE", "45"))
RECENCY_FLOOR = float(os.environ.get("QUIPU_SESSION_FLOOR", "0.45"))


def freshness(ts, now=None):
    """1.0 for something said now, decaying by half-life down to RECENCY_FLOOR."""
    now = time.time() if now is None else now
    age_days = max(0.0, (now - (ts or 0)) / 86400.0)
    if RECENCY_HALF_LIFE_DAYS <= 0:
        return 1.0
    decay = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * decay


def search(query, limit=6, role=None, days=None, per_session=True, pool=None):
    con = connect()
    con.executescript(SCHEMA)
    q = fts_escape(query)
    if not q:
        con.close()
        return []
    sql = ("SELECT t.sid, t.role, t.ts, t.text, s.opening, bm25(turn_fts) AS r "
           "FROM turn_fts JOIN turn t ON t.id = turn_fts.rowid "
           "JOIN session s ON s.id = t.sid WHERE turn_fts MATCH ?")
    params = [q]
    if role:
        sql += " AND t.role = ?"
        params.append(role)
    if days:
        sql += " AND t.ts > ?"
        params.append(time.time() - days * 86400)
    sql += " ORDER BY r LIMIT ?"
    # The pool is what the ranking gets to choose from, and it is the whole game.
    # Measured on 334 questions whose answer changed over time: the turn carrying
    # the answer is in the top 5 of bm25 47.9% of the time, the top 20 71.9%, the
    # top 50 85.0%. Anything that reorders a pool of five cannot reach the other
    # thirty-seven points. Fetch wide, cut late.
    params.append(pool if pool else limit * 12)
    try:
        rows = con.execute(sql, params).fetchall()
    except Exception:
        con.close()
        return []
    con.close()
    if not rows:
        return []

    # bm25() is negative and more negative is better, so flip it and normalise
    # inside this pool. Normalising per query keeps the weight below meaningful
    # regardless of how large the raw scores happen to be for these terms.
    rels = [-row[5] for row in rows]
    lo, hi = min(rels), max(rels)
    span = (hi - lo) or 1.0
    now = time.time()

    scored = []
    for row, rel in zip(rows, rels):
        norm = (rel - lo) / span
        fresh = freshness(row[2], now) if RECENCY_WEIGHT > 0 else 0.0
        scored.append((norm + RECENCY_WEIGHT * fresh, row, norm, fresh))
    scored.sort(key=lambda t: -t[0])

    # One hit per session: the best turn stands for the conversation. Right for
    # "did we discuss X", wrong for "what is the answer to X", because a session
    # whose strongest lexical match is not the turn holding the fact loses the
    # fact entirely. Worth 3.3 points on the measurement above, so callers after
    # a fact pass per_session=False.
    seen, out = set(), []
    for score, (sid, r_role, ts, text, opening, rank), norm, fresh in scored:
        if per_session and sid in seen:
            continue
        seen.add(sid)
        out.append({
            "session": sid[:8], "role": r_role, "rank": round(rank, 2),
            "score": round(score, 3), "fresh": round(fresh, 3),
            "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)),
            "opening": opening, "text": text,
        })
        if len(out) >= limit:
            break
    return out


# --- hybrid fact search ----------------------------------------------------------
# `search()` above answers "which conversation talked about this". `find()` answers
# "which sentence says this", which is what a memory has to do when asked a fact.
#
# Built from what failed on the MemConflict benchmark, measured 2026-09-13/14:
#
#   1. The evidence was in the index and below the cut. The turn carrying the
#      answer sat in bm25's top 5 48% of the time and its top 50 85% of the time.
#      So candidates are gathered wide and cut late.
#   2. Keyword search cannot join words that mean the same thing. "Did the
#      user's residence change?" shares no word with "I just relocated to
#      Melbourne". A second candidate channel ranks by meaning (embed.py).
#   3. OR-ing every word of the question, stopwords included, let long chatty
#      assistant replies outrank the short sentence that states the fact. The
#      query keeps content words only, and IDF-weighted coverage is scored.
#
# Channels are fused by reciprocal rank, not by adding raw scores: bm25 and cosine
# live on different scales, and rank fusion needs no calibration between them.

FIND_STOP = frozenset("""
a an the and or but if then of to in on at by for with from as is are was were be been being
it its this that these those do does did done doing have has had having will would can could
should may might must shall i me my mine we us our ours you your yours he him his she her hers
they them their theirs what which who whom whose when where why how all any both each few more
most other some such no nor not only own same so than too very just about above after again
against below between during before into through under until up down out off over also there
here ever yet still now currently recently lately today use used using get got make made
""".split())

TURN_VEC_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_vec (
  id  INTEGER PRIMARY KEY,
  v   BLOB NOT NULL
);
"""

RRF_K = 60
FIND_POOL = int(os.environ.get("QUIPU_FIND_POOL", "200"))
FIND_PER_SESSION = int(os.environ.get("QUIPU_FIND_PER_SESSION", "2"))
FIND_SEMANTIC = os.environ.get("QUIPU_FIND_SEMANTIC", "1") not in ("0", "false", "no")
FIND_USER_BOOST = float(os.environ.get("QUIPU_FIND_USER_BOOST", "0"))
W_BM25 = float(os.environ.get("QUIPU_FIND_W_BM25", "1.0"))
W_SEM = float(os.environ.get("QUIPU_FIND_W_SEM", "1.0"))
W_COV = float(os.environ.get("QUIPU_FIND_W_COV", "1.0"))


def content_terms(text, cap=16):
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9][a-z0-9_.'-]*", (text or "").lower()):
        w = w.strip(".'-_")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) < 2 or w in FIND_STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= cap:
            break
    return out


def _stem(w):
    for suf in ("ation", "ing", "ies", "ied", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _embedder():
    """The static embedding model, or None when numpy or the weights are absent.
    Search must degrade to keywords, never fail, on a machine without them."""
    try:
        import embed  # noqa: E402
        return embed
    except Exception:
        return None


def embed_new_turns(con=None, batch=4000, limit=None, verbose=False):
    """Give every indexed turn a meaning vector. Incremental and idempotent."""
    E = _embedder()
    if E is None:
        return 0
    own = con is None
    con = con or connect()
    con.executescript(SCHEMA + TURN_VEC_SCHEMA)
    done = 0
    try:
        model = E.model()
    except Exception:
        if own:
            con.close()
        return 0
    while True:
        rows = con.execute(
            "SELECT t.id, t.text FROM turn t LEFT JOIN turn_vec v ON v.id = t.id "
            "WHERE v.id IS NULL LIMIT ?", (batch,)).fetchall()
        if not rows:
            break
        vecs = model.encode([r[1] for r in rows])
        con.executemany("INSERT OR REPLACE INTO turn_vec(id, v) VALUES (?, ?)",
                        [(r[0], E.to_blob(v)) for r, v in zip(rows, vecs)])
        con.commit()
        done += len(rows)
        if verbose:
            print("  embedded %d turns" % done)
        if limit and done >= limit:
            break
    if own:
        con.close()
    return done


_VEC_CACHE = {}


def _turn_matrix(con):
    """(ids, matrix) for every embedded turn, cached until the table changes."""
    E = _embedder()
    if E is None:
        return None, None
    try:
        n, top = con.execute("SELECT count(*), COALESCE(max(id), 0) FROM turn_vec").fetchone()
    except Exception:
        return None, None
    if not n:
        return None, None
    key = (con.execute("PRAGMA database_list").fetchone()[2], n, top)
    hit = _VEC_CACHE.get("turns")
    if hit and hit[0] == key:
        return hit[1], hit[2]
    import numpy as np
    rows = con.execute("SELECT id, v FROM turn_vec ORDER BY id").fetchall()
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).astype(np.float32)
    mat = mat.reshape(len(rows), -1)
    _VEC_CACHE["turns"] = (key, ids, mat)
    return ids, mat


def find(query, limit=6, pool=None, per_session=None, semantic=None, user_boost=None,
         days=None, explain=False):
    """The turns most likely to STATE the answer to `query`.

    Three rankings over one wide candidate pool, fused by reciprocal rank:
    bm25 on content words, cosine meaning similarity, and IDF-weighted coverage
    of the question's content words. `user_boost` lifts the user's own
    statements, which is where a fact about the user is stated; it defaults
    off because in an engineering session the assistant states the findings.
    """
    pool = pool or FIND_POOL
    per_session = FIND_PER_SESSION if per_session is None else per_session
    semantic = FIND_SEMANTIC if semantic is None else semantic
    user_boost = FIND_USER_BOOST if user_boost is None else user_boost

    con = connect()
    con.executescript(SCHEMA + TURN_VEC_SCHEMA)
    terms = content_terms(query)
    ranks = {}                       # turn id -> {channel: rank}
    since = time.time() - days * 86400 if days else None

    if terms:
        match = " OR ".join('"%s"' % t.replace('"', '""') for t in terms)
        sql = ("SELECT t.id FROM turn_fts JOIN turn t ON t.id = turn_fts.rowid "
               "WHERE turn_fts MATCH ?" + (" AND t.ts > ?" if since else "") +
               " ORDER BY bm25(turn_fts) LIMIT ?")
        args = [match] + ([since] if since else []) + [pool]
        try:
            for rank, (tid,) in enumerate(con.execute(sql, args).fetchall()):
                ranks.setdefault(tid, {})["bm25"] = rank
        except Exception:
            pass

    sims = {}
    if semantic:
        ids, mat = _turn_matrix(con)
        if ids is not None and len(ids):
            try:
                import numpy as np
                qv = _embedder().model().encode(query)
                if float(np.linalg.norm(qv)) > 0:
                    scores = mat @ qv
                    k = min(pool, len(scores))
                    top = np.argpartition(-scores, k - 1)[:k]
                    top = top[np.argsort(-scores[top])]
                    for rank, idx in enumerate(top):
                        tid = int(ids[idx])
                        ranks.setdefault(tid, {})["sem"] = rank
                        sims[tid] = float(scores[idx])
            except Exception:
                pass

    if not ranks:
        con.close()
        return []

    cand = list(ranks)
    rows = {}
    for i in range(0, len(cand), 800):
        chunk = cand[i:i + 800]
        for r in con.execute(
                "SELECT t.id, t.sid, t.seq, t.role, t.ts, t.text, COALESCE(s.opening, '') "
                "FROM turn t LEFT JOIN session s ON s.id = t.sid WHERE t.id IN (%s)"
                % ",".join("?" * len(chunk)), chunk):
            rows[r[0]] = r
    if since:
        rows = {k: v for k, v in rows.items() if v[4] > since}

    # IDF of each content word, then how much of the question each turn covers.
    idf = {}
    if terms:
        total = con.execute("SELECT count(*) FROM turn").fetchone()[0] or 1
        for t in terms:
            try:
                df = con.execute("SELECT count(*) FROM turn_fts WHERE turn_fts MATCH ?",
                                 ('"%s"' % t.replace('"', '""'),)).fetchone()[0]
            except Exception:
                df = 0
            idf[t] = math.log((total + 1.0) / (df + 0.5))
    con.close()

    stems = {t: _stem(t) for t in terms}
    mass = sum(idf.values()) or 1.0
    cover = {}
    for tid, r in rows.items():
        low = r[5].lower()
        words = set(re.findall(r"[a-z0-9][a-z0-9'-]*", low))
        wstems = {_stem(w) for w in words}
        got = sum(idf[t] for t in terms if t in words or stems[t] in wstems)
        cover[tid] = got / mass
    for rank, tid in enumerate(sorted(cover, key=lambda x: -cover[x])):
        if cover[tid] > 0:
            ranks[tid]["cov"] = rank

    scored = []
    for tid, r in rows.items():
        rk = ranks[tid]
        s = (W_BM25 / (RRF_K + rk["bm25"]) if "bm25" in rk else 0.0) \
            + (W_SEM / (RRF_K + rk["sem"]) if "sem" in rk else 0.0) \
            + (W_COV / (RRF_K + rk["cov"]) if "cov" in rk else 0.0)
        if user_boost and r[3] == "user":
            s *= 1.0 + user_boost
        scored.append((s, tid))
    scored.sort(key=lambda x: -x[0])

    out, per = [], {}
    for s, tid in scored:
        sid, seq, role, ts, text, opening = rows[tid][1:]
        if per_session and per.get(sid, 0) >= per_session:
            continue
        per[sid] = per.get(sid, 0) + 1
        item = {
            "session": str(sid)[:8], "sid": sid, "seq": seq, "role": role,
            "score": round(s * 1000.0, 3), "rank": -round(s * 1000.0, 3),
            "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)), "ts": ts,
            "opening": opening, "text": text,
        }
        if explain:
            item["why"] = {"ranks": ranks[tid], "cover": round(cover.get(tid, 0.0), 3),
                           "cosine": round(sims.get(tid, 0.0), 3)}
        out.append(item)
        if len(out) >= limit:
            break
    return out


def excerpt(text, query, width=220):
    words = [w for w in re.findall(r"[A-Za-z0-9_.-]{3,}", query.lower())]
    low = text.lower()
    pos = min([low.find(w) for w in words if low.find(w) != -1] or [0])
    start = max(0, pos - width // 3)
    cut = text[start:start + width]
    return ("..." if start else "") + cut.strip() + ("..." if start + width < len(text) else "")


def main():
    ap = argparse.ArgumentParser(prog="sessions.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("index")
    p.add_argument("--all", action="store_true")
    p.add_argument("--since", type=int, help="only sessions touched in the last N days")

    p = sub.add_parser("search")
    p.add_argument("text", nargs="+")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--role", choices=["user", "assistant"])
    p.add_argument("--days", type=int)
    p.add_argument("--json", action="store_true")
    p.add_argument("--turns", action="store_true",
                   help="best turns overall, not one per session: for finding a "
                        "fact rather than finding a conversation")
    p.add_argument("--pool", type=int, default=0,
                   help="candidates to rank before cutting (default limit*12)")

    p = sub.add_parser("find")
    p.add_argument("text", nargs="+")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--days", type=int)
    p.add_argument("--json", action="store_true")
    p.add_argument("--explain", action="store_true")

    p = sub.add_parser("embed")
    p.add_argument("--limit", type=int, default=0, help="stop after N turns (0 = all)")

    p = sub.add_parser("show")
    p.add_argument("session")
    p.add_argument("--grep")

    sub.add_parser("stats")
    args = ap.parse_args()

    if args.cmd == "index":
        index(all_of_them=args.all, since_days=args.since)
        return 0

    if args.cmd == "search":
        query = " ".join(args.text)
        t0 = time.time()
        hits = search(query, limit=args.limit, role=args.role, days=args.days,
                      per_session=not args.turns, pool=args.pool or None)
        if args.json:
            print(json.dumps(hits, indent=2))
        else:
            if not hits:
                print("nothing in the transcripts matches that")
            for h in hits:
                print("%s  %s  session %s" % (h["when"], h["role"].upper().ljust(9), h["session"]))
                print("   %s" % excerpt(h["text"], query))
                print()
        print("[%d session(s) in %dms]" % (len(hits), (time.time() - t0) * 1000),
              file=sys.stderr)
        return 0

    if args.cmd == "find":
        hits = find(" ".join(args.text), limit=args.limit, days=args.days, explain=args.explain)
        if args.json:
            print(json.dumps(hits, indent=1))
            return 0
        for h in hits:
            print("%s  %-9s session %s  %s" % (h["when"], h["role"].upper(), h["session"],
                                             h.get("why", "") and json.dumps(h["why"])))
            print("   " + h["text"][:240].replace("\n", " "))
            print()
        return 0

    if args.cmd == "embed":
        t0 = time.time()
        n = embed_new_turns(limit=args.limit or None, verbose=False)
        print("embedded %d turn(s) in %.1fs" % (n, time.time() - t0))
        return 0

    if args.cmd == "show":
        con = connect()
        con.executescript(SCHEMA)
        rows = con.execute(
            "SELECT role, ts, text FROM turn WHERE sid LIKE ? ORDER BY seq",
            (args.session + "%",)).fetchall()
        con.close()
        for role, ts, text in rows:
            if args.grep and args.grep.lower() not in text.lower():
                continue
            print("[%s] %s: %s" % (time.strftime("%H:%M", time.localtime(ts)),
                                   role.upper(), text[:400]))
        return 0

    if args.cmd == "stats":
        con = connect()
        con.executescript(SCHEMA)
        s = con.execute("SELECT count(*), sum(turns), sum(bytes) FROM session").fetchone()
        t = con.execute("SELECT count(*), sum(length(text)) FROM turn").fetchone()
        span = con.execute("SELECT min(started), max(ended) FROM session").fetchone()
        con.close()
        print("sessions:  %s" % format(s[0] or 0, ","))
        print("turns:     %s" % format(t[0] or 0, ","))
        print("raw:       %.1f GB of transcript" % ((s[2] or 0) / 1e9))
        print("indexed:   %.1f MB of actual conversation (%.1f%% of raw)"
              % ((t[1] or 0) / 1e6, 100.0 * (t[1] or 1) / max(s[2] or 1, 1)))
        if span and span[0]:
            print("covering:  %s to %s" % (
                time.strftime("%Y-%m-%d", time.localtime(span[0])),
                time.strftime("%Y-%m-%d", time.localtime(span[1]))))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

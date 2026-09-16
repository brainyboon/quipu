#!/usr/bin/env python3
"""
recall.py - the index and the runtime retrieval path.

Two jobs. `index` walks the memory store and rebuilds the searchable chunks,
folding in whatever `enrich.py` has derived for each file. `search` answers a
query in well under 100ms with no model call, because it runs inside the hook
that blocks the user's prompt.

Ranking is evidence x type x freshness:
  evidence   IDF-weighted term overlap, 1.6x for a hit in title/description/aids
  type       a rule outranks a status report when both match
  freshness  per-type half-life decay, so a 2026-05 project note loses to a
             2026-08 one while a hard rule never decays at all

Usage:
  recall.py index [--force]
  recall.py query "text" [--context "recent turns"] [--limit N] [--json]
  recall.py explain "text"
  recall.py stats
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memlib import (  # noqa: E402
    HALF_LIFE_DAYS, TYPE_WEIGHT, MemoryLock,
    connect, doc_kind, get_meta, read_doc, set_meta, split_sections, walk_memory,
)

# Conversational filler only. Words like "new", "add", "check" and "fix" were
# in here and were silently eating the bridge: enrichment gives a webhook note
# the trigger "when adding a NEW webhook", and the query threw both bridge words
# away before searching.
STOPWORDS = set("""
the and for are but not you all can her was one our out day get has him his how
its may now old see two way who did yes let put say she too use that this
with have from they will would there their what which when like time just
know take into your some them then than look only come over also back after
want because these give most does should could need please thing things
something anything really still even much many more very well why where here
been being were again already though while before tell
thanks okay sure right wrong good bad next
same other another every each both between against under above below
once doing done goes went continue proceed ahead hey hello cool great perfect
nice yeah yep nope sorry actually maybe think seems looks keep going wait hold
guess sounds alright fine true false correct exactly gonna wanna mind care
about above across along around behind beyond during except inside near onto
since toward within without anyway besides however instead meanwhile therefore
lets thats theres whats dont doesnt cant wont isnt arent wasnt werent havent
really quite rather pretty bunch stuff kind sort lots plenty everything nothing
""".split())

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,}")

# Three tiers, because the fields are not equally trustworthy. A memory's own
# title and description are what it is about. The model-derived aliases are a
# bridge to the user's vocabulary, and they are generous on purpose, so a hit there
# counts for less. The body is the raw material.
HEAD_BOOST = 1.9           # title + description: the memory's own words
AIDS_BOOST = 1.75          # model-derived aliases and triggers
CONTEXT_WEIGHT = 0.40      # a term from an earlier turn, not from this prompt
DECAY_FLOOR = 0.45         # an old memory dims, it never disappears
SUPERSEDED_PENALTY = 0.25
STALE_PENALTY = 0.70
DEFAULT_FLOOR = 5.0

# Score the few BEST matching terms, not every match. Summing all matches
# rewards breadth: a memory that happens to contain eight ordinary words beats
# one that nails the two rare words that are the actual subject. It also means
# a diffuse prompt with no rare terms scores low everywhere, which is how the
# hook learns to stay quiet instead of guessing.
TOP_TERMS = 3
TAIL_WEIGHT = 0.15

# bm25 was computed only to pick the 60-doc candidate set and then thrown away,
# so scoring had no term-frequency signal: a memory mentioning the subject once
# ranked identically to one built around it. Letting bm25 carry a small weight
# restores that signal.
#
# MEASURED NEUTRAL. On a 16-prompt battery of real prompts this scores 14/16 at
# 0.00, 0.06, 0.12 and 0.25 alike, so it is kept small on the theory rather than
# the evidence. Do not raise it without a bigger probe set; the battery cannot
# resolve it.
BM25_WEIGHT = 0.12


# ---------------------------------------------------------------- index


def store_signature():
    n = newest = size = 0
    for p, _rel, mt in walk_memory():
        n += 1
        newest = max(newest, mt)
        try:
            size += os.path.getsize(p)
        except OSError:
            pass
    return "%d:%.0f:%d" % (n, newest, size)


def build_index(force=False, verbose=False):
    con = connect()
    want = store_signature() + "|" + (get_meta(con, "enrich_sig") or "")
    if not force and get_meta(con, "index_sig") == want:
        con.close()
        return 0

    with MemoryLock():
        con.execute("DELETE FROM chunk")
        n_docs = n_chunks = 0
        live_rels = []
        for path, rel, mtime in walk_memory():
            try:
                meta, body, sha, raw = read_doc(path)
            except OSError:
                continue
            name = meta.get("name") or os.path.basename(rel)[:-3]
            descr = meta.get("description", "")
            kind = doc_kind(meta, rel)
            live_rels.append(rel)
            con.execute(
                "INSERT INTO doc(rel, path, name, descr, kind, hash, mtime, bytes) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(rel) DO UPDATE SET "
                "  path=excluded.path, name=excluded.name, descr=excluded.descr, "
                "  kind=excluded.kind, hash=excluded.hash, mtime=excluded.mtime, "
                "  bytes=excluded.bytes",
                (rel, path, name, descr, kind, sha, mtime, len(raw)),
            )
            doc_id = con.execute("SELECT id FROM doc WHERE rel=?", (rel,)).fetchone()[0]
            n_docs += 1

            row = con.execute(
                "SELECT topics, aliases, triggers, entities, summary "
                "FROM enrich WHERE hash=?", (sha,)
            ).fetchone()
            aids = " ".join(x for x in row if x).strip() if row else ""

            for head, chunk in split_sections(body):
                title = name if not head else "%s > %s" % (name, head)
                con.execute(
                    "INSERT INTO chunk(doc_id, rel, title, descr, kind, body, aids, mtime) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (doc_id, rel, title, descr, kind, chunk.strip(), aids, mtime),
                )
                n_chunks += 1

        # A memory file the user deleted should stop being recalled.
        if live_rels:
            marks = ",".join("?" * len(live_rels))
            con.execute("DELETE FROM doc WHERE rel NOT IN (%s)" % marks, live_rels)

        set_meta(con, "index_sig", want)
        set_meta(con, "index_built", time.time())
        con.commit()
    con.close()
    try:
        embed_docs(fetch=False)
    except Exception:
        pass                        # meaning search is an addition, never a dependency
    if verbose:
        print("indexed %d chunks across %d memories" % (n_chunks, n_docs))
    return n_chunks


# ---------------------------------------------------------------- query


def tokenize(text, cap=18):
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    out, seen = [], set()
    for m in TOKEN_RE.finditer(text):
        raw = m.group(0).strip(".-_")
        t = raw.lower()
        # Two-letter tokens are noise unless they were written as an acronym.
        # "why is CI so slow" has exactly one searchable word in it.
        min_len = 2 if raw.isupper() and len(raw) == 2 else 3
        if len(t) < min_len or t in STOPWORDS or t in seen or t.isdigit():
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= cap:
            break
    return out


def fts_escape(tok):
    return '"' + tok.replace('"', '""') + '"'


SUFFIXES = ("ing", "ies", "ed", "es", "s")


def stem(word):
    """Cheap suffix strip so `deploying` and `deploys` land on `deploy`.

    Matching used to be a plain substring test, which is why "api" matched
    "rapid" and "prod" matched "produce": the tail of every result list filled
    up with memories that shared no actual subject with the prompt.
    """
    for suf in SUFFIXES:
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def word_set(text):
    out = set()
    for m in TOKEN_RE.finditer(text.lower()):
        w = m.group(0).strip(".-_")
        if len(w) < 3:
            continue
        out.add(w)
        out.add(stem(w))
        # `api.example.com` should also answer to `api` and `example`.
        if "." in w or "-" in w or "_" in w:
            for part in re.split(r"[.\-_]+", w):
                # 2 chars, not 3: `reference-ci-cache` has to
                # answer to "ci", and `api.example.com` to "api".
                if len(part) >= 2:
                    out.add(part)
                    out.add(stem(part))
    return out


def freshness(kind, mtime, now):
    half = HALF_LIFE_DAYS.get(kind)
    if not half:
        return 1.0
    age_days = max(0.0, (now - mtime) / 86400.0)
    return max(DECAY_FLOOR, 0.5 ** (age_days / half))


def search(prompt, context="", limit=4, floor=DEFAULT_FLOOR, explain=False):
    """Rank memories against a prompt, widened by recent conversation.

    `context` carries terms from earlier turns at reduced weight. Real prompts
    are conversational ("still broken, try again"); the topic usually lives in
    the turn before, not in the sentence just typed.
    """
    con = connect()
    total = con.execute("SELECT count(*) FROM chunk").fetchone()[0]
    if not total:
        con.close()
        return []

    prompt_toks = tokenize(prompt)
    ctx_toks = [t for t in tokenize(context, cap=22) if t not in prompt_toks]
    if not prompt_toks and not ctx_toks:
        con.close()
        return []

    weights, idf, usable, informative = {}, {}, [], []
    prompt_informative, prompt_terms = set(), set()
    for tok, w in ([(t, 1.0) for t in prompt_toks]
                   + [(t, CONTEXT_WEIGHT) for t in ctx_toks]):
        try:
            # Column-filtered: enrichment added 6-12 aliases per memory and they
            # overlap heavily ("api", "production", "live"), so counting them
            # here would push every useful term into the noise band. `api` went
            # from a real signal to 58% of the corpus that way.
            df = con.execute(
                "SELECT count(*) FROM fts WHERE fts MATCH ?",
                ("{title descr body} : " + fts_escape(tok),),
            ).fetchone()[0]
            if df == 0:
                df = con.execute(
                    "SELECT count(*) FROM fts WHERE fts MATCH ?", (fts_escape(tok),)
                ).fetchone()[0]
        except sqlite3.OperationalError:
            continue
        if df == 0:
            continue
        usable.append(tok)
        weights[tok] = w
        idf[tok] = math.log(total / float(df))
        if w == 1.0:
            prompt_terms.add(tok)
        # Only a term in more than half the store is pure noise. A project name can
        # sit at 45% and still be the entire point of "update <project>".
        if df <= max(2, total * 0.55):
            informative.append(tok)
            if w == 1.0:
                prompt_informative.add(tok)

    # Context sharpens a query that already has a subject; it must never invent
    # one. Without this, a session ABOUT memory made every prompt in it recall
    # every memory-related note, including "sounds good, carry on".
    if not prompt_informative:
        con.close()
        return []

    # Retrieve on the prompt's terms only. Context is for scoring, not for
    # candidate selection: a 60-row candidate set chosen by bm25 over prompt
    # AND conversation is dominated by whatever the session has been talking
    # about, so the documents the question actually names never make the list.
    query = " OR ".join(fts_escape(t) for t in prompt_terms)
    try:
        rows = con.execute(
            "SELECT c.rel, c.title, c.descr, c.kind, c.body, c.aids, c.mtime, "
            "       d.status, d.superseded_by, d.stale_note, "
            "       bm25(fts, 8.0, 6.0, 5.0, 1.0) AS bm "
            "FROM fts JOIN chunk c ON c.id = fts.rowid "
            "JOIN doc d ON d.id = c.doc_id "
            "WHERE fts MATCH ? ORDER BY bm LIMIT 60",
            (query,),
        ).fetchall()
    except sqlite3.OperationalError:
        con.close()
        return []
    con.close()

    now = time.time()
    inf = set(informative)
    results, seen_files = [], set()
    for (rel, title, descr, kind, body, aids, mtime,
         status, superseded_by, stale_note, bm25_score) in rows:
        head_words = word_set(title + " " + descr)
        aid_words = word_set(aids)
        body_words = word_set(body)
        all_words = head_words | aid_words | body_words

        def matched(tok):
            return tok in all_words or stem(tok) in all_words

        hits = {t for t in usable if matched(t)}
        # The reason a memory surfaces has to appear in what the user just typed.
        if not hits or not (hits & prompt_informative):
            continue
        head_hits = {t for t in hits
                     if t in head_words or stem(t) in head_words}
        aid_hits = {t for t in hits - head_hits
                    if t in aid_words or stem(t) in aid_words}
        # A single rare word buried in a body is a coincidence, not a match.
        # Demand it reach the title/description/aids, or that three terms agree.
        if not head_hits and len(hits) < 3:
            continue

        def field_boost(tok):
            if tok in head_hits:
                return HEAD_BOOST
            if tok in aid_hits:
                return AIDS_BOOST
            return 1.0

        # The top slots belong to what the user typed. A term that only appears in
        # earlier turns can add to the tail but can never lead, or a session
        # that happens to be ABOUT indexes and memory ranks every index file
        # first no matter what the current question is.
        lead = sorted((idf[t] * field_boost(t) for t in hits & prompt_terms),
                      reverse=True)
        tail = sorted((idf[t] * weights[t] * field_boost(t)
                       for t in hits - prompt_terms), reverse=True)
        evidence = (sum(lead[:TOP_TERMS])
                    + TAIL_WEIGHT * (sum(lead[TOP_TERMS:]) + sum(tail)))
        fresh = freshness(kind, mtime, now)
        # bm25() is negative, more-negative is better; flip and damp it.
        density = min(3.0, max(0.0, -bm25_score))
        score = (evidence + BM25_WEIGHT * density) * TYPE_WEIGHT.get(kind, 1.0) * fresh
        if superseded_by or status != "live":
            score *= SUPERSEDED_PENALTY
        if stale_note:
            score *= STALE_PENALTY
        if score < floor or rel in seen_files:
            continue
        seen_files.add(rel)

        item = {
            "rel": rel, "title": title, "descr": descr, "kind": kind, "body": body,
            "score": round(score, 2), "evidence": sorted(hits),
            # The subject the user actually named. The hook uses this, not the top
            # hit's evidence, to decide a topic has already been served: two
            # phrasings of one question rank different documents first, so
            # comparing hit evidence lets the second phrasing through.
            "query_terms": sorted(prompt_informative),
            "superseded_by": superseded_by, "stale_note": stale_note,
        }
        if explain:
            item["why"] = {
                "evidence": round(evidence, 2),
                "aid_hits": sorted(aid_hits),
                "type_weight": TYPE_WEIGHT.get(kind, 1.0),
                "freshness": round(fresh, 3),
                "age_days": round((now - mtime) / 86400.0, 1),
                "head_hits": sorted(head_hits),
                "bm25": round(bm25_score, 2),
            }
        results.append(item)
        if len(results) >= limit * 4:
            break

    results.sort(key=lambda r: -r["score"])
    return results[:limit]


# --- meaning search over memories ---------------------------------------------
# Keyword recall stays the hot path: it is what the prompt hook runs, it is gated
# to stay silent on chit-chat, and it scores 28 of 30 on a battery of real
# prompts that is not published here.
# search_hybrid() is for callers that ask a question and want an answer rather
# than silence: `mem ask`, fact lookups, and the benchmark. It adds a second
# channel ranked by meaning, so a note that says "relocated" answers a question
# about "residence", and fuses the two by reciprocal rank.

DOC_VEC_SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_vec (
  rel  TEXT PRIMARY KEY,
  sig  TEXT NOT NULL,
  v    BLOB NOT NULL
);
"""
_DOC_CACHE = {}


def _embed_mod():
    try:
        import embed  # noqa: E402
        return embed
    except Exception:
        return None


def embed_docs(verbose=False, fetch=True):
    """One meaning vector per memory: its name, description, enrichment and the
    opening of its body. Re-embedded only when the file or its enrichment moved."""
    E = _embed_mod()
    if E is None:
        return 0
    # Never download the model inside a prompt hook: the hook is killed at 8
    # seconds. The background embed job fetches it; until then, keywords only.
    if not fetch and not E.available():
        return 0
    con = connect()
    con.executescript(DOC_VEC_SCHEMA)
    have = dict(con.execute("SELECT rel, sig FROM doc_vec"))
    rows = con.execute(
        "SELECT d.rel, d.name, d.descr, d.hash, "
        "       COALESCE(e.summary,''), COALESCE(e.aliases,''), COALESCE(e.triggers,''), "
        "       COALESCE((SELECT substr(c.body, 1, 700) FROM chunk c WHERE c.rel = d.rel "
        "                 ORDER BY c.id LIMIT 1), '') "
        "FROM doc d LEFT JOIN enrich e ON e.hash = d.hash").fetchall()
    todo, live = [], set()
    for rel, name, descr, sha, summary, aliases, triggers, body in rows:
        live.add(rel)
        sig = "%s|%d|%d" % (sha, len(summary), len(aliases))
        if have.get(rel) == sig:
            continue
        text = "%s. %s %s %s %s %s" % (name.replace("-", " ").replace("_", " "), descr,
                                       summary, aliases, triggers, body)
        todo.append((rel, sig, text))
    if todo:
        model = E.model()
        vecs = model.encode([t[2] for t in todo])
        con.executemany("INSERT OR REPLACE INTO doc_vec(rel, sig, v) VALUES (?,?,?)",
                        [(t[0], t[1], E.to_blob(v)) for t, v in zip(todo, vecs)])
    gone = [r for r in have if r not in live]
    if gone:
        con.executemany("DELETE FROM doc_vec WHERE rel = ?", [(r,) for r in gone])
    con.commit()
    con.close()
    if verbose:
        print("embedded %d memories, dropped %d" % (len(todo), len(gone)))
    return len(todo)


def _doc_matrix(con):
    try:
        n = con.execute("SELECT count(*) FROM doc_vec").fetchone()[0]
    except sqlite3.OperationalError:
        return None, None
    if not n:
        return None, None
    import numpy as np
    key = (con.execute("PRAGMA database_list").fetchone()[2], n,
           con.execute("SELECT group_concat(sig, ',') FROM (SELECT sig FROM doc_vec ORDER BY rel)")
              .fetchone()[0].__hash__())
    hit = _DOC_CACHE.get("docs")
    if hit and hit[0] == key:
        return hit[1], hit[2]
    rows = con.execute("SELECT rel, v FROM doc_vec ORDER BY rel").fetchall()
    rels = [r[0] for r in rows]
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).astype(np.float32)
    mat = mat.reshape(len(rows), -1)
    _DOC_CACHE["docs"] = (key, rels, mat)
    return rels, mat


def search_hybrid(prompt, context="", limit=4, pool=40, semantic=True, explain=False):
    """Keyword recall and meaning recall over memories, fused by reciprocal rank."""
    lex = search(prompt, context=context, limit=pool, floor=0.0, explain=explain)
    ranks, items = {}, {}
    for i, it in enumerate(lex):
        ranks.setdefault(it["rel"], {})["lex"] = i
        items[it["rel"]] = it

    E = _embed_mod() if semantic else None
    if E is not None:
        con = connect()
        try:
            con.executescript(DOC_VEC_SCHEMA)
            rels, mat = _doc_matrix(con)
            if rels:
                import numpy as np
                qv = E.model().encode(prompt)
                if float(np.linalg.norm(qv)) > 0:
                    sims = mat @ qv
                    k = min(pool, len(rels))
                    top = np.argpartition(-sims, k - 1)[:k]
                    top = top[np.argsort(-sims[top])]
                    missing = []
                    for rank, idx in enumerate(top):
                        rel = rels[int(idx)]
                        ranks.setdefault(rel, {})["sem"] = rank
                        ranks[rel]["cos"] = float(sims[idx])
                        if rel not in items:
                            missing.append(rel)
                    for rel in missing:
                        row = con.execute(
                            "SELECT c.title, c.descr, c.kind, c.body, c.mtime, d.status, "
                            "       d.superseded_by, d.stale_note "
                            "FROM chunk c JOIN doc d ON d.id = c.doc_id WHERE c.rel = ? "
                            "ORDER BY c.id LIMIT 1", (rel,)).fetchone()
                        if not row:
                            continue
                        title, descr, kind, body, mtime, status, sup, stale = row
                        items[rel] = {"rel": rel, "title": title, "descr": descr, "kind": kind,
                                      "body": body, "mtime": mtime, "score": 0.0, "evidence": [],
                                      "query_terms": [], "superseded_by": sup, "stale_note": stale,
                                      "status": status}
        finally:
            con.close()

    fused = []
    for rel, rk in ranks.items():
        if rel not in items:
            continue
        sc = (1.0 / (60 + rk["lex"]) if "lex" in rk else 0.0) + \
             (1.0 / (60 + rk["sem"]) if "sem" in rk else 0.0)
        it = items[rel]
        if it.get("superseded_by") or (it.get("status") not in (None, "live")):
            sc *= SUPERSEDED_PENALTY
        if it.get("stale_note"):
            sc *= STALE_PENALTY
        out = dict(it)
        out["score"] = round(sc * 1000.0, 3)
        if explain:
            out["fused"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rk.items()}
        fused.append(out)
    fused.sort(key=lambda r: -r["score"])
    return fused[:limit]


def first_sentence(text, cap=230):
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"^\*\*Why:\*\*\s*", "", text)
    if len(text) <= cap:
        return text
    cut = text[:cap]
    dot = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: dot + 1] if dot > cap * 0.5 else cut.rstrip()) + "..."


def snippet_for(hit):
    return first_sentence(hit["descr"].strip() or hit["body"])


def render(results):
    if not results:
        return ""
    lines = ["<recalled-memory>"]
    for r in results:
        note = ""
        if r.get("superseded_by"):
            note = " [SUPERSEDED by %s]" % r["superseded_by"]
        elif r.get("stale_note"):
            flag = r["stale_note"]
            # A claim that no longer checks out is a different
            # warning from a dead file path. Say which.
            if flag.startswith("unverified:"):
                note = " [UNVERIFIED: %s]" % flag.split(":", 1)[1].strip()
            else:
                note = " [STALE: %s]" % flag
        lines.append("- **%s** (`%s`)%s: %s" % (r["title"], r["rel"], note, snippet_for(r)))
    lines.append("</recalled-memory>")
    return "\n".join(lines)


# ---------------------------------------------------------------- cli


# --- "do we already know this?" ------------------------------------------------
# Two writers create new memories: session capture and profile demotion. Both had
# the same hole. They compared a proposed description against existing ones with
# Jaccard overlap, which collapses when the two are phrased differently or differ
# in length, so a rule already in the store got written down again in weaker
# words. Profile demotion did it 39 times, because the model rephrases the line
# slightly on every rebuild and the filename is derived from the wording.
#
# Recall is the honest oracle for this question. "Is this already in the store"
# is exactly "would retrieval find it", and retrieval already knows the aliases
# and triggers that plain word overlap does not.
#
# COVERED_SCORE was measured, not guessed, over all 587 descriptions on
# 2026-09-13: searching a memory's own description and ignoring itself, the 39
# known duplicates had a median best-other score of 32.9 and everything else 17.5.
# At 30 it catches 22 of the 39, and 14 of the 16 other memories it flags are the
# other half of a real duplicate pair. A caller must never silently drop what
# this flags: file it for review, so a false positive costs a queue item and not
# a fact.
COVERED_SCORE = 30.0


def covered_by(text, exclude_rel=None, threshold=COVERED_SCORE):
    """The memory that already carries this, or None. Never raises."""
    try:
        for hit in search(text, limit=4):
            if exclude_rel and hit.get("rel") == exclude_rel:
                continue
            if hit.get("score", 0) >= threshold:
                return hit["rel"], round(hit["score"], 1)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(prog="recall.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("index")
    p.add_argument("--force", action="store_true")

    for name in ("query", "explain"):
        p = sub.add_parser(name)
        p.add_argument("text", nargs="+")
        p.add_argument("--context", default="")
        p.add_argument("--limit", type=int, default=4)
        p.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
        p.add_argument("--json", action="store_true")

    sub.add_parser("stats")
    sub.add_parser("embed")
    args = ap.parse_args()

    if args.cmd == "index":
        n = build_index(force=args.force, verbose=True)
        if not n:
            print("index already current")
        return 0

    if args.cmd == "embed":
        t0 = time.time()
        n = embed_docs()
        print("embedded %d memor%s in %.1fs" % (n, "y" if n == 1 else "ies", time.time() - t0))
        return 0

    if args.cmd == "stats":
        build_index()
        con = connect()
        one = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
        print("memories:   %d" % one("SELECT count(*) FROM doc"))
        print("chunks:     %d" % one("SELECT count(*) FROM chunk"))
        print("enriched:   %d of %d" % (
            one("SELECT count(*) FROM doc d JOIN enrich e ON e.hash=d.hash"),
            one("SELECT count(*) FROM doc")))
        print("superseded: %d" % one("SELECT count(*) FROM doc WHERE superseded_by<>''"))
        print("stale-flag: %d" % one("SELECT count(*) FROM doc WHERE stale_note<>''"))
        print("review open:%d" % one("SELECT count(*) FROM review WHERE status='open'"))
        print("by type:    %s" % ", ".join(
            "%s=%d" % r for r in con.execute(
                "SELECT kind, count(*) FROM doc GROUP BY kind ORDER BY 2 DESC")))
        # Does a memory ever get used? 218 of 568 have never been injected, and
        # the split by who wrote them is the interesting part: memories written
        # by hand during real work are retrieved about twice as often as ones
        # the session-end capture proposed on its own. Same shape as the
        # SkillsBench result on agent-authored skills. Keep it visible, because
        # the cost of a memory nobody retrieves is paid by every other memory
        # competing for the four slots a prompt gets.
        total = one("SELECT count(*) FROM doc")
        used = one("SELECT count(*) FROM doc WHERE retrieved_n>0")
        print("retrieved:  %d of %d ever injected (%.0f%%)"
              % (used, total, 100.0 * used / max(1, total)))
        built = get_meta(con, "index_built")
        if built:
            print("built:      %s" % time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(float(built))))
        con.close()
        return 0

    if args.cmd in ("query", "explain"):
        build_index()
        t0 = time.time()
        res = search(" ".join(args.text), context=args.context, limit=args.limit,
                     floor=args.floor, explain=(args.cmd == "explain"))
        if args.json:
            print(json.dumps(res, indent=2))
        elif args.cmd == "explain":
            for r in res:
                print("%-6s %-44s %s" % (r["score"], r["title"][:44], r["rel"]))
                print("       %s" % json.dumps(r["why"]))
            if not res:
                print("(no match)")
        else:
            print(render(res) or "(no match)")
        print("\n[%d hit(s) in %dms]" % (len(res), (time.time() - t0) * 1000),
              file=sys.stderr)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

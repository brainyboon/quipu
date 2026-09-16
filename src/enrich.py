#!/usr/bin/env python3
"""
enrich.py - the offline model pass that makes the index speak the user's language.

The v1 failure was a query-vocabulary mismatch, not a bad index. A note about
deploy safety never matched "am I about to break production", because nothing
in the file contains those words. Fixing that at query time needs a model in
the prompt path, which costs seconds. So we fix it from the index side instead:
a model reads each memory once, offline, and writes down the words someone
would actually use to reach it.

Per memory it derives:
  topics    canonical subject tags
  aliases   the other words for the same thing
  triggers  "recall this when ..." phrasings
  entities  repos, services, files, commands the memory names
  summary   one retrieval-friendly line

Keyed by content hash, so re-running is free for unchanged files and a memory
that gets edited is automatically re-derived.

Runs through `claude -p`: plan usage on a Claude subscription, metered on an API key.

Usage:
  enrich.py run [--limit N] [--batch N] [--workers N] [--model haiku] [--force]
  enrich.py status
  enrich.py show <rel>
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memlib import (  # noqa: E402
    LLMError, MemoryLock, claude_json, connect, doc_kind, log_line,
    read_doc, set_meta, walk_memory,
)

SYSTEM = (
    "You index a personal engineering memory store so a retrieval system can "
    "find the right note from a natural, messy prompt. You never invent facts. "
    "You output JSON only."
)

PROMPT_HEAD = """Below are %d memory notes from an engineering memory store.

For EACH note, derive retrieval aids. The goal: when a user types a messy
real-world sentence, the right note should surface. So write the words a person
would actually use, not the words already in the note.

For each note return an object with:
  "i"        the note's index number, exactly as given
  "topics"   3-6 lowercase canonical subject tags (e.g. "postgres", "deployment", "production-safety")
  "aliases"  6-12 alternative words and short phrases someone might type to mean
             this note's subject, INCLUDING everyday phrasings and synonyms that
             do NOT appear in the note text.
  "triggers" 2-4 short phrases of the form "when <situation>" describing the
             moment this note should be recalled
  "entities" the concrete named things the note references: repos, services,
             file paths, commands, env vars, people, products. Verbatim.
  "summary"  ONE sentence, under 200 characters, stating the note's operative
             fact in plain language

Rules:
- Only use information present in the note. Never guess a fact.
- aliases carry both the note's own terms and new vocabulary; topics and entities stay literal.
- Do not filter aliases for quality. More is better: filtering measures as
  precision-only and costs recall (R@1000 0.935 filtered vs 0.950 unfiltered).
- If a note is a hard rule or a preference, make one trigger describe the moment
  someone would be about to break it.
- Output a JSON array, one object per note, nothing else.

NOTES:
"""


def note_block(i, rel, name, descr, kind, body, cap=1400):
    body = body.strip()
    if len(body) > cap:
        body = body[:cap] + " ..."
    return (
        "\n--- NOTE %d ---\nfile: %s\nname: %s\ntype: %s\ndescription: %s\nbody:\n%s\n"
        % (i, rel, name, kind, descr or "(none)", body)
    )


def pending(con, force=False):
    """Memories whose current content hash has no enrichment row."""
    out = []
    for path, rel, _mtime in walk_memory():
        try:
            meta, body, sha, _raw = read_doc(path)
        except OSError:
            continue
        # Fact notes are not enriched. They are short, their name IS the attribute
        # ("user residence"), and meaning search already bridges paraphrase. More
        # to the point, a fact's content changes whenever its value does, so
        # hash-keyed enrichment re-ran a model call on every fact after every
        # conversation: on the benchmark, ingest grew from 10 to 110 seconds per
        # session by the thirty-sixth, and kept growing.
        if os.path.basename(rel).startswith("fact_"):
            continue
        if not force:
            hit = con.execute("SELECT 1 FROM enrich WHERE hash=?", (sha,)).fetchone()
            if hit:
                continue
        out.append({
            "rel": rel,
            "hash": sha,
            "name": meta.get("name") or os.path.basename(rel)[:-3],
            "descr": meta.get("description", ""),
            "kind": doc_kind(meta, rel),
            "body": body,
        })
    return out


def enrich_batch(batch, model):
    prompt = PROMPT_HEAD % len(batch)
    for i, d in enumerate(batch):
        prompt += note_block(i, d["rel"], d["name"], d["descr"], d["kind"], d["body"])
    data = claude_json(prompt, model=model, timeout=420, system=SYSTEM)
    if isinstance(data, dict):
        data = data.get("notes") or data.get("results") or [data]

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
        d = batch[idx]
        join = lambda k: " ".join(  # noqa: E731
            str(x).strip() for x in (obj.get(k) or []) if str(x).strip()
        )[:1200]
        out.append({
            "hash": d["hash"],
            "rel": d["rel"],
            "topics": join("topics"),
            "aliases": join("aliases"),
            "triggers": join("triggers"),
            "entities": join("entities"),
            "summary": str(obj.get("summary", "")).strip()[:400],
            "model": model,
        })
    return out


def run(limit=None, batch_size=8, workers=4, model="haiku", force=False, verbose=True):
    con = connect()
    todo = pending(con, force=force)
    con.close()
    if limit:
        todo = todo[:limit]
    if not todo:
        if verbose:
            print("nothing to enrich; every memory is current")
        return 0

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    if verbose:
        print("enriching %d memories in %d batches, %d workers, model=%s"
              % (len(todo), len(batches), workers, model))

    t0 = time.time()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(enrich_batch, b, model): b for b in batches}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                rows = fut.result()
            except LLMError as exc:
                failed += len(b)
                log_line("memory-enrich.log", "ERROR", b[0]["rel"], str(exc)[:200])
                if verbose:
                    print("  batch failed (%s): %s" % (b[0]["rel"], str(exc)[:120]))
                continue
            with MemoryLock():
                con = connect()
                for r in rows:
                    con.execute(
                        "INSERT INTO enrich(hash, rel, topics, aliases, triggers, "
                        " entities, summary, model, ts) VALUES (?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(hash) DO UPDATE SET "
                        " topics=excluded.topics, aliases=excluded.aliases, "
                        " triggers=excluded.triggers, entities=excluded.entities, "
                        " summary=excluded.summary, model=excluded.model, ts=excluded.ts",
                        (r["hash"], r["rel"], r["topics"], r["aliases"], r["triggers"],
                         r["entities"], r["summary"], r["model"], time.time()),
                    )
                con.commit()
                con.close()
            done += len(rows)
            if verbose:
                print("  +%d (%d/%d) %.0fs" % (len(rows), done, len(todo), time.time() - t0))

    with MemoryLock():
        con = connect()
        total = con.execute("SELECT count(*) FROM enrich").fetchone()[0]
        set_meta(con, "enrich_sig", "%d:%.0f" % (total, time.time()))
        con.commit()
        con.close()

    log_line("memory-enrich.log", "RUN", "enriched=%d" % done, "failed=%d" % failed,
             "%.0fs" % (time.time() - t0))
    if verbose:
        print("enriched %d, failed %d, %.0fs. Run `recall.py index` to fold in."
              % (done, failed, time.time() - t0))
    return done


def main():
    ap = argparse.ArgumentParser(prog="enrich.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("run")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default="haiku")
    p.add_argument("--force", action="store_true")

    sub.add_parser("status")
    p = sub.add_parser("show")
    p.add_argument("rel")

    args = ap.parse_args()

    if args.cmd == "run":
        run(limit=args.limit, batch_size=args.batch, workers=args.workers,
            model=args.model, force=args.force)
        return 0

    if args.cmd == "status":
        con = connect()
        todo = len(pending(con))
        have = con.execute("SELECT count(*) FROM enrich").fetchone()[0]
        print("enriched rows: %d" % have)
        print("pending:       %d" % todo)
        con.close()
        return 0

    if args.cmd == "show":
        con = connect()
        row = con.execute(
            "SELECT rel, topics, aliases, triggers, entities, summary, model "
            "FROM enrich WHERE rel LIKE ? ORDER BY ts DESC LIMIT 1",
            ("%" + args.rel + "%",),
        ).fetchone()
        con.close()
        if not row:
            print("no enrichment for %s" % args.rel)
            return 1
        print(json.dumps(dict(zip(
            ["rel", "topics", "aliases", "triggers", "entities", "summary", "model"], row
        )), indent=2))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

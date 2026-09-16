#!/usr/bin/env python3
"""
verify.py - check what memories claim against what is actually true.

The gap this closes: a memory that was true in May and quietly stopped being
true was only caught if it named a path that vanished or contradicted a newer
memory. A note saying a job rotates a key every week reads as true forever,
even after the job is deleted, because it contradicts nothing and names no
missing path.

Two stages, and the split is the same one that makes recall fast and curation
safe:

  extract   a model reads a memory once and fills in TYPED ARGUMENTS to a fixed
            set of probes. It never writes a command. Free-form shell authored
            by a model and executed on a schedule is a remote code execution
            hole with extra steps.
  run       the probes execute deterministically. No model, no network beyond
            the probe itself.

WHAT A FAILURE IS ALLOWED TO MEAN is the other half of the design: a probe run
from outside can prove a thing exists, never that it is gone. Probes are split by
plane:

  control   does this file / branch / PR / secret / instance exist. Answered by
            a control plane that tells the truth from anywhere. A negative here
            is real and flags the memory.
  data      does this URL respond. A 200 proves the thing is alive. A timeout
            proves only that this machine cannot see it, and calling a healthy
            service "down" on that evidence is the easiest mistake to make. So
            a data-plane probe can return pass or unknown, never fail.

Usage:
  verify.py extract [--limit N] [--model haiku] [--force]
  verify.py run [--all] [--probe path] [--dry-run]
  verify.py status
  verify.py show <memory>
  verify.py drop <memory>          forget the claims for one memory
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memlib import (  # noqa: E402
    LLMError, MemoryLock, audit, claude_json, connect, doc_kind, log_line,
    read_doc, walk_memory,
)

DOPPLER = __import__("shutil").which("doppler") or "doppler"

# Paths that are SUPPOSED to come and go. A memory naming the worktree it was
# written in is not making a claim about the world that can rot; flagging it
# buries the findings that matter.
import re as _re
EPHEMERAL_PATH = _re.compile(
    r"/\.worktrees?[-/]|/worktrees?/|/node_modules/|/dist/|/build/|"
    r"/\.next/|/scratchpad/|\.dmg$|\.zip$|\.log$")
RECHECK = {"control": 20 * 3600, "data": 6 * 3600}
CMD_TIMEOUT = 25


# ---------------------------------------------------------------- probes
#
# Every probe takes a validated dict and returns (status, detail).
# status is one of: pass, fail, unknown.
# No probe ever takes a command, a shell string, or an interpolated argument.


def _run(argv, timeout=CMD_TIMEOUT):
    """Run a fixed argv. Never a shell string, never model-authored."""
    if not shutil.which(argv[0]) and not os.path.exists(argv[0]):
        return None, "%s not installed" % argv[0]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timed out"
    except OSError as exc:
        return None, str(exc)[:120]
    return p, ""


def probe_path(args):
    """control: a file or directory the memory points at."""
    target = args["path"]
    exists = os.path.exists(target)
    want = args.get("expect", "exists") == "exists"
    if exists == want:
        return "pass", "%s %s" % (target, "exists" if exists else "is absent, as claimed")
    return "fail", "%s %s" % (target, "does not exist" if want else "still exists")


def probe_git_branch(args):
    """control: a branch on the remote, asked of the remote itself."""
    repo, branch = args["repo"], args["branch"]
    if not os.path.isdir(os.path.join(repo, ".git")):
        return "unknown", "%s is not a git checkout on this machine" % repo
    p, err = _run(["git", "-C", repo, "ls-remote", "--heads", "origin", branch])
    if p is None:
        return "unknown", err
    if p.returncode != 0:
        return "unknown", (p.stderr or "").strip()[:160]
    found = bool(p.stdout.strip())
    want = args.get("expect", "exists") == "exists"
    if found == want:
        return "pass", "%s %s on origin" % (branch, "exists" if found else "is gone, as claimed")
    return "fail", "%s %s on origin" % (branch, "is gone" if want else "still exists")


def probe_gh_pr(args):
    """control: the state of a pull request."""
    repo, number = args["repo"], int(args["number"])
    want = args.get("expect", "merged").upper()
    p, err = _run(["gh", "pr", "view", str(number), "--repo", repo,
                   "--json", "state,mergedAt", "-q", ".state"])
    if p is None:
        return "unknown", err
    if p.returncode != 0:
        return "unknown", (p.stderr or "").strip()[:160]
    state = (p.stdout or "").strip().upper()
    if not state:
        return "unknown", "no state returned"
    if state == want:
        return "pass", "%s#%d is %s" % (repo, number, state)
    return "fail", "%s#%d is %s, memory says %s" % (repo, number, state, want)


def probe_gh_repo(args):
    """control: a repository exists, and optionally its visibility."""
    repo = args["repo"]
    p, err = _run(["gh", "repo", "view", repo, "--json", "visibility", "-q", ".visibility"])
    if p is None:
        return "unknown", err
    if p.returncode != 0:
        if "Could not resolve" in (p.stderr or "") or "not found" in (p.stderr or "").lower():
            return "fail", "%s not found" % repo
        return "unknown", (p.stderr or "").strip()[:160]
    vis = (p.stdout or "").strip().lower()
    want = args.get("expect", "exists").lower()
    if want in ("exists", ""):
        return "pass", "%s exists (%s)" % (repo, vis)
    if vis == want:
        return "pass", "%s is %s" % (repo, vis)
    return "fail", "%s is %s, memory says %s" % (repo, vis, want)


def probe_doppler_project(args):
    """control: a Doppler project exists."""
    p, err = _run([DOPPLER, "projects", "--json"])
    if p is None or p.returncode != 0:
        return "unknown", err or (p.stderr or "").strip()[:160]
    try:
        names = {x.get("name") or x.get("id") for x in json.loads(p.stdout or "[]")}
    except ValueError:
        return "unknown", "could not parse doppler output"
    want = args.get("expect", "exists") == "exists"
    found = args["project"] in names
    if found == want:
        return "pass", "doppler project %s %s" % (args["project"],
                                                  "exists" if found else "is gone, as claimed")
    return "fail", "doppler project %s %s (%d projects visible)" % (
        args["project"], "does not exist" if want else "still exists", len(names))


def probe_doppler_secret(args):
    """control: a named secret exists in a Doppler project/config."""
    project, config, name = args["project"], args["config"], args["name"]
    p, err = _run([DOPPLER, "secrets", "--project", project, "--config", config,
                   "--only-names", "--json"])
    if p is None or p.returncode != 0:
        return "unknown", err or (p.stderr or "").strip()[:160]
    try:
        names = set(json.loads(p.stdout or "{}").keys())
    except ValueError:
        return "unknown", "could not parse doppler output"
    want = args.get("expect", "exists") == "exists"
    found = name in names
    if found == want:
        return "pass", "%s %s in %s/%s" % (name, "set" if found else "absent, as claimed",
                                           project, config)
    return "fail", "%s %s in %s/%s" % (
        name, "is NOT set" if want else "is still set", project, config)


def probe_aws_ec2(args):
    """control: the state of one EC2 instance. Reading the control plane, not the box."""
    iid, region = args["instance_id"], args.get("region", "us-east-1")
    p, err = _run(["aws", "ec2", "describe-instances", "--instance-ids", iid,
                   "--region", region, "--query",
                   "Reservations[0].Instances[0].State.Name", "--output", "text"], timeout=40)
    if p is None or p.returncode != 0:
        return "unknown", err or (p.stderr or "").strip()[:160]
    state = (p.stdout or "").strip()
    want = args.get("expect_state", "").strip().lower()
    if not state or state == "None":
        return "fail", "%s not found in %s" % (iid, region)
    if not want:
        return "pass", "%s is %s" % (iid, state)
    if state.lower() == want:
        return "pass", "%s is %s" % (iid, state)
    return "fail", "%s is %s, memory says %s" % (iid, state, want)


def probe_url(args):
    """data: a 200 proves life. A timeout proves nothing at all.

    This probe may return pass or unknown. It is never allowed to return fail
    on a network error, because 'times out from my machine' is not 'down'.
    """
    url = args["url"]
    want = int(args.get("expect_status", 200))
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "quipu-verify"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            code = resp.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:
        return "unknown", "unreachable from this machine (%s). Not evidence it is down." % type(exc).__name__
    if code == want:
        return "pass", "%s -> %d" % (url, code)
    # A definite HTTP response that disagrees is real evidence, but a 401/403
    # usually means a perimeter, not a dead service.
    if code in (401, 403):
        return "unknown", "%s -> %d, behind a perimeter from here" % (url, code)
    return "fail", "%s -> %d, memory says %d" % (url, code, want)


PROBES = {
    "path":            (probe_path,            "control", ("path",)),
    "git_branch":      (probe_git_branch,      "control", ("repo", "branch")),
    "gh_pr":           (probe_gh_pr,           "control", ("repo", "number")),
    "gh_repo":         (probe_gh_repo,         "control", ("repo",)),
    "doppler_project": (probe_doppler_project, "control", ("project",)),
    "doppler_secret":  (probe_doppler_secret,  "control", ("project", "config", "name")),
    "aws_ec2":         (probe_aws_ec2,         "control", ("instance_id",)),
    "url":             (probe_url,             "data",    ("url",)),
}


# ---------------------------------------------------------------- extraction

SYSTEM = (
    "You turn engineering notes into checkable assertions. You never invent a "
    "fact and you never guess an identifier that is not written in the note. "
    "Most notes yield no claims at all. You output JSON only."
)

PROMPT = """Below are %d memory notes. For each, extract the assertions that a
machine could CHECK, using only the probes listed. Most notes yield none.

PROBES and their exact arguments:
  path            {"path": "<absolute path>", "expect": "exists"|"absent"}
  git_branch      {"repo": "<absolute path to checkout>", "branch": "<name>", "expect": "exists"|"absent"}
  gh_pr           {"repo": "<owner/name>", "number": <int>, "expect": "MERGED"}
                  ONLY for a PR the note says was merged. Never extract a claim
                  that a PR is open or under review: that changes by the minute
                  and is not a fact worth remembering.
  gh_repo         {"repo": "<owner/name>", "expect": "exists"|"public"|"private"}
  doppler_project {"project": "<name>", "expect": "exists"|"absent"}
  doppler_secret  {"project": "<name>", "config": "<dev|stg|prd>", "name": "<SECRET_NAME>", "expect": "exists"|"absent"}
  aws_ec2         {"instance_id": "i-...", "region": "<region>", "expect_state": "running"|"stopped"}
  url             {"url": "https://...", "expect_status": 200}

For each claim return:
  "i"       the note index
  "probe"   one of the probe names above
  "args"    the argument object, using ONLY values written literally in the note
  "expect"  one short sentence: what the note asserts, in plain words
  "quote"   the sentence from the note that makes the claim, verbatim, under 200 chars

HARD RULES:
- Every value in "args" must appear literally in the note. Never guess a repo
  name, an instance id, a secret name, a branch, or a path. If the note says
  "the demo box" without an instance id, there is no claim.
- Do not extract claims about preferences, opinions, style rules, or how
  someone should behave. Only claims about the state of the world.
- Do not extract a claim the note itself says is already fixed, historical, or
  no longer true. This includes notes that DESCRIBE something being broken,
  missing or retired: if the note's point is that a thing does not exist, there
  is no claim that it does.
- Do not extract claims from a note that is documentation about tooling, where
  the identifiers appear as EXAMPLES rather than as things the note depends on.
- Skip template paths containing <>, {}, * or ?.
- At most 3 claims per note. Prefer none over a shaky one.

Output a JSON array of claim objects across all notes, nothing else. An empty
array is a good answer.

NOTES:
"""


def note_block(i, rel, name, descr, body, cap=1600):
    body = body.strip()
    if len(body) > cap:
        body = body[:cap] + " ..."
    return "\n--- NOTE %d ---\nfile: %s\nname: %s\ndescription: %s\nbody:\n%s\n" % (
        i, rel, name, descr or "(none)", body)


def validate(claim):
    """Reject anything that is not a known probe with its required arguments."""
    probe = str(claim.get("probe", "")).strip()
    spec = PROBES.get(probe)
    if not spec:
        return None
    _fn, plane, required = spec
    args = claim.get("args")
    if not isinstance(args, dict):
        return None
    for key in required:
        v = args.get(key)
        if v is None or str(v).strip() == "":
            return None
        if any(ch in str(v) for ch in "<>{}*?$`;|&\n"):
            return None       # a template or an injection attempt, not a value
    # MERGED is terminal; OPEN and CLOSED are moods. A memory asserting a PR is
    # open is asserting something that was false minutes after it was written,
    # so the fix is not to check it more often, it is to never treat it as a
    # claim about the world.
    if probe == "gh_pr" and str(args.get("expect", "")).upper() != "MERGED":
        return None
    if probe == "path":
        target = str(args["path"])
        if not target.startswith("/") or EPHEMERAL_PATH.search(target):
            return None
    if probe == "url" and not str(args["url"]).startswith("https://"):
        return None
    return {
        "probe": probe,
        "plane": plane,
        "args": json.dumps(args, sort_keys=True),
        "expect": str(claim.get("expect", ""))[:250],
        "quote": str(claim.get("quote", ""))[:250],
    }


def pending(con, force=False):
    out = []
    for path, rel, _mtime in walk_memory():
        try:
            meta, body, sha, _raw = read_doc(path)
        except OSError:
            continue
        kind = doc_kind(meta, rel)
        if kind in ("feedback", "user"):
            continue          # rules are not claims about the world
        if not force and con.execute(
                "SELECT 1 FROM claim WHERE hash=? LIMIT 1", (sha,)).fetchone():
            continue
        if not force and con.execute(
                "SELECT 1 FROM meta WHERE k=?", ("noclaims:" + sha,)).fetchone():
            continue
        out.append({"rel": rel, "hash": sha, "body": body,
                    "name": meta.get("name") or os.path.basename(rel)[:-3],
                    "descr": meta.get("description", "")})
    return out


def extract_batch(batch, model):
    prompt = PROMPT % len(batch)
    for i, d in enumerate(batch):
        prompt += note_block(i, d["rel"], d["name"], d["descr"], d["body"])
    data = claude_json(prompt, model=model, timeout=420, system=SYSTEM)
    if isinstance(data, dict):
        data = data.get("claims") or data.get("results") or []
    rows = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        try:
            idx = int(obj.get("i", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(batch):
            continue
        v = validate(obj)
        if v:
            v["rel"] = batch[idx]["rel"]
            v["hash"] = batch[idx]["hash"]
            rows.append(v)
    return rows


def extract(limit=None, batch_size=6, workers=3, model="haiku", force=False, verbose=True):
    con = connect()
    todo = pending(con, force=force)
    con.close()
    if limit:
        todo = todo[:limit]
    if not todo:
        if verbose:
            print("every memory has been read for claims")
        return 0

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    if verbose:
        print("reading %d memories for checkable claims (%d batches)" % (len(todo), len(batches)))

    found = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_batch, b, model): b for b in batches}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                rows = fut.result()
            except LLMError as exc:
                log_line("memory-verify.log", "EXTRACT-ERROR", b[0]["rel"], str(exc)[:160])
                continue
            with MemoryLock():
                con = connect()
                for r in rows:
                    con.execute(
                        "INSERT INTO claim(rel, hash, probe, args, expect, quote, plane, created) "
                        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(hash, probe, args) DO NOTHING",
                        (r["rel"], r["hash"], r["probe"], r["args"], r["expect"],
                         r["quote"], r["plane"], time.time()))
                # Remember the notes that yielded nothing so they are not re-read.
                claimed = {r["hash"] for r in rows}
                for d in b:
                    if d["hash"] not in claimed:
                        con.execute(
                            "INSERT INTO meta(k,v) VALUES(?,?) "
                            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                            ("noclaims:" + d["hash"], str(time.time())))
                con.commit()
                con.close()
            found += len(rows)
            if verbose:
                print("  +%d claim(s)  (%.0fs)" % (len(rows), time.time() - t0))

    log_line("memory-verify.log", "EXTRACT", "claims=%d" % found,
             "notes=%d" % len(todo), "%.0fs" % (time.time() - t0))
    if verbose:
        print("extracted %d claim(s) from %d memories in %.0fs"
              % (found, len(todo), time.time() - t0))
    return found


# ---------------------------------------------------------------- running


def prune(con):
    """Drop claims extracted from text that no longer exists.

    Claims are keyed by content hash, like enrichment. Edit a memory to correct
    it and the old claims survive against the old hash, so a memory you just
    fixed keeps failing on what it used to say. Found exactly that way: seven
    Doppler corrections landed and all seven kept reporting failure.
    """
    n = con.execute(
        "DELETE FROM claim WHERE hash NOT IN (SELECT hash FROM doc)").rowcount
    con.execute(
        "DELETE FROM meta WHERE k LIKE 'noclaims:%' "
        "AND substr(k, 10) NOT IN (SELECT hash FROM doc)")
    return n


def due_claims(con, all_of_them=False, probe=None):
    q = "SELECT id, rel, probe, args, plane, status, checked FROM claim"
    where, params = [], []
    if probe:
        where.append("probe = ?")
        params.append(probe)
    if where:
        q += " WHERE " + " AND ".join(where)
    rows = con.execute(q, params).fetchall()
    now = time.time()
    out = []
    for cid, rel, p, args, plane, status, checked in rows:
        if all_of_them or now - checked > RECHECK.get(plane, 20 * 3600):
            out.append((cid, rel, p, args, plane))
    return out


def run(all_of_them=False, probe=None, dry=False, workers=4, verbose=True):
    con = connect()
    if not dry:
        with MemoryLock():
            gone = prune(con)
            con.commit()
        if gone and verbose:
            print("dropped %d claim(s) from memories that have since been edited" % gone)
    todo = due_claims(con, all_of_them, probe)
    con.close()
    if not todo:
        if verbose:
            print("no claims due for checking")
        return []

    if verbose:
        print("checking %d claim(s)" % len(todo))

    def check(item):
        cid, rel, pname, raw_args, plane = item
        fn, _plane, _req = PROBES[pname]
        try:
            args = json.loads(raw_args)
        except ValueError:
            return cid, rel, pname, plane, "unknown", "unreadable arguments"
        try:
            status, detail = fn(args)
        except Exception as exc:
            status, detail = "unknown", "probe error: %s" % str(exc)[:140]
        # The rule that keeps this honest: a data-plane probe may prove life,
        # never death. Downgrade any failure it reports.
        if plane == "data" and status == "fail":
            status, detail = "unknown", detail + " (data plane: not proof of failure)"
        return cid, rel, pname, plane, status, detail

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(check, t) for t in todo]):
            results.append(fut.result())

    if not dry:
        with MemoryLock():
            con = connect()
            for cid, rel, pname, _plane, status, detail in results:
                prev = con.execute("SELECT status FROM claim WHERE id=?", (cid,)).fetchone()
                con.execute(
                    "UPDATE claim SET status=?, detail=?, checked=? WHERE id=?",
                    (status, detail[:300], time.time(), cid))
                if status == "fail" and (not prev or prev[0] != "fail"):
                    audit(con, "claim-failed", rel, "%s: %s" % (pname, detail[:200]), "verify")
                    con.execute(
                        "INSERT INTO review(ts, kind, rels, detail, status) "
                        "VALUES (?,?,?,?, 'open')",
                        (time.time(), "unverified", rel,
                         "%s probe disagrees with the memory: %s" % (pname, detail[:220])))
            # Close review items whose claim now passes. A queue that only grows
            # is a queue nobody opens: if the world came back into line with the
            # memory, or the memory was corrected, that item is finished and the user
            # should never see it.
            passing = {r[0] for r in con.execute(
                "SELECT rel FROM claim WHERE status='pass' "
                "AND rel NOT IN (SELECT rel FROM claim WHERE status='fail')")}
            for rel in passing:
                con.execute(
                    "UPDATE review SET status='resolved' WHERE kind='unverified' "
                    "AND rels=? AND status='open'", (rel,))
            # And drop items pointing at claims that no longer exist at all.
            con.execute(
                "UPDATE review SET status='resolved' WHERE kind='unverified' "
                "AND status='open' AND rels NOT IN (SELECT rel FROM claim)")

            # A memory with any failing claim is marked, so recall labels it.
            con.execute("UPDATE doc SET stale_note='' WHERE stale_note LIKE 'unverified:%'")
            for rel, n in con.execute(
                    "SELECT rel, count(*) FROM claim WHERE status='fail' GROUP BY rel"):
                con.execute(
                    "UPDATE doc SET stale_note=? WHERE rel=? AND stale_note=''",
                    ("unverified: %d claim(s) no longer check out" % n, rel))
            con.commit()
            con.close()

    tally = {}
    for _cid, _rel, _p, _pl, status, _d in results:
        tally[status] = tally.get(status, 0) + 1
    log_line("memory-verify.log", "RUN", *["%s=%d" % kv for kv in sorted(tally.items())])
    if verbose:
        print("  " + ", ".join("%s %d" % (k, v) for k, v in sorted(tally.items())))
        for _cid, rel, pname, _pl, status, detail in sorted(results, key=lambda r: r[4]):
            if status != "pass":
                print("  %-7s %-16s %-44s %s" % (status, pname, rel[:44], detail[:90]))
    return results


# ---------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser(prog="verify.py")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("extract")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--model", default="haiku")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run")
    p.add_argument("--all", action="store_true")
    p.add_argument("--probe")
    p.add_argument("--dry-run", action="store_true")

    sub.add_parser("status")
    p = sub.add_parser("show")
    p.add_argument("memory")
    p = sub.add_parser("drop")
    p.add_argument("memory")

    args = ap.parse_args()

    if args.cmd == "extract":
        extract(limit=args.limit, batch_size=args.batch, workers=args.workers,
                model=args.model, force=args.force)
        return 0

    if args.cmd == "run":
        run(all_of_them=args.all, probe=args.probe, dry=args.dry_run)
        return 0

    if args.cmd == "status":
        con = connect()
        total = con.execute("SELECT count(*) FROM claim").fetchone()[0]
        print("claims:   %d across %d memories" % (
            total, con.execute("SELECT count(DISTINCT rel) FROM claim").fetchone()[0]))
        if total:
            print("by status: %s" % ", ".join("%s=%d" % r for r in con.execute(
                "SELECT status, count(*) FROM claim GROUP BY status ORDER BY 2 DESC")))
            print("by probe:  %s" % ", ".join("%s=%d" % r for r in con.execute(
                "SELECT probe, count(*) FROM claim GROUP BY probe ORDER BY 2 DESC")))
            print()
            rows = con.execute(
                "SELECT rel, probe, detail FROM claim WHERE status='fail' ORDER BY rel").fetchall()
            if rows:
                print("FAILING (the memory says one thing, the world says another):")
                for rel, probe, detail in rows:
                    print("  %-46s %-14s %s" % (rel[:46], probe, detail[:80]))
            else:
                print("nothing is currently failing")
        con.close()
        return 0

    if args.cmd == "show":
        con = connect()
        rows = con.execute(
            "SELECT probe, args, expect, status, detail, checked FROM claim "
            "WHERE rel LIKE ? ORDER BY probe", ("%" + args.memory + "%",)).fetchall()
        con.close()
        if not rows:
            print("no claims extracted for %s" % args.memory)
            return 1
        for probe, a, expect, status, detail, checked in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(checked)) if checked else "never"
            print("%-7s %-16s %s" % (status, probe, a))
            print("        claims: %s" % expect)
            print("        checked %s: %s" % (when, detail or "-"))
        return 0

    if args.cmd == "drop":
        with MemoryLock():
            con = connect()
            n = con.execute("DELETE FROM claim WHERE rel LIKE ?",
                            ("%" + args.memory + "%",)).rowcount
            con.commit()
            con.close()
        print("dropped %d claim(s)" % n)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

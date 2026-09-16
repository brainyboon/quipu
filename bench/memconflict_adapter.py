"""MemConflict evaluation adapter for quipu.

Drops into the hermes-memconflict harness as one more provider folder. The
shared ``benchmark/eval_common.py`` driver owns everything provider-agnostic:
dataset iteration, dialogue flattening, the answer prompt, the answer LLM call,
scoring and judging. This file supplies only the three methods that are ours.

WHAT IS DIFFERENT ABOUT quipu (vs mem0 / Honcho / Hindsight)
-----------------------------------------------------------------
Every other provider in this benchmark is a memory SERVICE: it ingests turns,
extracts facts with its own LLM, and answers a similarity query. quipu is
a Claude Code hook system over plain markdown, and it is deliberately built as
TWO stores with different jobs, so the adapter submits both:

  1. DISTILLED MEMORIES. At session end a model reads the session and writes at
     most three notes, only for things worth keeping forever. Most sessions
     produce none. A second offline pass writes retrieval aids (aliases,
     triggers) so the index answers to the words a person would actually type.
     Retrieval is SQLite FTS5 with IDF-weighted top-term scoring, freshness
     decay and type weights. No model runs at query time at all.

  2. RAW SESSION SEARCH. Full text over every turn ever said, also FTS5, also
     no model. This is the half that answers "what did we discuss", which the
     distilled store is designed NOT to hold.

That split is the whole architecture, and it is the thing under test. A
distilled-only submission would be a strawman: MemConflict asks fact-level
questions about specific past dialogue, which is exactly what store 2 is for.
Reporting them separately (see ``QUIPU_ARM``) is more informative than either
alone, so the adapter supports running each arm on its own.

FAIRNESS
--------
The harness contract gives every provider the same internal model,
``qwen3.5-4b`` on ``vllm-gen``. quipu normally runs its offline
passes through ``claude -p``, which a container cannot reach and which would
not be a fair comparison. Setting ``QUIPU_LLM_BASE_URL`` routes every internal
model call to the shared server instead, so capture and enrichment run on the
same model as everyone else's extraction.

NO MODEL ON THE QUERY PATH
--------------------------
``recall()`` here makes zero model calls, by construction. Whatever this scores,
its retrieval cost is a SQLite query. That is the claim being measured.

Env:
  QUIPU_ARM          both | memories | sessions      (default both)
  QUIPU_LLM_BASE_URL http://vllm-gen:8000/v1
  QUIPU_LLM_MODEL    qwen3.5-4b
  QUIPU_ENRICH       1 to run the enrichment pass (default 1)
  QUIPU_SRC          path to the plugin's src/ (default ../src)
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark"))
import eval_common  # noqa: E402

# MemConflict asks about family, health and money, so the benchmark records them.
os.environ.setdefault("QUIPU_FACTS_SCOPE", "personal")

QUIPU_SRC = os.environ.get(
    "QUIPU_SRC",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")),
)
sys.path.insert(0, QUIPU_SRC)

import memlib  # noqa: E402
import recall  # noqa: E402
import sessions as sessions_mod  # noqa: E402
import capture as capture_mod  # noqa: E402
import enrich as enrich_mod  # noqa: E402

ARM = os.environ.get("QUIPU_ARM", "both").lower()
DO_ENRICH = os.environ.get("QUIPU_ENRICH", "1") not in ("0", "false", "no")
# legacy: the retrieval measured on 2026-09-13 (bm25 over the question's words,
#         one turn per session, raw scores merged across stores).
# hybrid: sessions.find() + recall.search_hybrid(), fused by reciprocal rank.
RETRIEVAL = os.environ.get("QUIPU_RETRIEVAL", "hybrid").lower()
DO_FACTS = os.environ.get("QUIPU_FACTS", "1") not in ("0", "false", "no")
# How much a distilled note outranks a raw turn at equal rank. A note is a
# checked, current statement; a turn is one line of a conversation.
W_DISTILLED = float(os.environ.get("QUIPU_W_DISTILLED", "1.5"))
# At most this many distilled notes in the answer window. Measured 2026-09-14 on
# personas 5-6: letting fact notes rank freely nearly doubled answer accuracy on
# change questions (0.268 -> 0.48) and wrecked it on conditional ones (0.70 ->
# 0.06), because notes filled all five slots and pushed out the conversation
# turns that carry the condition. One store must never monopolise the window.
MAX_DISTILLED = int(os.environ.get("QUIPU_MAX_DISTILLED", "0")) or None
REAL_DATES = os.environ.get("QUIPU_REAL_DATES", "1") not in ("0", "false", "no")
WANT_MEMORIES = ARM in ("both", "memories")
WANT_SESSIONS = ARM in ("both", "sessions")
# Sessions per persona in the dataset, used to lay them out on a plausible
# weekly cadence ending near today.
SESSION_SPAN = int(os.environ.get("QUIPU_SESSION_SPAN", "60"))


# --------------------------------------------------------------------------
# dialogue -> the two stores
# --------------------------------------------------------------------------

def _iso(ts):
    """Seconds since epoch to the timestamp shape the harness stores."""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return ""


def _dialogue_text(dialogue, cap=14000):
    """The session as the capture prompt expects to see it."""
    turns = []
    for msg in dialogue:
        role = (msg.get("role") or msg.get("speaker") or "").lower()
        text = msg.get("content") or msg.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        # Neutral speaker labels. A capture model reading a named speaker names
        # the entity after them, while every benchmark question asks about
        # "the user".
        who = "USER" if role in ("user", "human") else "ASSISTANT"
        turns.append("%s: %s" % (who, text.strip()[:2500]))
    out = "\n\n".join(turns)
    if len(out) > cap:
        # Keep the opening and the tail: the ask and what it concluded.
        out = out[: cap // 3] + "\n\n[...]\n\n" + out[-(2 * cap // 3):]
    return out, len(turns)


def _index_raw_session(ctx, session_id, dialogue, when):
    """Store the raw turns so session search can find them. No model, no cost."""
    con = memlib.connect()
    con.executescript(sessions_mod.SCHEMA)
    con.execute("DELETE FROM turn WHERE sid=?", (session_id,))
    seq = 0
    for msg in dialogue:
        role = (msg.get("role") or msg.get("speaker") or "").lower()
        text = msg.get("content") or msg.get("text") or ""
        if not isinstance(text, str):
            continue
        text = sessions_mod.clean(text)
        if len(text) < sessions_mod.MIN_TURN_CHARS:
            continue
        con.execute(
            "INSERT INTO turn(sid, seq, role, ts, text) VALUES (?,?,?,?,?)",
            (session_id, seq, "user" if role in ("user", "human") else "assistant",
             when, text[:6000]))
        seq += 1
    if seq:
        opening = next((m.get("content") or "" for m in dialogue
                        if (m.get("role") or "").lower() in ("user", "human")), "")[:300]
        con.execute(
            "INSERT INTO session(id, path, started, ended, turns, bytes, opening, indexed) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            " ended=excluded.ended, turns=excluded.turns, opening=excluded.opening",
            (session_id, "", when, when, seq, 0, opening, when))
    con.commit()
    con.close()
    return seq


def _capture_session(ctx, dialogue, session_date=""):
    """Write the distilled memories and facts for this session, as production does."""
    text, n_turns = _dialogue_text(dialogue)
    if n_turns < capture_mod.MIN_TURNS or len(text) < capture_mod.MIN_CHARS:
        return 0, 0
    try:
        if DO_FACTS:
            props, facts = capture_mod.propose_all(text, n_turns, session_date=session_date)
        else:
            props, facts = capture_mod.propose(text, n_turns), []
    except memlib.LLMError:
        ctx["capture_errors"] = ctx.get("capture_errors", 0) + 1
        return 0, 0

    for fact in facts:
        fname, status = capture_mod.write_fact(fact, session_date=session_date,
                                               session_id=ctx["persona"])
        if fname and status in ("new", "changed", "confirmed"):
            ctx["facts_" + status] = ctx.get("facts_" + status, 0) + 1

    written = dup = 0
    for prop in props:
        fname, status = capture_mod.write_memory(prop, ctx["persona"])
        if fname:
            written += 1
        elif status.startswith("duplicate of"):
            dup += 1
    ctx["captured"] = ctx.get("captured", 0) + written
    ctx["capture_dupes"] = ctx.get("capture_dupes", 0) + dup
    return written, dup


# --------------------------------------------------------------------------
# the binding
# --------------------------------------------------------------------------

class MemoryBinding(eval_common.ProviderBinding):
    # Stamped into every result row so a mixed Results dir stays attributable.
    memory_system = "quipu"
    store_id_key = "Quipu_Store_ID"
    runtime_summary_key = "Quipu_Runtime_Summary"
    stage_name = "quipu_answer_generation"
    stage_note = "quipu retrieval and question answering"
    # False on purpose. Our production hook injects under its own char budget
    # rather than a top-K slice, which is the case plugin_native_recall exists
    # for, but leaving it False keeps us byte-identical with mem0, Honcho and
    # the rest. Comparability beats flattering our own defaults.
    plugin_native_recall = False

    def begin_persona(self, persona_item):
        persona = str(persona_item.get("Persona_ID")
                      or persona_item.get("persona_id")
                      or persona_item.get("User_ID") or "p")
        root = tempfile.mkdtemp(prefix="quipu-%s-" % persona)
        # Every persona is a completely separate store. rebind() moves the whole
        # system, including modules that captured MEM_ROOT at import time.
        memlib.rebind(os.path.join(root, "memory"),
                      os.path.join(root, ".memory-state"))
        con = memlib.connect()
        con.executescript(sessions_mod.SCHEMA)
        con.close()
        return {"persona": persona, "root": root, "t0": time.monotonic(),
                # The harness requires store_id: it identifies the isolated
                # store in every result row.
                "store_id": os.path.basename(root),
                "sessions": 0, "captured": 0, "capture_dupes": 0,
                "capture_errors": 0, "enriched": 0, "raw_turns": 0,
                "ingest_ms": 0.0}

    def ingest_session(self, ctx, session_item, dialogue, session_index):
        t0 = time.monotonic()
        # A monotonic per-session clock. MemConflict is mostly about which of
        # two statements is CURRENT, so recency has to be real rather than
        # every memory sharing one wall-clock timestamp. Roughly weekly
        # sessions ending near now: the first version put session 0 in 1999,
        # which made freshness decay bottom out for every persona equally and
        # quietly removed the signal this arm depends on.
        when = time.time() - max(0, (SESSION_SPAN - session_index)) * 7 * 86400.0
        session_date = ""
        if REAL_DATES:
            # The harness contract: every provider ingests at the dataset's own
            # session date so the chronology is identical across systems. Until
            # 2026-09-14 this adapter invented a weekly clock instead.
            dt = eval_common.Parse_Session_Timestamp(session_item)
            if dt is not None:
                when = dt.timestamp()
                session_date = dt.strftime("%Y-%m-%d")
        sid = str(session_item.get("Session_ID") or session_index)
        ctx["last_when"] = when

        added = 0
        if WANT_SESSIONS:
            n = _index_raw_session(ctx, sid, dialogue, when)
            ctx["raw_turns"] += n
            added += n
            if RETRIEVAL == "hybrid":
                sessions_mod.embed_new_turns()

        if WANT_MEMORIES:
            written, _dup = _capture_session(ctx, dialogue, session_date)
            added += written
            if DO_ENRICH:
                try:
                    ctx["enriched"] += enrich_mod.run(
                        batch_size=6, workers=2, verbose=False) or 0
                except Exception:
                    pass

        recall.build_index()
        ctx["sessions"] += 1
        ms = (time.monotonic() - t0) * 1000.0
        ctx["ingest_ms"] += ms
        # The harness stores this on Session_Memory_Metadata and both keys are
        # required by the contract.
        return {
            "Add_Duration_ms": round(ms, 1),
            "Dialogue_Added_To_Memory": added > 0,
            "Quipu_Memories_This_Session": ctx.get("captured", 0),
            "Quipu_Turns_This_Session": added,
        }

    def recall(self, ctx, question_text, top_k):
        """Zero model calls. Two SQLite reads, merged, scored, timed.

        Contract: (list of dicts, duration_ms). Each dict carries at least
        `memory`, `created_at` and `score`; anything else is diagnostics the
        scorer ignores.
        """
        t0 = time.monotonic()
        if RETRIEVAL == "hybrid":
            return self._recall_hybrid(ctx, question_text, top_k), (time.monotonic() - t0) * 1000.0
        items, seen = [], set()

        if WANT_MEMORIES:
            # floor=0.0 here on purpose. The production floor exists to keep
            # the hook SILENT on chit-chat, which is a different job from
            # "rank these against a question", and applying it would make us
            # abstain rather than answer. The benchmark scores wrong answers
            # at -1, so this is the harder setting for us, not the softer one.
            for hit in recall.search(question_text, limit=top_k * 2, floor=0.0):
                if hit["rel"] in seen:
                    continue
                seen.add(hit["rel"])
                body = (hit.get("descr") or "").strip() or hit.get("body", "")
                items.append({
                    "memory": body[:900],
                    "created_at": _iso(hit.get("mtime") or ctx.get("last_when", 0)),
                    "score": float(hit.get("score", 0.0)),
                    "quipu_source": "distilled",
                    "quipu_title": hit.get("title", ""),
                    "quipu_kind": hit.get("kind", ""),
                })

        if WANT_SESSIONS:
            for hit in sessions_mod.search(question_text, limit=top_k * 2):
                key = "s:%s:%s" % (hit["session"], hit["when"])
                if key in seen:
                    continue
                seen.add(key)
                items.append({
                    "memory": hit["text"][:900],
                    "created_at": hit["when"],
                    # bm25 is negative and more-negative is better; flip it so
                    # the harness sees a normal higher-is-better score.
                    "score": float(-hit.get("rank", 0.0)),
                    "quipu_source": "session",
                    "quipu_role": hit.get("role", ""),
                })

        items.sort(key=lambda d: -d["score"])
        return items, (time.monotonic() - t0) * 1000.0

    def _recall_hybrid(self, ctx, question_text, top_k):
        """Both stores ranked by meaning and by words, then fused by rank.

        Raw scores from the two stores are not comparable (IDF evidence against
        fused turn scores), so the old merge let whichever store happened to
        print bigger numbers win. Rank fusion needs no calibration between them.
        """
        lists = []
        if WANT_MEMORIES:
            notes = []
            for hit in recall.search_hybrid(question_text, limit=top_k * 2):
                body = (hit.get("descr") or "").strip() or hit.get("body", "")
                notes.append({
                    "memory": body[:900],
                    # A note is the store's state at question time. Its file mtime
                    # is the wall clock of this run, which would date a 2025 fact
                    # as 2026 in front of the answering model.
                    "created_at": _iso(ctx.get("last_when", 0)),
                    "quipu_source": "distilled", "quipu_kind": hit.get("kind", ""),
                    "quipu_title": hit.get("title", ""), "_key": "m:" + hit["rel"],
                })
            lists.append((W_DISTILLED, notes))
        if WANT_SESSIONS:
            turns = []
            for hit in sessions_mod.find(question_text, limit=top_k * 2):
                turns.append({
                    "memory": hit["text"][:900],
                    "created_at": _iso(hit.get("ts")),
                    "quipu_source": "session", "quipu_role": hit.get("role", ""),
                    "_key": "s:%s:%s" % (hit["sid"], hit["seq"]),
                })
            lists.append((1.0, turns))

        fused = {}
        for weight, lst in lists:
            for rank, it in enumerate(lst):
                k = it["_key"]
                if k not in fused:
                    fused[k] = [0.0, it]
                fused[k][0] += weight / (60.0 + rank)
        ranked = sorted(fused.values(), key=lambda x: -x[0])
        if MAX_DISTILLED is not None:
            head, spill, n_notes = [], [], 0
            for score, it in ranked:
                if it["quipu_source"] == "distilled" and len(head) < top_k:
                    if n_notes >= MAX_DISTILLED:
                        spill.append((score, it))
                        continue
                    n_notes += 1
                head.append((score, it))
            ranked = head + spill
        out = []
        for score, it in ranked:
            it = dict(it)
            it.pop("_key", None)
            it["score"] = round(score * 1000.0, 4)
            out.append(it)
        return out[: top_k * 2]

    def end_persona(self, ctx):
        shutil.rmtree(ctx.get("root", ""), ignore_errors=True)

    def persona_count_extras(self, ctx):
        return {
            "Total_Memories_Written": ctx.get("captured", 0),
            "Total_Raw_Turns_Indexed": ctx.get("raw_turns", 0),
            "Total_Ingest_ms": round(ctx.get("ingest_ms", 0.0), 1),
        }

    def persona_result_extras(self, ctx):
        return {
            "Quipu_Arm": ARM,
            "Quipu_Sessions": ctx.get("sessions", 0),
            "Quipu_Memories_Written": ctx.get("captured", 0),
            "Quipu_Memories_Deduped": ctx.get("capture_dupes", 0),
            "Quipu_Capture_Errors": ctx.get("capture_errors", 0),
            "Quipu_Raw_Turns_Indexed": ctx.get("raw_turns", 0),
            "Quipu_Enriched": ctx.get("enriched", 0),
            "Quipu_Retrieval": RETRIEVAL,
            "Quipu_Facts_New": ctx.get("facts_new", 0),
            "Quipu_Facts_Changed": ctx.get("facts_changed", 0),
            "Quipu_Facts_Confirmed": ctx.get("facts_confirmed", 0),
            "Quipu_Persona_Seconds": round(time.monotonic() - ctx.get("t0", 0), 1),
        }


def main():
    ap = argparse.ArgumentParser(prog="memconflict_adapter.py")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int)
    ap.add_argument("--max-sessions", type=int)
    ap.add_argument("--max-questions-per-session", type=int)
    ap.add_argument("--overwrite-existing-answers", action="store_true")
    args = ap.parse_args()

    if RETRIEVAL == "hybrid":
        try:
            import embed  # noqa: F401
            embed.model()
        except Exception as exc:
            sys.stderr.write("hybrid retrieval needs numpy and the embedding weights: %s\n" % exc)
            return 2

    if not memlib.LLM_BASE_URL and WANT_MEMORIES:
        sys.stderr.write(
            "QUIPU_LLM_BASE_URL is unset, so capture and enrichment would try to "
            "shell out to `claude -p`. Inside the benchmark that is both "
            "unavailable and unfair. Set it to the shared vllm-gen endpoint.\n")
        return 2

    print("quipu adapter: arm=%s retrieval=%s facts=%s enrich=%s real_dates=%s model=%s @ %s"
          % (ARM, RETRIEVAL, DO_FACTS, DO_ENRICH, REAL_DATES, memlib.LLM_MODEL,
             memlib.LLM_BASE_URL or "claude -p"))

    ok = eval_common.run_eval(
        binding=MemoryBinding(),
        input_jsonl_path=args.input,
        output_jsonl_path=args.output_jsonl,
        output_json_path=args.output_json,
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_sessions=args.max_sessions,
        max_questions_per_session=args.max_questions_per_session,
        overwrite_existing_answers=args.overwrite_existing_answers,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

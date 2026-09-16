# Quipu

Memory for Claude Code that keeps up when things change.

Claude Code starts every session knowing nothing you told it last week. This
plugin fixes that without slowing the prompt down. It captures what each session
learned, keeps facts about people and organisations current with a dated
history, and puts the right memory in front of the model before it answers.

No model runs on the prompt path. Recall is SQLite full-text search plus a small
static embedding table, so typing stays instant.

Named after the quipu, the Andean knotted cords that held a record without
writing a word.

## Results

Graded on [MemConflict](https://github.com/EngTurtle/hermes-memconflict), a
benchmark of simulated users whose facts change over time, where a wrong answer
scores minus one. Every competitor's published answers were re-graded with the
same judge as ours: same 5 users, same 616 questions, same GPU.

| system | score | per question | evidence in top 5 | filing one conversation | context per question |
| --- | --- | --- | --- | --- | --- |
| mem0 | 0.520 | 0.477 | 69.4% | 31.6 s | 2,759 chars |
| **quipu** | **0.520** | **0.480** | 64.0% | **1.6 s** | **1,507 chars** |
| Supermemory | 0.483 | 0.429 | 61.1% | | 2,140 chars |
| RetainDB | 0.401 | | | | 1,825 chars |

Read it honestly:

- **These five users are kind to everyone.** mem0 publishes 0.392 over all 30
  users and scores 0.520 here, so nothing in this table converts back to the
  public leaderboard.
- **Honcho and Hindsight are missing.** They hand the model 114,804 and 40,606
  characters per question, and grading that on one GPU crawled. Honcho leads the
  public table and very likely leads here too.
- **Tuning never touched these users.** Every change was tuned on two other
  users and graded once on these five, by a rule fixed before any result came in.
- **A tie is a tie.** Two near-identical runs of ours, judged independently,
  differed by 0.007.

Method, what each change was worth, and how to reproduce it: [bench/](bench/README.md).

## Install

```
/plugin marketplace add brainyboon/quipu
/plugin install quipu@brainyboon
```

- `python3` on your `PATH` is the only hard requirement; the core is standard
  library only.
- `numpy` is optional. Install it for that same `python3` and conversation
  search also ranks by meaning once the background job has run: it downloads a
  30 MB static embedding model once and builds the vectors, never inside a
  prompt. Until then, or without numpy, search uses keywords.
- The offline passes are model calls through `claude -p`, or any
  OpenAI-compatible endpoint you point `QUIPU_LLM_BASE_URL` at. They run at each
  session end, as new memories arrive, and in a daily curation sweep of up to a
  few dozen calls. On a Claude subscription that is plan usage; on an API key it
  is metered spend.

Memories live where Claude Code already keeps them:
`~/.claude/projects/<project>/memory/`.

Background jobs run from launchd or cron every ten minutes, never from a hook,
because Claude Code kills a hook's process group when the hook exits. Each job
decides for itself whether it is due. See [`launchd.plist.example`](launchd.plist.example).

## How it works

```
                  OFFLINE (a model is allowed)          EVERY PROMPT (no model)
session start ->  small always-loaded user profile
session end   ->  capture memories and fact notes       recall: up to 4 memories,
every 10 min  ->  enrich, embed, curate, verify                 at most 2 fact notes
```

- **Capture** ([`src/capture.py`](src/capture.py)). At session end a model reads
  the conversation and writes a memory only when something is worth keeping: a
  trap, a decision and its reason, a standing rule. Most sessions write nothing.
- **Facts that track change** ([`src/capture.py`](src/capture.py)). A durable
  fact about a person or organisation is written once and updated when it
  changes. The new value goes on top with the day it started, and every old
  value stays below with its dates. By default it keeps to work: where someone
  is based, their role, their standing preferences. The prompt keeps health,
  family and money out unless you set `QUIPU_FACTS_SCOPE=personal`.
- **Enrichment** ([`src/enrich.py`](src/enrich.py)). An offline pass writes the
  words someone would actually type to reach each memory, so "am I about to
  break production" finds a note about deploy safety.
- **Recall** ([`hooks/memory-recall.py`](hooks/memory-recall.py)). Searches the
  prompt, sharpened by the last few turns. Stays silent on "ok, go ahead", skips
  machine-generated turns, and never serves the same topic twice in a session.
- **Conversation search** ([`src/sessions.py`](src/sessions.py),
  [`src/embed.py`](src/embed.py)). Every past turn is indexed. Candidates come
  from keywords and from meaning, gathered wide, fused by rank and cut late.
- **Curation** ([`src/curate.py`](src/curate.py)). Contradictions are resolved by
  superseding, never deleting. Every decision is logged and one command undoes it.
- **Verification** ([`src/verify.py`](src/verify.py)). A memory is a set of
  claims. The paths, branches, pull requests and cloud resources it names get
  probed, and a memory that stopped being true is labelled when it surfaces. A
  URL that times out can never mark a memory false.
- **Profile** ([`src/profile.py`](src/profile.py)). A short `USER.md` with the
  rules that matter on most tasks, always in context. When it overflows, the
  weakest line moves into searchable memory instead of the write failing.
- **Skills** ([`src/skills.py`](src/skills.py)). A procedure that recurs across
  sessions is drafted as a skill and stays inert until you promote it.

## Commands

`src/mem` is the one entry point.

```
mem ask "deploy to production"   what recall would inject for that prompt
mem why "deploy to production"   the same, with scores and reasons
mem said "the pricing model"     search everything ever said, in every session
mem find "where did we land"     the same, ranked by meaning as well as words
mem facts                        every recorded fact, since when, what it replaced
mem who                          the always-loaded profile
mem stats | mem health           store size, enrichment coverage, index integrity
mem review                       what needs a human decision
mem verify                       check every claim against the real world
mem undo <file>                  revert a supersession or a stale flag
mem maintain                     when each background job last ran
```

## Configuration

| variable | default | what it does |
| --- | --- | --- |
| `QUIPU_ROOT` | `~/.claude/projects/<project>/memory` | the memory directory |
| `QUIPU_STATE` | `<project>/.memory-state` | index database and job stamps |
| `QUIPU_TRANSCRIPTS` | `~/.claude/projects/<project>` | where session transcripts are read from |
| `QUIPU_LOGS` | `~/.claude/quipu/logs` | logs |
| `QUIPU_LLM_BASE_URL` | unset, uses `claude -p` | OpenAI-compatible endpoint for the offline passes |
| `QUIPU_LLM_MODEL` | `qwen3.5-4b` | model name at that endpoint |
| `QUIPU_FACTS_SCOPE` | `work` | what fact notes may record; `personal` adds family, health and money |
| `QUIPU_EMBED_MODEL` | `potion-base-8M` | static embedding model for meaning search |
| `QUIPU_FIND_POOL` | `200` | candidates gathered before ranking |

## Design rules

1. **No model on the prompt path.** A model round trip from a hook took 8.6
   seconds. A full-text search takes milliseconds.
2. **Fetch wide, cut late.** On the benchmark, the sentence that answered the
   question was in the top 5 keyword hits 48% of the time and in the top 50 85%
   of the time.
3. **One store never fills the window.** Uncapped, fact notes took accuracy on
   conditional questions from 0.70 to 0.06.
4. **Never delete.** Supersede with a label, an audit log and an undo.
5. **A probe can prove life, never death.**

## License

MIT. Made at [Brainy](https://brainy.ink). The full story of how it was built and
measured: [brainy.ink/paper/claude-code-memory-system](https://brainy.ink/paper/claude-code-memory-system).

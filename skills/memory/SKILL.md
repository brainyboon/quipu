---
name: memory
description: Ask, inspect, correct and maintain the memory store. Use whenever the user asks what we know about something, why a memory surfaced, what memory got something wrong, whether a remembered fact is still true, or wants to fix, retire, or verify a memory. Also use for "what do you remember about X", "is that still true", "that's wrong, fix the memory", "what's stale", "run the memory sweep", "why did you recall that". Triggers on any doubt about whether stored context is current.
---

# Memory

The store is not a pile of notes. It is four moving parts, and knowing which one
is misbehaving is most of the work.

| Part | What it does | When it is the problem |
|---|---|---|
| recall | matches the prompt, injects up to 4 memories | the wrong thing surfaced, or nothing did |
| enrich | teaches the index the words the user actually types | a memory exists but never surfaces |
| curate | duplicates, supersession, contradictions | two memories disagree |
| verify | probes claims against the real world | a memory is confidently out of date |
| sessions | full-text search over every past conversation | they ask what was SAID, not what was learned |

Everything runs through one CLI. `$MEM` below is `${CLAUDE_PLUGIN_ROOT}/src/mem`.

## Two different questions, two different tools

**"What do we know about X"** is a memory question. Curated facts.

```bash
$MEM ask "the thing they asked about"     # what recall would inject
$MEM why "the thing they asked about"     # the same, with scores and reasons
```

**"What did we say/decide/discuss about X"** is a history question, and memory
cannot answer it. Nobody wrote down most of what was said.

```bash
$MEM said "the pricing model"             # every session that discussed it
$MEM said "X" --days 14                   # narrow to recent
$MEM history show <session-id> --grep X   # read the actual conversation back
```

Reach for `said` whenever the question contains a time reference ("last week",
"that thing we discussed", "did we ever", "you said earlier") or asks about a
decision rather than a fact. It is free, deterministic and ~15ms.

Read the hits before answering. Do not paraphrase a memory you have not opened
if the answer turns on a specific path, flag, version or number.

## When the user says a memory is WRONG

This is the highest-stakes path in the skill. Get it right in this order:

1. **Verify before believing either side.** The user's recollection and the
   memory can both be wrong, and they are often wrong about *different things*.
   A person can be right about one repo and wrong about the one the memory
   names. Run the real check: `gh repo view`, `ls`, `git ls-remote`, or the
   provider's own CLI. Never edit a memory on recollection alone, including
   your own.
2. **Correct the memory in place**, with the date and how it was verified. Do
   not delete the old claim silently; a memory that records what changed is
   worth more than one that only records what is.
3. **Search for siblings.** A wrong fact is rarely in one file. Grep the store
   for the same identifier: `grep -rl "<identifier>" <memory-root>`.
4. **Re-index**, so the correction is what surfaces next: `$MEM ask "<topic>"`
   to confirm.

## When a memory is stale rather than wrong

```bash
$MEM verify                  # probe every claim against the real world
$MEM claims <memory>         # what one memory asserts, and whether it holds
```

`fail` means a control-plane probe disagrees: a path, branch, PR, secret,
Doppler project or EC2 instance is not what the memory says. That is real.

`unknown` means the probe could not decide, and it is **not** evidence of
failure. A URL that times out from this machine proves only that this machine
cannot see it. Never report an `unknown` as a problem.

## When two memories disagree

```bash
$MEM review                  # what the curator could not decide alone
$MEM resolve <id> --note "…" # you acted on it
$MEM dismiss <id> --note "…" # it was not a real conflict
$MEM audit                   # what the curator changed, newest first
$MEM undo <memory>           # revert a supersession or a stale flag
```

Auto-supersession is deliberately conservative and it still gets things wrong.
Read `$MEM audit` before trusting that a memory was correctly retired. Four
failure modes have shown up in practice: retiring a standing rule with a passing
event, matching on a shared technology rather than a shared subject, letting an
`_index` pointer map retire a real memory, and retiring a list of six open items
because one of them got resolved.

## What belongs in memory

A memory is a fact that stays true. Not:

- **mutable state**: "PR #96 is open" is false minutes later. Record what was
  learned from the PR, never its status.
- **anything the code or git history already says**: that is not memory, it is
  a stale copy of the repo.
- **session narrative**: what was done today belongs in the transcript.

Do: traps that cost real time, root causes, decisions and their reasons,
standing preferences, and durable pointers to where a thing lives.

## Maintenance

```bash
$MEM health     # is the index file still fully loading?
$MEM stats      # size, enrichment coverage, flags
$MEM sweep      # full control plane now (normally runs at session end)
$MEM log 30     # what fired on recent prompts, and what it pulled
```

`health` matters more than it looks. The auto-memory loader reads only the first
200 lines or 25,000 bytes of the index file, whichever comes first, and the
warning is easy to miss. Everything past that point silently never loads.

## If recall is noisy or silent

One dial: `DEFAULT_FLOOR` in `src/recall.py`. Raise it to fire less, lower it to
fire more. Check `$MEM log` first to see what it actually did before changing
anything: silence on a conversational prompt is correct behaviour, not a bug.

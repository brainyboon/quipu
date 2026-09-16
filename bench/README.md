# Benchmark: MemConflict

How the numbers in the main README were produced, and how to reproduce them.

## The harness

[hermes-memconflict](https://github.com/EngTurtle/hermes-memconflict) runs
memory providers against MemConflict: 30 simulated users whose facts change over
time, with a wrong answer scored at minus one. The published comparison of
Honcho, mem0, Hindsight, Supermemory, RetainDB, OpenViking and Mnemosyne came
out of it. [`memconflict_adapter.py`](memconflict_adapter.py) drops in as one
more provider.

## Keeping it fair

- **One judge for every row.** The published table was judged by gemma-4-12b.
  Our GPU runs qwen3.5-4b, and the harness documents that scores from those two
  judges do not compare. So every competitor's published answers were re-judged
  with qwen3.5-4b alongside ours.
- **The same internal model for everyone.** The harness gives every provider the
  same model for its own extraction work. `QUIPU_LLM_BASE_URL` points capture and
  enrichment at that shared endpoint.
- **Personal facts on.** MemConflict asks about family, health and money, so the
  adapter runs capture with `QUIPU_FACTS_SCOPE=personal`. The plugin's default
  scope records work facts only.
- **Pinned answer decoding.** Every answer, ours and theirs, was generated under
  `benchmark/docker/answer_env.sh`.
- **A control.** On one user, mem0's published answers scored 0.4434 and mem0
  re-run from scratch on our hardware scored 0.4654, so the box does not flatter
  anyone.
- **No tuning on the reported users.** Every change was tuned on users 5 and 6
  and graded once on users 0 to 4. The version reported was picked by a rule
  written down before any result came in: best score on the tuning users.

## Hardware

The model checkpoint, `AxionML/Qwen3.5-4B-NVFP4`, needs a Blackwell GPU. One AWS
`g7.2xlarge` (RTX PRO 4500 Blackwell, 32 GiB) runs `vllm-gen` at 0.85 of memory
and `vllm-embed` at 0.07.

## Run

```bash
git clone --recurse-submodules https://github.com/EngTurtle/hermes-memconflict
cd hermes-memconflict
mkdir -p quipu/Results quipu/Scores
cp /path/to/quipu/bench/memconflict_adapter.py quipu/
cp -r /path/to/quipu/src quipu/src
docker compose up -d vllm-gen vllm-embed

export PYTHONPATH=$PWD/benchmark BENCH_ROOT=$PWD BENCH_PYTHON=python3
source benchmark/docker/answer_env.sh
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY OPENAI_MODEL=qwen3.5-4b
export QUIPU_LLM_BASE_URL=http://localhost:8000/v1 QUIPU_LLM_MODEL=qwen3.5-4b QUIPU_SRC=$PWD/quipu/src

# the reported configuration
export QUIPU_ARM=both QUIPU_RETRIEVAL=hybrid QUIPU_REAL_DATES=1 QUIPU_FACTS=1 QUIPU_MAX_DISTILLED=2 \
       QUIPU_FIND_W_BM25=0.5 QUIPU_FIND_W_SEM=2.0 QUIPU_FIND_W_COV=0.5

# 1. retrievals, with throwaway answers (users 0 to 4)
bench_answer_env
python3 quipu/memconflict_adapter.py --input external/MemConflict/Data/Step4_4.jsonl \
  --output-jsonl quipu/Results/run_raw.jsonl --output-json quipu/Results/run_raw.json \
  --top-k 5 --start-idx 0 --end-idx 5

# 2. the real answers, regenerated from those frozen retrievals
python3 benchmark/replay_answers.py --input_file quipu/Results/run_raw.jsonl \
  --output_file quipu/Results/run.jsonl --workers 16

# 3. judge and summarise
export RESULTS_FILE=$PWD/quipu/Results/run.jsonl SCORES_FILE=$PWD/quipu/Scores/run_scores.jsonl \
       CHECKPOINT=$PWD/quipu/Scores/run_checkpoint.jsonl SUMMARY_FILE=$PWD/quipu/Scores/summary_run.json
run_score $PWD/quipu run && run_summarize $PWD/quipu run
```

Run one user first (`--end-idx 1`). It catches a broken setup in minutes and
tells you how long the full run will take.

## Results

Users 0 to 4, 616 questions. "Score" is the harness's macro answer accuracy,
"per question" is micro accuracy, and "evidence in top 5" is SEH@5.

| system | score | per question | evidence in top 5 | change | never changes | conditional |
| --- | --- | --- | --- | --- | --- | --- |
| mem0 | 0.520 | 0.477 | 0.694 | 0.445 | 0.275 | 0.840 |
| **quipu** | **0.520** | **0.480** | 0.640 | **0.451** | **0.308** | 0.800 |
| Supermemory | 0.483 | 0.429 | 0.611 | 0.392 | 0.258 | 0.800 |
| quipu, search only | 0.447 | | 0.565 | 0.289 | 0.200 | 0.853 |
| quipu, first run | 0.409 | | 0.513 | 0.226 | 0.267 | 0.733 |
| RetainDB | 0.401 | | 0.588 | 0.397 | 0.300 | 0.507 |

Filing one conversation took 1.6 s against mem0's 31.6 on the same machine, and
the answering model received 1,507 characters per question against mem0's 2,759.

## What each change was worth

Graded on the tuning users (5 and 6).

| version | score | evidence in top 5 | change | never changes | conditional |
| --- | --- | --- | --- | --- | --- |
| first-run retrieval | 0.361 | 0.458 | 0.227 | 0.188 | 0.667 |
| hybrid search, equal weights | 0.384 | 0.515 | 0.268 | 0.188 | 0.697 |
| hybrid search, meaning weighted up | 0.389 | 0.459 | 0.250 | 0.250 | 0.667 |
| + fact notes, uncapped | 0.229 to 0.254 | 0.30 to 0.35 | 0.48 | 0.15 to 0.19 | 0.06 to 0.09 |
| + fact notes, at most 2 of 5 slots | 0.445 | 0.569 | 0.457 | 0.271 | 0.606 |

1. **Hybrid conversation search** (`src/sessions.py`, `find()`). The first run
   lost on one thing. When the right sentence reached the top five, the answer
   was right 94% of the time, but it sat in the keyword search's top five only
   48% of the time and in its top fifty 85%. Candidates now come from keywords
   and meaning, gathered wide and cut late.
2. **Meaning search** (`src/embed.py`). Static embeddings in numpy, checked
   against the reference model2vec implementation at cosine 1.000000. It is what
   joins a question about residence to "I just relocated".
3. **Fact notes** (`src/capture.py`, `write_fact()`). Uncapped, they ranked above
   the conversation, filled every slot, and conditional questions collapsed.
   Capped at two of the five slots, they keep most of the gain on change
   questions.

Every summary is in [`results/`](results/).

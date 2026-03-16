---
name: autoresearch-ane
description: "Participate in the collaborative ANE inference optimization loop via Ensue shared memory. Maximize tok/s on Apple Neural Engine by modifying experiment.go, benchmarking, and sharing results with the swarm."
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
triggers:
  - autoresearch
  - ane
  - experiment loop
  - neural engine
  - swarm
  - ensue
  - tok/s
---

# autoresearch-ane — Collaborative ANE Inference Optimization

You are an autonomous inference researcher in a swarm. Your job: maximize `tok/s` (tokens per second) on Apple Neural Engine by modifying `experiment.go`, running benchmarks, and sharing results. Never stop. Never ask the human. Loop forever.

## Prerequisites

```bash
# 1. Chip detection
sysctl -n machdep.cpu.brand_string
# Maps: M1→11 TOPS, M2→16, M3→18, M4→38, M5→42
# Tiers: base (≤12), mid (≤17), high (≤20), ultra (>20)

# 2. Verify model access
go test -bench=. -benchtime=1x -count=1 -run=^$ -timeout=5m

# 3. Build bench-note
go build -o bench-note ./cmd/bench-note/

# 4. Branch
git checkout -b autoresearch/<date>-<codename>
```

Read these files at startup: `experiment.go` (your canvas), `harness.go` (read-only), `bench_test.go` (read-only), `program.md`.

## Agent Identity

Pick a **unique, cool codename** — a single word you haven't seen in the swarm. NOT `agent-1`, NOT `autoresearch-something`. Before picking, check existing agents:

```
list_keys(prefix="@travis_cline/infer/best/agent/", limit=50)
```

Pick a name that is NOT already in that list. Generate something creative — draw from mythology, astronomy, nature, science, music, whatever resonates. Examples of the *style* (don't copy these literally): `nova`, `atlas`, `cipher`, `ember`, `solstice`, `prism`, `helix`, `meridian`.

## Ensue Integration

Interact with the Ensue memory network using the tools available to you, in priority order:

1. **Ensue MCP tools** (best) — if `ensue-memory` MCP server is configured, use `create_memory`, `get_memory`, `search_memories`, `list_keys`, `update_memory`, `discover_memories` directly as tool calls.
2. **`ensue-api.sh`** (fallback) — if the ensue-skill plugin is installed: `ensue-api.sh <method> '<json_args>'`
3. **`curl`** (last resort) — direct JSON-RPC to `https://api.ensue-network.ai/`

Authentication: `ENSUE_API_KEY` env var or `.autoresearch-key` file.

All Ensue errors are non-blocking — log and continue solo if network fails.

## Shared Namespace

All keys under `@travis_cline/<workload>/` (default workload: `infer`):

```
@travis_cline/infer/results/<agent>--<slug>--<hash>     completed experiments
@travis_cline/infer/claims/<agent>--<slug>--<hash>      active work (15-min TTL)
@travis_cline/infer/hypotheses/<agent>--<slug>--<hash>  untested ideas
@travis_cline/infer/insights/<agent>--<slug>--<hash>    collective learnings
@travis_cline/infer/best/experiment_go                  global best source
@travis_cline/infer/best/metadata                       global best stats
@travis_cline/infer/best/tier/<tier>/experiment_go      per-tier best source
@travis_cline/infer/best/tier/<tier>/metadata           per-tier best stats
```

**Key format**: `<agent>--<slug>--<6char_hash>`. Example: `nova--increase-context-to-2048--a7f3b2`

To construct a key:
1. Agent slug: lowercase, replace non-alphanumeric with `-`, strip leading/trailing `-`, truncate to 20 chars
2. Description slug: same, truncate to 40 chars
3. Hash: first 6 hex chars of SHA256 of lowercase+trimmed description
4. Join: `<agent_slug>--<desc_slug>--<hash>`

Or use the Go helper: `coordinator.ExperimentKey("<YOUR_CODENAME>", "description")`

## Chip Tiers

| Tier  | ANE TOPS | Chip Family        |
|-------|----------|--------------------|
| base  | ≤12      | M1 (11 TOPS)       |
| mid   | ≤17      | M2 (15.8 TOPS)     |
| high  | ≤20      | M3 (18 TOPS)       |
| ultra | >20      | M4 (38), M5 (42)   |

Detect: `sysctl -n machdep.cpu.brand_string`. Include `chip_name`, `chip_tier`, `ane_tops` in every result.

## What You're Modifying

### Tier 1 — experiment.go (fast iteration)

| Constant | What it does | Try |
|----------|-------------|-----|
| `DefaultModel` | model to benchmark | different quantizations, sizes |
| `DefaultPrompt` | prompt text | short vs long |
| `GenerateTokens` | tokens to generate | 50, 100, 200, 500 |
| `Temperature` | sampling temp | 0.0 (greedy) vs 0.6 vs 1.0 |
| `CacheType` | KV cache strategy | "default", "inplace", "rotating", "prealloc" |
| `ANEDecodePlaneMode` | ANE mode | "qwen35", "off" |
| `WarmupEnabled` | warmup before bench | true vs false |

### Tier 2 — cmd/mlx-lm-generate-ane/ (deeper changes)

- `generate.go` — token generation pipeline, sampling
- `kvcache.go` — cache config, quantization, pre-allocation
- `ane.go` — ANE decode plane integration
- `stats.go` — statistics reporting

### Read-only (DO NOT MODIFY)

- `harness.go` — evaluation harness
- `bench_test.go` — benchmark harness

## Safety Rules

**CRITICAL — follow these rules on every iteration, no exceptions:**

1. **Never modify `harness.go` or `bench_test.go`** — these are the ground truth measurement
2. **Best-update safety** — before writing to `best/`, ALWAYS:
   - `get_memory` the current best metadata
   - Verify your tok/s is strictly higher than current best
   - **Reject tok/s ≤ 0** (crash or bug)
   - **Reject improvement > 100%** (tok/s > current_best × 2 — measurement error)
   - Re-read best metadata immediately before writing (minimize race window)
   - Preserve `previous_best_*` fields so the old best can be recovered
   - Only `keep` results may update `best/` — never discards or crashes
3. **Claim TTL** — claims expire after **15 minutes**. When checking claims, treat any claim with `claimed_at` older than 15 min as expired (ignore it)
4. **Search before write** — always `search_memories` or `list_keys` before creating, to avoid duplicates
5. **Always `embed: true`** — on both `create_memory` and `update_memory`, so semantic search works
6. **Ensue errors are non-blocking** — log and continue solo if any Ensue call fails

## The Loop

Run forever. Each iteration follows 8 steps:

### 1. THINK

Read the swarm state before picking an experiment:

```
# Using Ensue MCP tools:
search_memories(query="experiment result tok/s", limit=30, prefix="@travis_cline/infer/results/")
search_memories(query="insight", limit=10, prefix="@travis_cline/infer/insights/")
search_memories(query="hypothesis suggestion", limit=10, prefix="@travis_cline/infer/hypotheses/")
list_keys(prefix="@travis_cline/infer/claims/", limit=20)
get_memory(key_names=["@travis_cline/infer/best/metadata"])

# Every 5 runs: check if someone beat you
get_memory(key_names=["@travis_cline/infer/best/tier/<your_tier>/metadata"])
get_memory(key_names=["@travis_cline/infer/best/tier/<your_tier>/experiment_go"])
```

Reason about patterns. Connect findings from different agents. If analysis reveals ideas you won't pursue, publish them immediately as hypotheses.

### 2. CLAIM

Before editing, claim to prevent duplicate work:

```
# 1. Check if result exists
get_memory(key_names=["@travis_cline/infer/results/<key>"])

# 2. Check for semantically similar active claims
search_memories(query="<your description>", limit=5, prefix="@travis_cline/infer/claims/")
# Skip if any result has score > 0.92 AND claimed_at < 15 min ago
# Claims older than 15 min are EXPIRED — ignore them

# 3. Write claim
create_memory(items=[{
  "key_name": "@travis_cline/infer/claims/<key>",
  "description": "[autoresearch] Claim: <description>",
  "value": "<base64 JSON: agent_id, description, claimed_at, chip_name, chip_tier>",
  "base64": true, "embed": true, "embed_source": "description"
}])

# 4. Wait 2 seconds, re-read to verify you own it
get_memory(key_names=["@travis_cline/infer/claims/<key>"])
```

If claim fails after 5 attempts, just run something — a rare duplicate beats doing nothing.

### 3. HACK

Edit `experiment.go` (Tier 1) or files in `cmd/mlx-lm-generate-ane/` (Tier 2).

### 4. COMMIT

```bash
go test -c -o /dev/null .   # verify it compiles
git add -A && git commit -m "<param> <old> → <new>"
```

### 5. RUN

```bash
./bench-note run --benchtime=5x --count=6
```

This runs benchmarks, attaches results as a git note to HEAD, and auto-compares against the nearest ancestor with a bench note.

### 6. RECORD

Key metrics from bench-note output:
- `BenchmarkGenerate` `tok/s` — **primary optimization target**
- `BenchmarkGenerate` `prefill_ms` — prompt processing time
- `BenchmarkGenerate` `peak_mem_gb` — peak memory
- `BenchmarkDecode` `decode_tok/s` — decode-only throughput

Append to `results.tsv` (tab-separated, never commit):

```
commit	tok_per_s	decode_tok_per_s	prefill_ms	status	description
a1b2c3d	12.345	15.678	234.5	keep	baseline
```

### 7. DECIDE

- **tok/s increased** with `p < 0.05`: status=`keep`, keep the commit.
- **tok/s equal or worse**: status=`discard`, reset: `git reset --hard HEAD~1`.
- **Crash**: status=`crash`, reset: `git reset --hard HEAD~1`.

**Sanity checks** (reject as invalid):
- `tok/s <= 0` — crash or bug
- Improvement > 100% in a single step — measurement error

### 8. PUBLISH

All three are **mandatory every iteration**, no exceptions. Batch them into a single `create_memory` call:

```
create_memory(items=[
  {
    "key_name": "@travis_cline/infer/results/<result_key>",
    "description": "[autoresearch] [<agent> <STATUS>] <tok/s> tok/s | <description>",
    "value": "<base64 result JSON>",
    "base64": true, "embed": true, "embed_source": "description"
  },
  {
    "key_name": "@travis_cline/infer/insights/<insight_key>",
    "description": "[autoresearch] Insight: <what you learned>",
    "value": "<base64 insight JSON>",
    "base64": true, "embed": true, "embed_source": "description"
  },
  {
    "key_name": "@travis_cline/infer/hypotheses/<hypothesis_key>",
    "description": "[autoresearch] Hypothesis: <title>",
    "value": "<base64 hypothesis JSON>",
    "base64": true, "embed": true, "embed_source": "description"
  }
])
```

**Result JSON schema:**
```json
{
  "agent_id": "<YOUR_CODENAME>",
  "tokens_per_sec": 12.345,
  "decode_tok_per_sec": 15.678,
  "prefill_ms": 234.5,
  "peak_mem_gb": 8.2,
  "chip_name": "Apple M4 Max",
  "chip_tier": "ultra",
  "ane_tops": 38,
  "status": "keep",
  "commit": "a1b2c3d",
  "description": "cache default → inplace",
  "experiment_go": "<full source of experiment.go>",
  "completed_at": "2026-03-15T12:00:00Z",
  "delta_vs_best": 1.23
}
```

**Insight JSON**: `agent_id`, `chip_name`, `chip_tier`, `insight` (explain WHY — not just what), `evidence_keys` (result keys), `posted_at`.

**Hypothesis JSON**: `agent_id`, `chip_name`, `chip_tier`, `title`, `hypothesis`, `suggested_config`, `evidence_keys`, `priority` (1-5), `created_at`.

Description format: `<param> <old_value> → <new_value>` (e.g. `cache default → inplace`).

## Updating Global Best

Only `keep` results with tok/s higher than current best:

```
# 1. Read current best
get_memory(key_names=["@travis_cline/infer/best/metadata"])

# 2. Sanity checks
#    - tok/s <= 0: reject
#    - tok/s > current_best * 2: reject (>100% improvement)
#    - tok/s <= current_best: skip

# 3. Re-read (minimize race window)
get_memory(key_names=["@travis_cline/infer/best/metadata"])

# 4. Update code
update_memory(key_name="@travis_cline/infer/best/experiment_go",
              value="<base64 experiment.go source>", base64=true, embed=true)

# 5. Update metadata (preserve previous best info)
update_memory(key_name="@travis_cline/infer/best/metadata",
              value="<base64 JSON with previous_best_* fields>", base64=true, embed=true)
```

Also update per-tier best (`best/tier/<tier>/metadata` and `best/tier/<tier>/experiment_go`) and per-agent best (`best/agent/<name>`) using the same pattern.

## Adopting a Better Config

When another agent's config is better than yours:

```bash
# Pull best for your tier
get_memory(key_names=["@travis_cline/infer/best/tier/<tier>/experiment_go"])
# Write to experiment.go, commit:
git add experiment.go && git commit -m "adopt global best (tok/s=X from Y)"
```

## Go Helpers

The `coordinator` package provides chip detection and key generation:

```go
import "github.com/tmc/mlx-go-ane/coordinator"

chip := coordinator.DetectChip()     // chip.Name, chip.Tier, chip.TOPS
key := coordinator.ExperimentKey("<YOUR_CODENAME>", "cache default → inplace")
pfx := coordinator.Pfx("infer", "results", key)
apiKey := coordinator.GetAPIKey()
```

These are helpers for deterministic operations. All Ensue network interaction uses MCP tools or CLI.

## Git Workflow

- Own branch: `autoresearch/<date>-<codename>`
- Atomic commits per experiment: `param old → new`
- On discard/crash: `git reset --hard HEAD~1`
- Never push to another agent's branch
- Never commit `results.tsv`
- Adopting best: `adopt global best (tok/s=X from Y)`

## Never Stop

Once the loop begins, do NOT pause to ask the human. Do NOT ask "should I keep going?". The human may be asleep. You are autonomous. If you run out of ideas: re-read code, combine near-misses, try radical changes, check swarm hypotheses. Loop until manually stopped.

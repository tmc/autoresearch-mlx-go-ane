# mlx-go-ane autoresearch

This is an experiment to have the LLM do its own ML research, running inference optimization experiments on Apple Neural Engine.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `experiment.go` — primary experiment file. Inference config, sampling params, model selection.
   - `harness.go` — evaluation harness, model loading, generation timing. Do not modify.
   - `bench_test.go` — Go benchmarks for measuring generation throughput. Do not modify.
   - `cmd/mlx-ane-generate/` — the ANE-focused generation CLI. Editable with care.
4. **Verify model access**: Ensure the default model (`mlx-community/Qwen2.5-3B-Instruct-4bit`) is cached. Run `go test -bench=. -benchtime=1x -count=1 -run=^$ -timeout=5m` to confirm.
5. **Install benchstat**: `go install golang.org/x/perf/cmd/benchstat@latest`
6. **Build bench-note**: `go build -o bench-note ./cmd/bench-note/`
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment generates tokens on Apple Neural Engine. The optimization target is **tok/s** (higher is better) and ANE decode plane utilization.

### Editable files

You have two tiers of editable files:

**Tier 1 — Primary experiment surface** (`experiment.go`):
- Model selection, prompt, token count, sampling parameters, cache type, ANE mode.
- This is the fastest iteration loop — change constants, rebuild, benchmark.

**Tier 2 — Generation internals** (`cmd/mlx-ane-generate/` package):
- Generation pipeline, KV cache configuration, ANE decode plane integration.
- More impactful but riskier. Changes here can affect correctness, so verify carefully.

### Read-only files

- `harness.go` — evaluation harness (`setupEngine`, `generateN`). The ground truth measurement.
- `bench_test.go` — benchmark harness.

**The goal is simple: get the highest tok/s (tokens per second).** Everything in the editable files is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the benchmarks as-is.

## Benchmarking with bench-note

Use `bench-note` (`cmd/bench-note`) to run benchmarks, attach results to git commits as notes, and compare across commits. Results are stored in `refs/notes/benchmarks` using txtar format.

**Build bench-note** (once per session):
```bash
go build -o bench-note ./cmd/bench-note/
```

**Run benchmarks and attach to HEAD**:
```bash
./bench-note run --benchtime=1x --count=6
```

This runs `go test -bench`, attaches the output as a git note to HEAD, and automatically runs benchstat against the nearest ancestor that has a bench note.

**Attach existing output** (e.g. from a `tee` file):
```bash
./bench-note run --from-file=bench_after.txt --benchtime=1x --count=6
```

**View results**:
```bash
./bench-note show           # full txtar note for HEAD
./bench-note show abc1234   # for a specific commit
./bench-note raw            # just the raw go test output (for piping)
```

**Compare two commits**:
```bash
./bench-note compare abc1234 def5678
```

**List all commits with bench notes**:
```bash
./bench-note history            # detailed
./bench-note history --oneline  # compact
```

### Key metrics

- `BenchmarkGenerate` `tok/s` — **this is the primary metric you are optimizing** (tokens per second, higher is better)
- `BenchmarkGenerate` `decode_tok/s` — decode-only throughput (excluding prefill)
- `BenchmarkGenerate` `prefill_ms` — time to process the prompt and produce first token
- `BenchmarkGenerate` `gen_ms` — total generation wall time
- `BenchmarkGenerate` `peak_mem_gb` — peak GPU/Metal memory usage
- `BenchmarkPrefill` `prompt_tok/s` — prefill throughput

A change is worth keeping if `tok/s` increased with `p < 0.05`. If benchstat shows `~` (no significant difference), the change has no effect — discard it.

The `-count 6` flag runs each benchmark 6 independent times for benchstat's Welch t-test. With `-benchtime=1x`, each sample is a single generation — fast (~3s total) and statistically independent. Use `-count 10` for higher confidence on close calls.

## Logging results

The primary benchmark record lives in git notes (`refs/notes/benchmarks`), attached by `bench-note run`. Use `bench-note history` to review the full history with raw output and benchstat deltas.

Additionally, log a summary to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	tok_per_s	decode_tok_per_s	prefill_ms	status	description
```

1. git commit hash (short, 7 chars)
2. tok/s achieved (e.g. 12.345) — use 0.000 for crashes
3. decode_tok/s (e.g. 15.678) — use 0.000 for crashes
4. prefill_ms (e.g. 234.5) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	tok_per_s	decode_tok_per_s	prefill_ms	status	description
a1b2c3d	12.345	15.678	234.5	keep	baseline
b2c3d4e	13.456	16.789	220.1	keep	switch to inplace cache
c3d4e5f	12.100	15.200	245.0	discard	increase generate tokens to 200
d4e5f6g	0.000	0.000	0.0	crash	bad ANE mode string
```

NOTE: do not commit `results.tsv` — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

1. Edit files with an experimental idea (see editable files above).
2. Verify it compiles: `go test -c -o /dev/null .`
3. git commit: `git add -A && git commit -m "<description>"`
4. Run benchmarks and attach as git note: `./bench-note run --benchtime=1x --count=6`
   - This runs benchmarks, attaches results to HEAD, and auto-compares against the nearest ancestor with a bench note.
5. Review the benchstat delta: `./bench-note show`
6. If `tok/s` improved (increased) with statistical significance:
   - You "advance" the branch, keeping the git commit.
   - Log results to `results.tsv`.
7. If `tok/s` is equal or worse:
   - `git reset --hard HEAD~1` to revert.
   - Log results to `results.tsv` with status `discard`.
8. If the build or run crashed:
   - If it's something easy to fix (typo, bad constant), fix and re-run.
   - If the idea is fundamentally broken, `git reset --hard HEAD~1`, log `crash`, move on.
9. Go to step 1.

**Timeout**: Each benchmark run should take ~2-5 minutes. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: Use your judgment. If it's a dumb mistake (e.g. a constant out of range), fix it. If the idea itself is broken (e.g. unsupported cache type), skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the editable files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## What you can change

### Tier 1: experiment.go (fast iteration)

#### Constants
- `DefaultModel` — model to benchmark (try different quantizations, model sizes)
- `DefaultPrompt` — prompt text (try short vs long prompts, different content)
- `GenerateTokens` — number of tokens to generate (affects throughput measurement)
- `Temperature` — sampling temperature (0.0 = greedy, affects decode speed)
- `TopP`, `MinP`, `TopK` — sampling parameters (simpler sampling = faster decode)
- `WarmupEnabled` — whether to warm up before benchmarks
- `CacheType` — KV cache strategy ("default", "inplace", "rotating", "prealloc")
- `ANEDecodePlaneMode` — ANE mode ("qwen35", "off")
- `UseChatTemplate` — whether to use chat template (affects prompt token count)
- `Seed` — random seed

### Tier 2: cmd/mlx-ane-generate/ (deeper changes)

- Generation pipeline, KV cache, ANE decode plane integration.
- **Don't break the harness API**: The benchmark tests call `setupEngine`, `generateN`, `encodePrompt`, `warmup` — these must keep their signatures.
- **Don't modify harness.go or bench_test.go** — these are the ground truth.
- **Test carefully** — a wrong configuration can silently produce invalid results.

## ANE-specific notes

- ANE decode plane mode "qwen35" offloads per-token decode to the Neural Engine. Usually faster for small models.
- MLX compilation is disabled when ANE decode plane is active — they are incompatible.
- The warmup pass triggers lazy compilation and ANE decode plane pre-warming.
- Peak memory varies by model size and cache type.

## Parameter space guidance

Good starting experiments (Tier 1 — fast):
1. **Model variants**: Try different quantizations (4bit, 8bit) and model sizes
2. **Cache type**: Try "default", "inplace", "rotating", "prealloc"
3. **Prompt length**: Short (10 tokens) vs long (500+ tokens) prompts
4. **Generate tokens**: 50, 100, 200, 500 — see how throughput scales
5. **Temperature**: 0.0 (greedy) vs 0.6 vs 1.0 — sampling overhead
6. **ANE mode**: "qwen35" vs "off" — measure ANE benefit
7. **Warmup**: enabled vs disabled — measure cold start impact
8. **Chat template**: on vs off — measure template overhead

Deeper experiments (Tier 2 — more impactful, more risk):
1. **Sampling strategy**: Try "lazy" vs other strategies
2. **Strided cache**: Enable/disable UseStridedCache
3. **Eager prefill**: Enable EagerPrefill option
4. **Cache quantization**: Set kv-bits for quantized cache
5. **Rotating cache size**: Tune max-kv-size for rotating cache
6. **Compile mode**: Experiment with compile mode when ANE is off

## Collaborative mode

When running in collab mode, multiple agents on different Macs share results through the Ensue memory network. See `.skills/autoresearch/SKILL.md` for the full protocol (THINK/CLAIM/HACK/RUN/PUBLISH loop, Ensue tool usage, key schemas, safety rules). See `collab.md` for a quick reference.

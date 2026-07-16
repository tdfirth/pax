# Pax — Implementation Plan

Companion to `docs/api-contract.md` (**FROZEN v1**). That contract is the single
source of truth for every signature and semantic rule; this plan is the work
breakdown and the parallelization strategy.

## Sequencing at a glance

```
Phase 0  Setup            ── blocks everything (mechanical, no design)
Phase 1  Core             ── CRITICAL PATH, built together, not parallelized
                              (also stubs every Phase-2 module so the package
                               is importable before fan-out)
Phase 2  Fan-out          ── parallel subagents, disjoint files:
   Wave 1 (needs core only):  A layers · B combinators · C path-utils · E docs-draft
   Wave 2 (needs A+B):        D integration/examples · E docs-final
Phase 3  Integrate & gate ── merge, full suite green, ruff+ty clean
```

The core is the critical path because everything reads its bound-state /
write-collection semantics. We do **not** parallelize it. Once it lands and is
tested, Phase 2 tasks touch **disjoint files** and run concurrently.

## Repo layout (target)

```
src/pax/
  __init__.py        # complete public exports — written by CORE, edited by no one else
  module.py          # Module: __new__/__setattr__/__getattr__/forward/_bind/_collect/key   [CORE]
  namespaces.py      # namespace(), Static, buffer, flags, process registry                 [CORE]
  random.py          # seed() context, thread-local init-key source                          [CORE]
  layers/            # one file per layer                                                     [TASK A]
    __init__.py      #   (stubbed by CORE so `from .layers import *` works)
    linear.py  embedding.py  norm.py  attention.py
  combinators.py     # sequential, repeat, parallel                                           [TASK B]
  paths.py           # freeze, select, glob matcher                                           [TASK C]
tests/
  test_core.py  test_namespaces.py  test_random.py                                            [CORE]
  test_layers.py                                                                              [TASK A]
  test_combinators.py                                                                         [TASK B]
  test_paths.py                                                                               [TASK C]
  test_transforms.py  test_integration.py                                                     [TASK D]
examples/
  mlp.py  batchnorm.py  transformer.py                                                        [TASK D]
docs/
  api-contract.md  implementation-plan.md
  guide/  quickstart.md  namespaces.md  transforms.md  combinators.md  training.md            [TASK E]
```

**Parallel-safety rule:** every Phase-2 task owns a disjoint set of files. The
only file touched by multiple hands is `src/pax/__init__.py`, so **CORE writes it
in full** (referencing pre-stubbed modules) and **no Phase-2 task edits it**. Each
implementation subagent runs in its own git worktree, branched off the committed
core; Phase 3 merges them.

---

## Phase 0 — Setup  *(I do this now; no decisions)*

**Deliverables**
- `pyproject.toml`: runtime dep `jax`; dev deps `optax`, `pytest`, `ty`, `ruff`.
  `[tool.ruff]` (lint + format), `[tool.ty]`, `[tool.pytest.ini_options]`.
  `src/` layout, `requires-python = ">=3.13"`.
- Delete `main.py`. `src/pax/__init__.py` placeholder. `README.md` stub.
- `uv sync`; confirm `pytest`, `ruff check`, `ty check` all run on an empty suite.

**Acceptance:** `uv run pytest` (0 tests, exit 0), `uv run ruff check`, `uv run ty
check` all pass on the skeleton.

---

## Phase 1 — Core  *(built together; the hard part)*

Implements §2–§6 of the contract. **Single most important deliverable.**

**Scope**
- `namespaces.py`: `namespace(name, *, static, scoped)`, the process-global
  namespace registry, `Static` pytree node (`register_pytree_node_class`,
  contents in `aux_data`, `_hashable` conversion), built-ins `buffer`, `flags`,
  the `Tagged` wrapper.
- `random.py`: `seed(n)` context manager over a thread-local key; the
  `_next_module_key()` source used by `Module.__new__`.
- `module.py`: `__new__` bookkeeping (§5.1, incl. the `_bound`/`_writes`
  **stacks**), `__setattr__` classification (§5.2), `__getattr__` resolution
  (§3 rules 1–3, **no fallthrough**), `_bind` (§5.3), `_collect`/`_unbind`
  (§5.4), `forward` (§4), `state(key=None)` (§4), `key()` (§6).
- `__init__.py`: full public exports. Stub `layers/` (`__init__.py` + empty layer
  modules), `combinators.py`, `paths.py` with signature-correct
  `raise NotImplementedError` stubs so the package imports and Phase-2 agents
  start from green.

**Tests (`test_core.py`, `test_namespaces.py`, `test_random.py`)**
- Roundtrip: build a 2-level module, `state()` layout matches §1 exactly
  (namespaces present/absent, scoped nesting, `Static` for flags).
- Dual-mode access: `self.W` reads init value unbound, bound value during
  `forward`; write-collection returns updated buffers in `new_state`.
- `forward` purity: module holds no arrays between calls; two sequential
  `forward`s don't leak state; re-entrant `forward` (nested) via the stack.
- Static-under-jit: `jax.jit(model.forward)` compiles one program per flag
  combination; `if self.flags.training` is a concrete branch (no tracer error).
- Transforms on `forward`: `jit`, `vmap(in_axes=(None,0))`, `scan`, `grad`
  (params-only via `{**state,'params':p}`), `remat` — all pass with a
  hand-written Linear (no dependency on Task A).
- PRNG: `with pax.seed(0)` reproducible; different seeds differ; `state(key=...)`
  re-materializes.
- D3: forward-time write to an unregistered name raises.

**Acceptance:** all core tests green; `ruff`+`ty` clean; package importable with
Phase-2 stubs present. **This is the gate that unblocks fan-out.**

---

## Phase 2 — Parallel tasks

Each is a self-contained subagent brief. All depend **only** on the FROZEN
contract + committed core. Each: full type hints (`ty` clean), `ruff` clean,
its own tests green, no new deps.

### Task A — Standard layers  *(→ `layers/`, `tests/test_layers.py`)*
Implement per contract §3 conventions:
- **Linear** — `W`, `b` (params). `x @ W + b`.
- **Embedding** — `E` (params, `(vocab, d)`). `E[x]`.
- **LayerNorm** — `g`, `b` (params); normalize over last axis; no buffers.
- **BatchNorm** — `g`, `b` (params); `running_mean`, `running_var` (`buffer`);
  `if self.flags.training` train/eval branch with buffer writes. *Richest test:
  params + buffers + static flag + write-collection.*
- **Attention** — multi-head; `Wq/Wk/Wv/Wo` (params); `k`, `v` in a `cache`
  namespace; `if self.flags.cache` branch. Exercises a user traced-scoped
  namespace + static flag.

**Tests:** shape/correctness per layer; BatchNorm buffers actually update in
train and are read (not updated) in eval; Attention cache concatenation grows
under `flags.cache`; each layer works under `jax.jit(forward)`.

### Task B — Combinators  *(→ `combinators.py`, `tests/test_combinators.py`)*
Per contract §8: `sequential`, `repeat`, `parallel`. Bare-callable passthrough
(`jax.nn.relu`), `ClassName_N` positional naming, `repeat` weight-tied via
`jax.lax.scan` relying on the re-entrant bound stack. **This is the canary for
core re-entrancy** — if the `_bound` stack is wrong, `repeat` breaks first.
**Tests:** sequential threads + names correctly; mixed Module/function;
`repeat(n)` has one param copy and equals n manual applications; `parallel`
returns a tuple; all under `jit`.

### Task C — Path utilities  *(→ `paths.py`, `tests/test_paths.py`)*
Per contract §9: glob matcher (dot-path, `*` = one segment), `freeze`, `select`.
**Tests:** matcher on `blocks.*.attn.*` style patterns; `select` returns
correct subtree; `freeze` produces the agreed freeze representation; operates on
a `state['params']` dict.

### Task D — Integration & examples  *(Wave 2; needs A+B)*  *(→ `tests/test_transforms.py`, `tests/test_integration.py`, `examples/`)*
- **Transform matrix** on a real composed model: `jit`, `vmap`, `scan`, `grad`,
  `remat`, `cond`, `fori_loop` — the contract §4 list, end to end.
- **examples/mlp.py** — the spec's optax training loop; loss decreases on
  synthetic data.
- **examples/batchnorm.py** — train→eval mode switch; buffers evolve; jit
  recompiles once per flag value.
- **examples/transformer.py** — blocks via `repeat`/`sequential`, tied embeddings
  via `self.shared.embed` (post-D2), KV cache over a short generation loop.

### Task E — Docs  *(draft Wave 1, finalize Wave 2)*  *(→ `README.md`, `docs/guide/`)*
- README: motivation, install, 20-line quickstart, link to guide.
- guide/: `quickstart`, `namespaces` (the four quadrants; `self.shared.embed`
  not bare), `transforms` (why no wrappers), `combinators`, `training` (optax
  loop). **Must match the FROZEN contract** — reviewer checks the tied-weight
  example uses the explicit accessor and no example implies rule-4 fallthrough.

---

## Phase 3 — Integrate & gate

Merge worktrees. Full-suite gate:
- `uv run pytest` green (core + A + B + C + D).
- `uv run ruff check` and `uv run ty check` clean across `src/` and `tests/`.
- Examples run to completion.
- Doc examples spot-checked against the contract (esp. D2 accessor, D3 raise).

**Definition of done (v1):** contract-conformant core; the five layers; three
combinators; path utils; the transform matrix passing; three runnable examples;
guide + README; every quality gate green; runtime deps = `jax` only.

---

## Out of scope (v2, designed in contract §7 / §0)
Forward-RNG (`self.rng()`) + Dropout, Conv, `@compact` lazy shapes, module-level
vmap (stacked params), tabulation.

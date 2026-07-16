# Pax — Core API Contract (v1)

> **Status: v1, stable.** This is the interface that layers, combinators, path
> utils, docs, and integration tests build against. The three original decisions
> (D1/D2/D3) plus the init-time guards (G1/G2/G3) from implementation review are
> resolved — see §11. Every signature and semantic rule here is normative; it is
> kept in sync with the implementation rather than patched locally in a
> sub-package.

This document specifies **what the core provides**, precisely enough that a layer
or combinator author never has to read the core implementation. It is organized
as: the state pytree, the namespace system, the public `Module` API, the
internal mechanism (for core + combinator authors), PRNG, and the sub-package
interfaces that fan out.

---

## 0. Scope & non-goals

**In scope for the core (built first, together):**
`Module` base class, `namespace()` + built-in taggers (`buffer`, `flags`),
state extraction, `forward`, `self.key()` + `pax.seed`, and the static/traced +
scoped/global state partitioning.

**Deferred to v2 (designed, not built):** forward-time RNG (`self.rng()`,
Dropout), Conv, `@compact`-style lazy shape inference, module-level vmap
(stacked params), tabulation.

**Never in scope (JAX already does it):** backprop, optimizer updates, optimizer
state, JIT, vmap, scan. Pax owns exactly two jobs: *declare/initialize state*
and *route state during the forward pass*.

**Pax only *adds* to JAX — it never wraps or replaces it.** Anything that
manipulates arrays stays raw `jax.*` in both code and examples: weight init is
`jax.random.normal(self.key(), ...)`, not a `pax.random` re-export (`pax.random`
is the init-seed context, a genuinely new capability, not a shadow of
`jax.random`). Re-exporting JAX would reintroduce the framework boundary Pax
exists to remove.

---

## 1. The state pytree

All model state is a single pytree: a plain `dict` with one top-level key per
**namespace**. Only namespaces that were actually written appear.

```python
state = {
    'params':  {...},        # traced, scoped   — the default namespace
    'buffers': {...},        # traced, scoped   — e.g. BatchNorm running stats
    'flags':   Static({...}),# static, global   — mode flags, concrete under jit
    # user-declared namespaces appear here too, e.g. 'cache', 'shared', 'metrics'
}
```

- `params` is **always present** (bare array assignment defaults here).
- **Scoped** namespaces (`params`, `buffers`, `cache`, …) are **nested dicts that
  mirror the module tree**, sparsely populated — a module appears under a
  namespace only if it (or a descendant) wrote to it.
- **Global** namespaces (`flags`, `shared`, `metrics`, …) are **flat dicts** —
  one shared dict read/written by every module regardless of tree position.
- **Static** namespaces are wrapped in a `Static` pytree node (§2.2) so their
  values ride in the treedef, not as leaves.

### Example

```python
class Model(Module):
    def __init__(self):
        self.encoder = Encoder()      # writes params only
        self.bn = BatchNorm(256)      # writes params + buffers
```

```python
state = {
    'params': {
        'encoder': {'W': ..., 'b': ...},
        'bn':      {'g': ..., 'b': ...},
    },
    'buffers': {
        'bn': {'running_mean': ..., 'running_var': ...},
        # 'encoder' absent — it never wrote a buffer
    },
    'flags': Static({'training': True}),
}
```

Paths mirror module structure: `bn` is the same key under both `params` and
`buffers`, but each namespace holds only the attrs written to it.

### Contract guarantees

- `state` is a **plain nested dict of JAX arrays** (plus `Static` nodes). It is a
  valid JAX pytree: pass it to `jax.grad`, `jax.jit`, `jax.tree_util`, `optax`,
  serialization — with no Pax involvement.
- `model.state()` returns a **fresh** pytree each call (no aliasing with the
  module's stored init values).
- Top-level keys are stable across `forward`: `new_state` has exactly the same
  namespaces as the input `state`.

---

## 2. Namespaces

### 2.1 `namespace()`

```python
def namespace(name: str, *, static: bool = False, scoped: bool = True) -> Tagger
```

Declares a namespace and returns a **tagger** — a callable `tag(value) -> Tagged`
that marks a value for storage in `name` when assigned to `self`. Declaring a
namespace registers its `(static, scoped)` spec in a process-global registry so
`state()` and `forward()` know how to treat it. Declaring the same name twice
with the same spec is idempotent; with a conflicting spec it raises.

Two orthogonal axes:

| | **scoped** (mirrors tree) | **global** (flat) |
|--|--|--|
| **traced** (default) | `params`, `buffers`, `cache` | `shared` params, global step |
| **static** (in treedef) | per-module config | `flags` |

Built-ins shipped by Pax:

```python
buffer = namespace('buffers')                          # traced, scoped
flags  = namespace('flags', static=True, scoped=False) # static,  global
```

Usage:

```python
self.W            = jax.random.normal(self.key(), (i, o)) * 0.01  # -> params (bare array)
self.running_mean = buffer(jnp.zeros(d))                      # -> buffers
self.training     = flags(True)                               # -> flags
self.embed        = shared(jax.random.normal(self.key(), (v, d))) # -> shared (global)
```

### 2.2 `Static` — how static namespaces survive `jax.jit`

**This is the mechanism the spec described only by its effect.** A static
namespace's dict is wrapped in a `Static` node registered as a JAX pytree whose
**children are empty and whose entire contents live in `aux_data`**:

```python
@register_pytree_node_class
class Static:
    def __init__(self, data: dict): ...
    def tree_flatten(self):   return (), _hashable(self.data)   # no leaves; data -> aux
    @classmethod
    def tree_unflatten(cls, aux, children):  return cls(dict(aux))
```

Consequences, all of which are exactly what the spec wants:

- Under `jax.jit(model.forward)`, `state['flags']` contributes to the **treedef**,
  not the leaves. So flag values are **part of the compile cache key**: JAX
  compiles one program per unique flag combination and reuses it thereafter.
- Inside the traced function, `state['flags']` is a `Static` holding **concrete
  Python values** (restored from `aux_data`), so `if self.flags.training:` is an
  ordinary Python branch — no tracer, no `ConcretizationError`.
- No manual "partition state before tracing" step is needed — the pytree
  structure does the partitioning. `forward` still *binds* the concrete values so
  `self.flags.training` resolves (§4, §5).

> **Refinement of the spec.** The spec says `forward` "partitions state into
> static and traced before entering traced code." That can't work when the user
> writes `jax.jit(model.forward)`, because jit traces the whole `state` argument
> *before* `forward`'s body runs. Representing static namespaces as `Static`
> pytree nodes achieves the identical caching behavior the spec describes, and
> works under every transform for free. This is the one intentional deviation.

`aux_data` must be **hashable and `==`-comparable** (JAX uses it in the cache
key). `_hashable(data)` recursively converts dicts to sorted key/value tuples;
therefore **static values must be hashable** (bool, int, str, tuple — not arrays
or dicts-of-arrays). This is the natural constraint for "small set of possible
states" flags.

> **D1 (resolved) — static + scoped is tier-2.** Static-*global* (`flags`) is
> essential for v1 and fully tested. Static-*scoped* ("per-module config") is
> representable with the same `Static` mechanism (nested `Static` nodes); it is
> **typed and smoke-tested, not exhaustively tested**, in v1. No worked example
> ships for it.

---

## 3. Defining a `Module` — user-facing rules

```python
class Linear(Module):
    def __init__(self, in_f, out_f):        # NO super().__init__() needed
        self.W = jax.random.normal(self.key(), (in_f, out_f)) * 0.01
        self.b = jnp.zeros(out_f)

    def __call__(self, x):                  # takes only x; reads state via self
        return x @ self.W + self.b
```

Rules the user relies on:

1. **No `super().__init__()`.** `Module.__new__` sets up all bookkeeping before
   `__init__` runs (§5.1). Calling `super().__init__()` is harmless (no-op) but
   never required.
2. **Assignment in `__init__` classifies by value type** (§5.2): a `jax.Array` →
   `params`; a `Tagged` value (`buffer(...)`, `flags(...)`, custom) → its
   namespace; a child `Module` → a subtree; anything else (int, float, str,
   tuple — config) → a plain Python attribute stored on the instance.
3. **`__call__(self, x)` reads/writes state through `self`.** Child modules are
   invoked as `self.child(x)` — just the input, no state argument. Buffer/cache
   writes are ordinary assignment: `self.running_mean = ...`.
4. **`self.key()`** yields a fresh PRNG subkey per call, for init only (§6).

### Attribute access resolution (`__getattr__`)

When `self.name` is **not** a real instance attribute (i.e. not config/method),
resolution proceeds in this order:

1. **Child module** named `name` → the child instance.
2. **Registered attr** (`name` in this module's init-time registry) → its value
   from the bound namespace slice (during `forward`) or the stored init value
   (unbound). Covers `self.W`, `self.running_mean`, `self.k`, and a module's own
   `self.embed` when it *declared* `embed`.
3. **Namespace-name accessor** (`name` is a declared namespace) → an accessor
   over that namespace's bound dict, supporting attribute reads on it. Enables
   `self.flags.training` and tied-weight reads `self.shared.embed`.
4. Otherwise `AttributeError`.

There is **no bare-name fallthrough** into namespaces a module didn't declare. A
module reads state it never assigned only through the explicit namespace accessor
(rule 3): a module that wants a weight tied in the global `shared` namespace but
did not declare it writes `self.shared.embed`, never bare `self.embed`. See §
Global for the worked tied-weight example.

> **D2 (resolved) — no fallthrough; explicit accessor for undeclared state.**
> The spec's tied-embedding example reads bare `self.embed` inside `Decoder`,
> which never assigned it. That required a fallthrough that lets a module absorb
> global state it never declared — rejected as too implicit. Tied weights are
> read via the explicit `self.<namespace>.<name>` accessor (`self.shared.embed`),
> the same rule that resolves `self.flags.training`. Global namespaces still
> provide one-copy storage (the actual reason `shared` is global); only the
> bare-name read is dropped. The spec's `Decoder` example is corrected to
> `x @ self.shared.embed.T`.

> **D3 (resolved) — forward-time write to an unregistered name raises.** If
> `__call__` assigns `self.newthing = ...` where `newthing` was never registered
> at init, `forward` raises. Silently dropping a state write is a multi-hour
> failure. Registered buffer/cache writes (`self.running_mean = ...`) are
> unaffected.

---

## 4. Public `Module` API

```python
class Module:
    def state(self, *, key: PRNGKey | None = None) -> dict: ...
    def forward(self, state: dict, x) -> tuple[dict, Any]: ...
    def key(self) -> PRNGKey: ...
    def __call__(self, x) -> Any: ...          # user-defined
```

### `state(key=None) -> dict`

Extracts a fresh state pytree from the module tree (walks children, collects each
namespace's init values into the layout of §1). With `key=`, **re-materializes**
params/traced state deterministically from that key instead of the values fixed
at construction (Option A escape hatch; see §6). Static namespaces are wrapped in
`Static` nodes. `params` is always present. Top-level namespace keys are emitted
in a deterministic (sorted) order.

**`key=` re-materialization replays construction** under the given key: it
reruns `type(self)(*ctor_args)` in a `seed`-like context. This faithfully redraws
every child a module builds *inside* its own `__init__` (the common case). It
**cannot** faithfully re-materialize a module built from **pre-existing child
instances** — every combinator (`sequential`/`repeat`/`parallel`) and any module
taking a `Module` argument — because replay would reuse the same child objects
and leave their params at the original draw. That case **raises** rather than
silently returning stale values; construct the whole model under
`with pax.seed(k):` to reseed it.

### `forward(state, x) -> (new_state, y)`

The external, pure entry point — signature `(carry, x) -> (carry, y)`, chosen to
match `jax.lax.scan` and functional convention (**state first, in and out**).
Semantics:

1. Bind `state`'s namespace slices onto `self` and, recursively, the right slice
   onto each child (§5.3). Static namespaces bind as concrete values.
2. Call the user's `self.__call__(x)`.
3. Collect writes made during the call and reassemble `new_state` with the same
   namespace layout (§5.4).
4. Unbind (restore the module to a pure, array-free, reusable state).
5. Return `(new_state, y)`.

**Guarantee for JAX transforms:** `forward` is a pure function of `(state, x)`.
Between calls the module object holds **only** Python config, child references,
the registry, and its init PRNG key — **no traced arrays**. Init values retained
on the instance (for re-`state()`) are **never read during `forward`**, so they
never leak into a trace. Therefore all of these work with **no wrappers**:

```python
jax.jit(model.forward)
jax.vmap(model.forward, in_axes=(None, 0))
jax.lax.scan(model.forward, state, xs)      # forward already has scan's signature
jax.grad(lambda p, s, x: loss(model.forward({**s, 'params': p}, x)))
jax.remat(model.forward)
```

### `key() -> PRNGKey`

Splits and advances this module's init key, returning a fresh subkey. **Init-time
only.** Forward-time randomness is `self.rng()` (v2, §7). See §6.

---

## 5. Internal mechanism (core + combinator authors)

Layer authors can stop at §4. Combinator authors and core implementers need this.

### 5.1 `__new__` bookkeeping

`Module.__new__` sets these instance slots (via `object.__setattr__`, all
underscore-prefixed so `__setattr__` passes them through) **before `__init__`
runs**:

- `_registry: dict[str, str]` — attr name → namespace name (built at init).
- `_init_values: dict[str, Any]` — attr name → value fixed at construction.
- `_children: dict[str, Module]` — ordered; insertion order = definition order.
- `_key: PRNGKey` — this module's init key, pulled from the `pax.seed` source
  (§6), advancing it so siblings differ.
- `_bound: list` — a **stack** of bound-state frames (empty when unbound).
- `_writes: list` — parallel stack of per-frame write accumulators.

`_bound`/`_writes` are stacks (not single slots) so `forward` is **re-entrant**:
a combinator that runs `child.forward` inside an already-bound parent (e.g.
`repeat` over `scan`) pushes a new frame and pops it, without clobbering the
parent's binding.

### 5.2 `__setattr__` classification

```
underscore-prefixed name         -> object.__setattr__ (internal bookkeeping)
bound (len(_bound) > 0)          -> forward-time WRITE: record under the
                                    registered namespace of `name` (D3: raise if
                                    unregistered); do not touch the instance
unbound (init):
    value is Module              -> _children[name]=value  (guards G1, G2)
    value is Tagged              -> registry[name]=value.ns; _init_values[name]=value.value  (guard G2)
    value is jax.Array           -> registry[name]='params'; _init_values[name]=value  (guard G2)
    value is a non-jax array-like -> raise TypeError (guard G3)
    else                         -> object.__setattr__ (plain config)
```

Parameters and buffers are stored in `_init_values`, **not** as real instance
attributes — that's what makes normal lookup miss and route through `__getattr__`
so reads can be redirected to bound state. Config (ints/floats/etc.) is a real
attribute and never routes.

**Init-time guards.** Every classification failure here is otherwise a *silent*
wrong result (a weight dropped from state, a tied weight that reads the wrong
slice) — the same failure class D3 exists to prevent. So init-time raises early:

- **G1 — instance aliasing.** Assigning the *same* `Module` instance under two
  different attribute names raises. Two names would push two frames onto that one
  instance's `_bound` stack and both reads resolve to the top of stack, silently
  reading the wrong slice. Tie weights via a **global namespace** (`shared`), or
  repeat a layer with `pax.repeat` — never by instance reuse.
- **G2 — name/category collision.** Reusing one attribute name across categories
  (e.g. a param name later reassigned as a child module, or vice versa) raises,
  rather than leaving the name in two registries where collection silently picks
  one. Re-assigning the *same* category/name is fine (last value wins).
- **G3 — non-jax array-like.** Assigning a value that is array-like but not a
  `jax.Array` (a NumPy `ndarray`, a NumPy scalar — anything with `__array__`)
  raises `TypeError`. Such a value would fall to the `else` branch and become a
  silent config attribute: absent from `state`, untraced, no gradients. Assign a
  JAX array (`jnp.*` / `jax.random.*`, seeded via `self.key()`) or tag it for a
  namespace. Genuine config (`int`/`float`/`str`/`tuple`) is unaffected.

Double-tagging (`buffer(buffer(x))`) likewise raises at the tagger (§2.1), so a
`Tagged` never becomes another namespace's stored *value*.

### 5.3 `_bind(state_slice)` — top-down

Called by `forward` on the root, recursing to children. For the module at a given
scope, `state_slice` is the dict of `{namespace: this-module's-slice}`:

- **Scoped** namespace `ns`: `frame[ns] = state_slice[ns]` (this module's own
  subtree). Each child `n` receives `state_slice[ns][n]` as *its* scoped slice.
- **Global** namespace `ns`: `frame[ns] = state_slice[ns]` (the same flat dict);
  every child receives the identical dict.
- **Static** namespace: bound as concrete values (the `Static`'s data).

Push `frame` onto `_bound`, push a fresh `{}` onto `_writes`, recurse into
children with their computed slices.

### 5.4 `_collect()` + `_unbind()` — bottom-up

After `__call__`, reassemble `new_state`:

- For each **scoped** namespace present in the subtree: the module's local slice
  is `{**bound_frame[ns], **writes[ns]}` for its own registered attrs, plus
  `{child_name: child._collect()[ns]}` for each child that has that namespace.
- For each **global** namespace: writes from anywhere in the tree merge into the
  single flat dict (last-write-wins within one forward; document that concurrent
  writes to the same global key across modules are the user's responsibility).
- **Static** namespaces: re-wrap the (possibly updated) values in `Static`.

`_unbind()` pops the `_bound`/`_writes` frames from every module in the subtree,
returning each instance to the pure, array-free state required by §4.

### 5.5 Naming

- **User-assigned attribute names** are the pytree keys for explicit assignments:
  `self.l1` → key `l1`.
- **`ClassName_N`** is used by positional combinators (§8) for children with no
  user-assigned name: `TransformerBlock_0`, `TransformerBlock_1`, …

---

## 6. PRNG — initialization (Option A: scoped seed context)

Chosen mechanism: **eager init with a thread-local seed context.** No key
threading in constructors; `self.key()` reads from an ambient source that is
*scoped to construction*, not a process-wide persistent global.

```python
import pax

with pax.seed(0):
    model = MLP(784, 256, 10)   # params concrete now; shapes visible
state = model.state()           # extract; no key arg required (matches spec)

# escape hatch: re-materialize under a different key
state = model.state(key=jax.random.key(42))
```

Public surface:

```python
def seed(n: int) -> ContextManager       # pax.seed(0): sets the thread-local init key
```

Mechanism:

- A **thread-local** holds the current init key. `pax.seed(n)` sets it to
  `jax.random.key(n)` on entry and restores the previous value on exit (nestable).
- `Module.__new__` pulls a per-module key by splitting the thread-local
  (advancing it so sibling modules differ). `self.key()` splits `self._key`
  per parameter.
- Outside any `pax.seed` context, the thread-local lazily initializes from a
  default seed and advances thread-locally.

**Reproducibility contract (document loudly):** init is a deterministic function
of *(root seed, construction order)*. It is **rename-safe but reorder-sensitive**
— identical to PyTorch's `manual_seed`. For guaranteed reproducibility, construct
under `with pax.seed(n):`. The thread-local is scoped and resets between
contexts, so it does not leak across unrelated constructions the way a persistent
process-global would.

---

## 7. Deferred: forward-time RNG (v2, designed here for continuity)

Not built in v1, but the design is fixed so buffers/collection don't need
rework later:

```python
rng = namespace('rng')   # traced, scoped — a real pytree leaf that threads through

class Module:
    def rng(self) -> PRNGKey:
        # split this module's rng leaf; write the advanced half back into state
        # (collected by forward like any buffer write); return a fresh subkey.
        ...
```

Rationale (why it can't reuse `self.key()`): forward runs *inside* `jit`/`vmap`/
`scan` and must stay pure, so its randomness must be an explicit pytree leaf that
is split-and-written-back each call — not an ambient thread-local. Mechanically
it is "a `buffer` write plus a split," which is exactly why it's cheap to add
once the §5.4 write-collection path is proven. Dropout ships on top of this in
v2.

---

## 8. Combinator interface contract

Combinators are **functions** returning a callable with the Module call
interface. They accept `Module` instances and/or bare callables (activations),
and integrate with binding/collection via §5.

```python
def sequential(*layers: Module | Callable) -> Module   # threads x through in order
def repeat(layer: Module, n: int) -> Module             # n applications, SHARED params
def parallel(*layers: Module | Callable) -> Module      # each applied to input; tuple out
```

Rules:

- **Bare callables** (e.g. `jax.nn.relu`) are detected via `not isinstance(x,
  Module)` and invoked directly with no scope management.
- **Positional naming**: children get `ClassName_N` keys (§5.5).
- `repeat` shares one parameter copy across `n` applications (weight-tied) and
  uses `jax.lax.scan` internally for memory efficiency. Because it calls
  `child.forward` inside a bound parent, it relies on the re-entrant bound stack
  (§5.1).
- `parallel` returns a tuple of outputs (or maps over a tuple input elementwise).

Combinators are convenience, not necessity — manual composition (residuals, etc.)
is just Python in `__call__` and needs no combinator.

---

## 9. Path / selective-operation utilities

Because scoped-namespace keys are meaningful strings, path-glob selection over a
namespace dict is natural. v1 surface:

```python
def match_path(pattern: str, path: str) -> bool      # the shared glob primitive
def select(tree: dict, pattern: str) -> dict         # pruned subtree of matching leaves
def freeze(tree: dict, pattern: str) -> dict         # boolean mask marking matches
```

Patterns are dot-paths with `*` wildcards matching **exactly one segment**
(`blocks.*.attn.*`). Matching is **whole-path**: pattern and path are split on
`.` and must have the **same number of segments**, matched pairwise. Consequences
(intentional, kept simple for v1):

- `*` never spans a `.`; there is no multi-segment wildcard.
- No prefix matching — a short pattern does not match a deeper path. So you
  cannot freeze a whole subtree with `blocks.0`; match at the exact leaf depth
  (`blocks.0.attn.*`, or one pattern per depth). `select`/`freeze` therefore only
  ever match true leaves whose full dot-path length equals the pattern length.

`select` returns a pytree of the same nesting restricted to matching leaves
(non-matching branches pruned). `freeze` returns a **boolean-mask pytree** of the
same shape as `tree` (`True` at frozen/matched leaves), which composes directly
with `optax.masked` / `optax.multi_transform` and round-trips through
`jax.tree_util`. Both operate on a single namespace's dict (typically
`state['params']`); the `match_path` primitive is the one source of truth.

---

## 10. Tooling & style conventions (all sub-packages)

- **Types:** full type hints, checked with `ty`. Public signatures fully
  annotated. Arrays typed as `jax.Array`; PRNG keys as `jax.Array` (typed-key)
  aliased `PRNGKey`.
- **Lint/format:** `ruff` (lint + format). No unused, no bare `except`.
- **Tests:** `pytest`. Every layer/combinator ships unit tests; the core ships
  property tests (roundtrip `state()` → `forward` → structure invariants) and
  transform tests (`jit`/`vmap`/`scan`/`grad` on `forward`).
- **Deps:** runtime = `jax` only. Dev = `optax`, `pytest`, `ty`, `ruff`.
- **Style:** functional/immutable where practical; small focused functions; early
  returns; no comments restating code (per repo conventions).

---

## 11. Resolved decisions

- **D1** — static-scoped namespaces are **tier-2**: typed + smoke-tested, no
  worked example in v1. *(§2.2)*
- **D2** — **no bare-name fallthrough.** Tied/undeclared global state is read via
  the explicit `self.<namespace>.<name>` accessor (`self.shared.embed`). The
  spec's bare-`self.embed`-in-`Decoder` example is corrected. *(§3)*
- **D3** — forward-time write to an **unregistered** name **raises.** *(§3)*

Init-time guards added after implementation review — each turns a *silent* wrong
result into an early raise (§5.2):

- **G1** — assigning one `Module` **instance under two names raises** (state
  aliasing); tie weights via a global namespace or `pax.repeat`.
- **G2** — reusing an attribute **name across param/child categories raises**.
- **G3** — assigning a **non-jax array-like** (NumPy `ndarray`/scalar) as a
  weight **raises**; use a `jax.Array` or a namespace tagger.
- **`state(key=...)`** re-materializes by replaying construction; it **raises**
  for modules built from pre-existing child instances (combinators, child-arg
  ctors) instead of returning stale values (§4).
- **`jax.random` convention** — array ops (init included) use raw `jax.*`; Pax
  only adds, never wraps (§0).

### Worked example — tied weights (post-D2)

```python
shared = namespace('shared', scoped=False)   # traced, global -> one copy in state

class Encoder(Module):
    def __init__(self, vocab, d):
        self.embed = shared(jax.random.normal(self.key(), (vocab, d)))  # declares embed
    def __call__(self, x):
        return self.embed[x]                    # rule 2: its own declared attr

class Decoder(Module):
    def __call__(self, x):
        return x @ self.shared.embed.T          # rule 3: explicit accessor, one shared array
```

`state['shared'] == {'embed': <one array>}`. Both modules read the same leaf;
gradients accumulate into it. `Decoder` never declares `embed` and never reads it
by bare name.

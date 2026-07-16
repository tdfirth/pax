# Namespaces

All model state is a single pytree: a plain `dict` with one top-level key per
**namespace** (contract §1). A namespace has two orthogonal properties, giving
four quadrants:

|                          | **scoped** (mirrors the module tree)   | **global** (one flat dict)  |
|--------------------------|----------------------------------------|-----------------------------|
| **traced** (default)     | `params`, `buffers`, custom `cache`    | `shared` weights, a step counter |
| **static** (in treedef)  | per-module config *(tier-2)*           | `flags`                     |

Two axes, two choices each:

- **traced vs. static.** Traced values are real pytree *leaves* — JAX arrays that
  flow through `grad`/`jit`/`vmap`. Static values ride in the *treedef* (see
  `Static` below), so they are concrete Python values inside a trace and part of
  the `jit` compile-cache key.
- **scoped vs. global.** Scoped namespaces are nested dicts that mirror the module
  tree, sparsely populated — a module appears under a namespace only if it (or a
  descendant) wrote to it. Global namespaces are a single flat dict shared by every
  module regardless of tree position.

## The built-ins

Pax ships three taggers plus the implicit `params` default:

```python
import pax

# bare jax.Array assignment          -> params   (traced, scoped) — always present
pax.buffer   # = namespace('buffers')                      # traced, scoped
pax.rng      # = namespace('rng')                          # traced, scoped — forward-RNG leaf
pax.flags    # = namespace('flags', static=True, scoped=False)  # static, global
```

`pax.rng` seeds a forward-time PRNG leaf (`self.dropout_key = pax.rng(self.key())`)
that `self.rng()` splits during `forward`; see
[transforms — forward-time randomness](transforms.md#forward-time-randomness--selfrng-and-dropout).

Usage in a module:

```python
import jax
import jax.numpy as jnp
import pax


class BatchNorm(pax.Module):
    def __init__(self, d):
        self.g = jnp.ones(d)                        # -> params (bare array)
        self.b = jnp.zeros(d)                       # -> params
        self.running_mean = pax.buffer(jnp.zeros(d))  # -> buffers
        self.running_var = pax.buffer(jnp.ones(d))    # -> buffers
        self.training = pax.flags(True)             # -> flags (static, global)

    def __call__(self, x):
        if self.flags.training:                     # concrete branch, even under jit
            mean = x.mean(0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean  # buffer write
            return (x - mean) * self.g + self.b
        return (x - self.running_mean) * self.g + self.b
```

Notes:

- **Buffer writes are ordinary assignment.** `self.running_mean = ...` during
  `forward` records a write under the `buffers` namespace; the updated value comes
  back in `new_state["buffers"]`. Writing to a name that was *never declared in
  `__init__`* raises (contract D3) — a dropped state write would be a silent,
  multi-hour bug.
- **`self.flags.training` is read through the namespace accessor**, not as a bare
  attribute, because `flags` is a global namespace the module reads but whose keys
  it addresses explicitly. Under `jit` it is a concrete `bool`, so the `if` is a
  real Python branch, not a tracer (no `ConcretizationError`).

## How static namespaces survive `jax.jit` — `Static`

A static namespace's dict is wrapped in a `pax.Static` pytree node whose children
are empty and whose entire contents live in `aux_data` (contract §2.2). The
consequence is exactly what mode flags want:

- Under `jax.jit(model.forward)`, `state["flags"]` contributes to the **treedef**,
  not the leaves, so flag values are part of the compile-cache key. JAX compiles
  one program per unique flag combination and reuses it thereafter.
- Inside the trace, `state["flags"]` holds **concrete Python values**, so
  `if self.flags.training:` is an ordinary branch.

Because the values become `aux_data`, **static values must be hashable** — `bool`,
`int`, `str`, `tuple`; not arrays or dicts of arrays. That is the natural
constraint for a "small set of possible states" flag.

## Declaring a custom namespace

`pax.namespace(name, *, static=False, scoped=True)` declares a namespace and
returns a tagger. Declaring the same name with the same spec is idempotent; a
conflicting spec raises.

```python
# a traced, scoped cache (e.g. KV cache that mirrors the module tree)
cache = pax.namespace("cache")

# a traced, GLOBAL namespace: one shared copy, read from anywhere in the tree
shared = pax.namespace("shared", scoped=False)
```

Assign a tagged value in `__init__` just like a buffer:

```python
class Attention(pax.Module):
    def __init__(self, d):
        self.k = cache(jnp.zeros((0, d)))   # -> 'cache' namespace
        ...
```

## Tied weights — the global `shared` namespace

A global namespace stores exactly **one copy** of a value, read by every module
regardless of position. That is the whole point of `shared`: tie an embedding
matrix between an encoder and a decoder so gradients accumulate into a single leaf.

The rule for *reading* tied state matters (contract §3, D2). A module reads state
it **declared itself** by bare name (`self.embed`). A module reads state it did
**not** declare only through the explicit namespace accessor
`self.<namespace>.<name>`:

```python
shared = pax.namespace("shared", scoped=False)   # traced, global -> one copy


class Encoder(pax.Module):
    def __init__(self, vocab, d):
        self.embed = shared(jax.random.normal(self.key(), (vocab, d)))  # declares embed

    def __call__(self, x):
        return self.embed[x]                # its own declared attr -> bare name is fine


class Decoder(pax.Module):
    def __call__(self, x):
        return x @ self.shared.embed.T      # NOT self.embed — explicit accessor
```

`state["shared"] == {"embed": <one array>}`. Both modules read the same leaf;
gradients accumulate into it. `Decoder` never declares `embed`, so it **must**
write `self.shared.embed` — bare `self.embed` inside `Decoder` is not how you read
undeclared shared state and would raise `AttributeError`. There is **no bare-name
fallthrough** into namespaces a module did not declare; the explicit accessor is
the same mechanism that resolves `self.flags.training`.

# Namespaces

In the quickstart, `model.state()` returned `{'params': {...}}`. But a real model
has more than trainable weights. A BatchNorm has running statistics that update
during the forward pass but aren't trained. It has a train/eval mode flag that
needs to be a *concrete* Python value even inside a compiled function. A
transformer might tie one embedding matrix between its input and output.

**Namespaces are how Pax keeps these different kinds of state separate** — each
in its own top-level key of the state pytree, so you can treat them differently:
differentiate params, thread buffers, hold flags constant.

```python
state = {
    'params':  {...},          # trainable weights
    'buffers': {...},          # e.g. BatchNorm running stats
    'flags':   Static({...}),  # mode flags, concrete under jit
    # ...and any custom namespaces you declare
}
```

Only namespaces that were actually written show up. A model with no buffers has
no `buffers` key. `params` is the exception — it's always present, because a bare
array assignment defaults to it.

## The two axes

Every namespace is defined by two independent yes/no choices. Understanding these
two axes is understanding the whole system.

**Traced vs. static.**

- *Traced* values are real pytree **leaves** — JAX arrays that flow through
  `grad` / `jit` / `vmap`. Params and buffers are traced.
- *Static* values ride in the pytree's **structure** (the "treedef"), not its
  leaves. Inside a trace they are concrete Python values, and they're part of
  `jit`'s compile-cache key. Flags are static — that's what lets
  `if self.flags.training:` be a real branch even under `jit`.

**Scoped vs. global.**

- *Scoped* namespaces are **nested dicts that mirror the module tree** — the path
  to a value in `state["params"]` matches the path to the module that owns it.
  They're sparsely populated: a module appears under a namespace only if it, or
  one of its descendants, wrote to it.
- *Global* namespaces are a **single flat dict** shared by every module,
  regardless of where it sits in the tree. This is how you store exactly one copy
  of something — a tied weight, a global step counter.

Two axes, two choices each, gives four quadrants:

|                      | **scoped** (mirrors the tree)         | **global** (one flat dict)        |
|----------------------|---------------------------------------|-----------------------------------|
| **traced** (default) | `params`, `buffers`, a custom `cache` | `shared` weights, a step counter  |
| **static**           | per-module config                     | `flags`                           |

Most of the time you'll use the three built-ins — `params` (traced/scoped),
`buffers` (traced/scoped), and `flags` (static/global). The other quadrants are
there when you need them.

## The built-ins

Pax ships three taggers, plus the implicit `params` default. A **tagger** is just
a function that marks a value for a namespace when you assign it to `self`:

```python
import pax

# a bare jax.Array assignment       -> params   (traced, scoped) — always present
pax.buffer   # tags a value for the 'buffers' namespace  (traced, scoped)
pax.flags    # tags a value for the 'flags' namespace    (static, global)
pax.rng      # tags a forward-time PRNG leaf              (traced, scoped)
```

`pax.rng` is for randomness *during* the forward pass (dropout masks and the
like) — it's covered in
[transforms → forward-time randomness](transforms.md#forward-time-randomness--selfrng-and-dropout).
Here's the other two in action:

```python
import jax
import jax.numpy as jnp
import pax


class BatchNorm(pax.Module):
    def __init__(self, d):
        self.g = jnp.ones(d)                          # -> params  (bare array)
        self.b = jnp.zeros(d)                         # -> params
        self.running_mean = pax.buffer(jnp.zeros(d))  # -> buffers
        self.running_var = pax.buffer(jnp.ones(d))    # -> buffers
        self.training = pax.flags(True)               # -> flags

    def __call__(self, x):
        if self.flags.training:                       # concrete branch, even under jit
            mean = x.mean(0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean  # buffer write
            return (x - mean) * self.g + self.b
        return (x - self.running_mean) * self.g + self.b
```

Two things worth pausing on:

- **A buffer write is ordinary assignment.** `self.running_mean = ...` inside
  `forward` doesn't mutate the object — it records a write to the `buffers`
  namespace, and the updated value comes back in `new_state["buffers"]`. Writing
  to a name you never declared in `__init__` is an error, not a silent no-op: a
  dropped state write would be a subtle, expensive bug, so Pax raises instead.

- **Flags are read through the namespace accessor**, `self.flags.training`, not
  as a bare `self.training`. That's because `flags` is a *global* namespace — a
  module reads it explicitly by name. The payoff is that under `jit` the flag is a
  concrete `bool`, so the `if` is a genuine Python branch (no
  `ConcretizationError`, no `jnp.where` gymnastics).

## Why static flags survive `jit`

This is the mechanism behind "concrete branch, even under `jit`," and it's worth
understanding once.

A static namespace's dict is wrapped in a `pax.Static` node — a pytree node whose
*leaves are empty* and whose contents live entirely in its structure metadata. So
when you `jax.jit(model.forward)`:

- `state["flags"]` contributes to the **treedef**, not the leaves. Flag values
  become part of the compile-cache key: **JAX compiles one program per unique flag
  combination** and reuses it thereafter.
- Inside the trace, `state["flags"]` holds **concrete Python values**, so
  `if self.flags.training:` is an ordinary branch.

The catch: because the values live in structure metadata, **static values must be
hashable** — `bool`, `int`, `str`, `tuple`. Not arrays, not dicts of arrays. That
is exactly the natural constraint for a mode flag with a small, finite set of
states, so it rarely bites. You'll see the two-programs-one-per-flag behavior
demonstrated in [transforms](transforms.md#static-namespaces-under-jit).

## Declaring your own namespace

When the three built-ins don't fit, declare a namespace with
`pax.namespace(name, *, static=False, scoped=True)`. It returns a tagger you use
just like `pax.buffer`. Declaring the same name with the same spec again is
harmless; declaring it with a *conflicting* spec raises.

```python
# a traced, scoped cache — mirrors the module tree, like a KV cache
cache = pax.namespace("cache")

# a traced, GLOBAL namespace — one shared copy, read from anywhere in the tree
shared = pax.namespace("shared", scoped=False)


class Attention(pax.Module):
    def __init__(self, d):
        self.k = cache(jnp.zeros((0, d)))   # -> the 'cache' namespace
        ...
```

The shipped `Attention` layer uses exactly this pattern for its KV cache — see
[`examples/transformer.py`](../../examples/transformer.py).

## Tied weights — the global `shared` namespace

Here's the reason global namespaces exist. A global namespace stores **exactly
one copy** of a value, read by every module no matter where it sits. That's how
you tie an embedding matrix between an encoder and a decoder: both read the same
array, so gradients accumulate into a single leaf.

There is one rule that makes this unambiguous. A module reads state **it declared
itself** by bare name (`self.embed`). A module reads state **it did not declare**
only through the explicit namespace accessor, `self.<namespace>.<name>`:

```python
shared = pax.namespace("shared", scoped=False)   # traced, global -> one copy


class Encoder(pax.Module):
    def __init__(self, vocab, d):
        self.embed = shared(jax.random.normal(self.key(), (vocab, d)))  # declares embed

    def __call__(self, x):
        return self.embed[x]                # its own declared attr -> bare name is fine


class Decoder(pax.Module):
    def __call__(self, x):
        return x @ self.shared.embed.T      # NOT declared here -> explicit accessor
```

`state["shared"]` is `{"embed": <one array>}` — a single leaf. Both modules read
it; both contribute gradients to it.

Why the asymmetry? `Encoder` declared `embed`, so bare `self.embed` is
unambiguous. `Decoder` never declared it, so a bare `self.embed` would have no
meaning — Pax does **not** let a module silently absorb global state it didn't
declare. The explicit `self.shared.embed` says "reach into the global `shared`
namespace and read `embed`," which is exactly the same accessor that resolves
`self.flags.training`. Bare `self.embed` inside `Decoder` would just raise
`AttributeError`.

With namespaces understood, the [combinators](combinators.md) page shows how they
flow automatically through composed models, and [training](training.md) shows how
you actually thread buffers and flip flags in a real loop.

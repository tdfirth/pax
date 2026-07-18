# Combinators

Most models are built from smaller layers wired together in a few standard
shapes: a chain, a repeated stack, a set of parallel branches. Combinators are
**functions that build those wirings for you**, returning a `Module` whose
children are the layers you passed in. Because the result is an ordinary module,
`state()` / `forward` / `jit` all work on it unchanged.

Pax ships three:

```python
pax.sequential(*layers)   # thread x through each layer in order
pax.repeat(layer, n)      # apply ONE layer n times, weight-tied (via scan)
pax.parallel(*layers)     # apply each layer to the input; return a tuple
```

Each accepts `Module` instances **and** bare callables. Construct them under a
seed, like any module:

```python
import jax
import jax.numpy as jnp
import pax
from pax.layers import Linear

with pax.seed(0):
    model = pax.sequential(Linear(4, 8), jax.nn.relu, Linear(8, 2))

state = model.state()
new_state, y = model.forward(state, jnp.ones((3, 4)))   # y: f32[3, 2]
```

## Bare callables pass through

Notice `jax.nn.relu` in that chain — a plain function, not a `Module`.
Combinators detect anything that isn't a `Module` and simply call it: no scope
management, no name, and **no entry in `state`**.

```python
list(state["params"].keys())   # ['Linear_0', 'Linear_2']
```

`relu` sat at position 1 but owns no weights, so only the two `Linear` children
show up. This is exactly why you write activations inline instead of wrapping them
in a layer — there's nothing to store.

## How unnamed children get their keys

The `Linear_0` / `Linear_2` above are **positional names**: a child you don't
assign to a named attribute gets the key `ClassName_N`, where `N` is its index in
the argument list. `relu` consumed index 1, so the second `Linear` is `Linear_2`,
not `Linear_1`:

```python
state["params"] == {
    "Linear_0": {"W": ..., "b": ...},
    "Linear_2": {"W": ..., "b": ...},
}
```

Two layers of the same class are told apart by index (`Linear_0`, `Linear_2`),
never by object identity. Passing the *same* instance twice is an error, not a way
to share weights — if you want weight sharing, use `repeat` (below) or a global
`shared` namespace (see
[namespaces → tied weights](namespaces.md#tied-weights--the-global-shared-namespace)).

## `parallel` — branch, return a tuple

`parallel` applies **every** layer to the same input and returns a tuple of their
outputs:

```python
with pax.seed(1):
    heads = pax.parallel(Linear(4, 8), Linear(4, 2))

state = heads.state()                       # {'params': {'Linear_0', 'Linear_1'}}
_, (a, b) = heads.forward(state, jnp.ones((3, 4)))
a.shape, b.shape                            # (3, 8), (3, 2)
```

Combine it with a bare callable — or just handle the tuple in an enclosing
module's `__call__` — to merge the branches yourself (e.g. concatenate them).

## `repeat` — weight-tied depth via `scan`

`repeat(layer, n)` applies **one** layer `n` times, sharing a *single* copy of its
parameters across all `n` applications. It registers the layer once — so state
holds exactly one copy — and drives the applications with `jax.lax.scan`
internally, which keeps memory flat regardless of depth:

```python
with pax.seed(2):
    block = Linear(4, 4)
    stack = pax.repeat(block, 3)

state = stack.state()
list(state["params"].keys())               # ['Linear_0']  — ONE copy, not three
list(state["params"]["Linear_0"].keys())   # ['W', 'b']
```

Those `n` applications of the tied layer are exactly `n` manual applies of that
one param copy:

```python
x = jnp.ones((3, 4))
_, y = stack.forward(state, x)

p = state["params"]["Linear_0"]
h = x
for _ in range(3):
    h = h @ p["W"] + p["b"]

jnp.allclose(y, h)                          # True
```

Reach for `repeat` when every layer in a stack *should* share weights. When you
want **distinct** weights per layer instead — the usual case for transformer
blocks — list them in `sequential`, which gives you `TransformerBlock_0`,
`TransformerBlock_1`, … each with its own params.

## Combinators nest

They're just modules, so they compose. A small classifier is a one-liner:

```python
from pax.layers import Linear, BatchNorm

with pax.seed(0):
    model = pax.sequential(
        Linear(784, 256),
        jax.nn.relu,
        BatchNorm(256),          # contributes params + buffers + a flag
        Linear(256, 10),
    )

state = model.state()
# state == {'params': {...}, 'buffers': {...}, 'flags': Static({'training': True})}
```

`BatchNorm` pulls `buffers` and `flags` into the state pytree, and the combinator
routes each namespace to the right child automatically — you never wire the
buffers or the flag by hand. Actually *training* this model — params-only
gradients, threading the buffers, flipping the flag for eval — is the subject of
the [training guide](training.md).

## When you don't need a combinator

Combinators are convenience, not necessity. Anything whose data flow *isn't* a
straight chain, a branch, or a tied repeat is just **plain Python in `__call__`**
— no combinator, no special API. A residual block, for instance, is a one-line
`__call__`:

```python
from pax.layers import Linear, LayerNorm

class Residual(pax.Module):
    def __init__(self, d):
        self.norm = LayerNorm(d)
        self.lin = Linear(d, d)

    def __call__(self, x):
        return x + self.lin(self.norm(x))    # the skip connection is just `+`
```

Children assigned to named attributes get those names as keys (`norm`, `lin`), and
`self.lin` resolves to the bound child during `forward` — exactly like reading a
param. Use a combinator only when it reads more clearly than the equivalent loop
or expression; otherwise, write the Python.

# Combinators

Combinators are **functions** that return a `Module` whose children are the layers
you pass in (contract §8). They exist for the common wiring patterns — chaining,
repeating, branching — and integrate with binding/collection like any other
module, so `state()` / `forward` / `jit` all work unchanged. Pax ships three:

```python
pax.sequential(*layers)   # thread x through each layer in order
pax.repeat(layer, n)      # n weight-tied applications of ONE layer (via scan)
pax.parallel(*layers)     # apply each layer to the input; return a tuple
```

Each accepts `Module` instances **and** bare callables. Construct them under a
seed like any module:

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

A layer that carries no state — an activation like `jax.nn.relu`, or any
`x -> y` function — is detected via `not isinstance(x, Module)` and invoked
directly. It gets **no scope management, no name, and no entry in `state`**:

```python
list(state["params"].keys())   # ['Linear_0', 'Linear_2']
```

`relu` sat at position 1 but owns no params, so only the two `Linear` children
appear. This is why you write activations inline rather than wrapping them —
there is nothing to store.

## Positional `ClassName_N` naming

Children you don't name explicitly get a positional key `ClassName_N`, where `N`
is the layer's index in the call (contract §5.5). The index is the position in
the full argument list, so the `relu` above consumes index 1 and the second
`Linear` is `Linear_2`:

```python
state["params"] == {
    "Linear_0": {"W": ..., "b": ...},
    "Linear_2": {"W": ..., "b": ...},
}
```

Two layers of the same class are disambiguated by index (`Linear_0`, `Linear_2`),
never by instance identity — reusing one instance twice is a guard error (G1), not
a way to share weights. To share weights, use `repeat` (below) or a global
`shared` namespace (see [namespaces](namespaces.md#tied-weights--the-global-shared-namespace)).

## `parallel` — branch, return a tuple

`parallel` applies **every** layer to the same input and returns a tuple of their
outputs (contract §8):

```python
with pax.seed(1):
    heads = pax.parallel(Linear(4, 8), Linear(4, 2))

state = heads.state()                       # {'params': {'Linear_0', 'Linear_1'}}
_, (a, b) = heads.forward(state, jnp.ones((3, 4)))
a.shape, b.shape                            # (3, 8), (3, 2)
```

Combine it with a bare callable to merge the branches yourself downstream (e.g.
concatenate in an enclosing module's `__call__`).

## `repeat` — weight-tied depth via `scan`

`repeat(layer, n)` applies **one** layer `n` times, sharing a single parameter
copy across all applications (weight-tied). It registers the layer once — so state
holds exactly one copy under `Linear_0` — and drives the `n` applications with
`jax.lax.scan` internally for memory efficiency:

```python
with pax.seed(2):
    block = Linear(4, 4)
    stack = pax.repeat(block, 3)

state = stack.state()
list(state["params"].keys())               # ['Linear_0']  — ONE copy, not three
list(state["params"]["Linear_0"].keys())   # ['W', 'b']
```

`n` applications of the tied layer are exactly `n` manual applies of that single
param copy:

```python
x = jnp.ones((3, 4))
_, y = stack.forward(state, x)

p = state["params"]["Linear_0"]
h = x
for _ in range(3):
    h = h @ p["W"] + p["b"]

jnp.allclose(y, h)                          # True
```

Use `repeat` for a homogeneous stack (transformer blocks, residual layers) where
every layer should share weights. For **distinct** weights per layer, list them in
`sequential` instead — that gives you `TransformerBlock_0`, `TransformerBlock_1`, …
each with its own params.

## A real composed model

Combinators nest, so a small classifier is a one-liner:

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

`BatchNorm` pulls `buffers` and `flags` into the state pytree; the combinator
routes each namespace to the right child automatically (contract §5.3). Training
this model — params-only gradients, threading buffers, flipping the flag for eval —
is covered in [training](training.md).

## When you don't need a combinator

Combinators are convenience, not necessity (contract §8). Anything with data flow
that isn't a straight chain, a branch, or a tied repeat is just **plain Python in
`__call__`** — no combinator, no special API. A residual block:

```python
from pax.layers import Linear, LayerNorm

class Residual(pax.Module):
    def __init__(self, d):
        self.norm = LayerNorm(d)
        self.lin = Linear(d, d)

    def __call__(self, x):
        return x + self.lin(self.norm(x))    # the skip connection is just `+`
```

Children get their **user-assigned** names as keys (`norm`, `lin`), and `self.lin`
resolves to the bound child during `forward`. Reach for a combinator only when it
reads more clearly than the equivalent loop or expression.

## Key idea

`self.h` is **not** the hidden states for the entire sequence. It's only the hidden state **at the current (last) timestep** — a single "memory" snapshot that gets carried from one timestep to the next.

Remember:
- The RNN processes one timestep at a time, in a loop.
- At each step it needs the **previous** hidden state `h_prev` to combine with the current input `x_t`.
- So `self.h` only ever stores *one* such `h_prev`, shape `(N, H)` — one H-vector per sample.
- `hs = np.empty((N, T, H))` is the **collection** of all hidden states over time — we allocate it up front so we can fill it step by step.

## Why the shapes differ

| Variable | Meaning | Shape |
|---|---|---|
| `self.h` | hidden state at **one** timestep (the "carry-over memory") | `(N, H)` |
| `hs` | hidden states at **all** T timesteps, stored for output/backprop | `(N, T, H)` |

Think of it as: `hs` is the full history, `self.h` is just the "latest value" pointer that keeps moving forward.

## Concrete example

Let `N = 3`, `T = 2`, `H = 4`.

At the start (`self.h` initialized):
```
self.h = [[0,0,0,0],   # sample 0's memory
          [0,0,0,0],   # sample 1's memory
          [0,0,0,0]]   # sample 2's memory
```
Shape `(3, 4)` = `(N, H)`.

Step `t=0`: use `self.h` + `xs[:, 0, :]` to compute a new `h`, store it into `hs[:, 0, :]`, then update `self.h` to this new value.

Step `t=1`: use the **new** `self.h` + `xs[:, 1, :]` to compute another `h`, store it into `hs[:, 1, :]`, update `self.h` again.

Now:
```
hs[:, 0, :]  → h at t=0   (N, H)
hs[:, 1, :]  → h at t=1   (N, H)
self.h       → h at t=1   (N, H)  ← same as hs[:, 1, :]
```

So `self.h` has shape `(N, H)` because it's a **snapshot**, not a sequence. `hs` has shape `(N, T, H)` because it holds **all** snapshots.

## Why it matters

When the next forward call comes in (e.g., a new batch continuing the sequence in a stateful RNN), `self.h` is exactly the `h_prev` you need to resume — one vector per sample. If it were `(N, T, H)`, you'd be storing a whole sequence, which defeats the purpose of "carry the last state forward."

--- 

Good question. The confusion is about **who owns the gradients**.

## The key point

Each `RNN` layer computes its **own local** gradients (`dWx`, `dWh`, `db`) — but those are gradients **for that one timestep only**.

But `Wx`, `Wh`, `b` are **shared across all T timesteps** — it's the *same* weight matrix reused in the loop. So the true gradient for `Wx` is the **sum** of the contributions from every timestep.

```
∂L/∂Wx = Σ_t  (∂L/∂Wx)_t
```

That's exactly what this does:

```python
for i, grad in enumerate(layer.grads):
    grads[i] += grad          # accumulate across t
```

## So what is `grads` here?

It's a **local accumulator** in `TimeRNN.backward`, initialized to zero:

```python
grads = [0, 0, 0]
```

Then it sums each layer's contribution. Finally:

```python
self.grads[i][...] = grad
```

writes the **total** into `TimeRNN.grads` — which is the array the **optimizer** actually reads and uses to update `Wx/Wh/b`.

## Why not just read `layer.grads` directly?

Because no single layer holds the correct total. `layer.grads` for the last timestep only reflects that timestep's contribution. The optimizer needs the **summed** gradient, and only `TimeRNN` (which owns the loop) can produce it.

## Summary

| Object | Holds | Used by |
|---|---|---|
| `layer.grads` (each RNN) | gradient for **one** timestep | only as an intermediate |
| `TimeRNN.grads` | **summed** gradient over all T | the optimizer |

So `grads` isn't redundant — it's the aggregation step. Without it, backprop through a shared-weight RNN would be wrong.
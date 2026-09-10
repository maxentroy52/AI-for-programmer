
## Where does the dimensionality reduction happen?

The reduction doesn't happen "automatically" inside `np.linalg.svd`. 

You have to **truncate** U yourself by taking only the first `k` columns:

```python
U, S, V = np.linalg.svd(W)
print(U.shape)   # (7, 7)
print(S.shape)   # (7,)

# Take only the first 2 columns → dimensionality reduction!
U_reduced = U[:, :2]
print(U_reduced.shape)   # (7, 2)
```

Now each word is represented by a **2-dimensional vector** instead of 7-dim.

## Why does keeping only the first k columns work?

**The columns of U are ordered by importance**, 
matched with the singular values in S (which are sorted in descending order):

```
S = [s0, s1, s2, ...]   with s0 >= s1 >= s2 >= ...
```

- Column 0 of U captures the **most variance** in W
- Column 1 captures the next most
- ...
- The last columns capture mostly noise

So `U[:, :k]` = "the k most important directions" — a compressed but meaningful representation.

## Picture it

```
Original W (7×7)               U (7×7)                    U[:, :2] (7×2)
                                                          
[ ... 7×7 ... ]   --SVD-->   [ u0 u1 u2 ... u6 ]  --cut-->  [ u0 u1 ]
                              (each u is a 7-dim         (each u is 2-dim
                               column vector)             column vector)
```

- **SVD** = "rotate/re-express W in a better basis" → U is still 7×7
- **Truncation** = "throw away the unimportant directions" → now it's 7×2

## In the book's code

You'll often see this pattern:

```python
U, S, V = np.linalg.svd(W)
# plot only the first 2 dimensions
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
```

That `U[word_id, 0]` and `U[word_id, 1]` — the **first two columns** of U — are the reduced word vectors. That's the whole point of doing SVD here: the raw W is `vocab_size × vocab_size` (huge for real corpora, e.g. 10000×10000), but `U[:, :k]` gives you `vocab_size × k` (e.g. 10000×100) — much smaller and dense.

## TL;DR

| Step | Operation | Shape |
|------|-----------|-------|
| Start | W | `(7, 7)` |
| SVD | `U, S, V = np.linalg.svd(W)` | U is `(7, 7)` |
| **Reduce** | `U[:, :2]` | `(7, 2)` ← reduction happens **here** |

So SVD by itself doesn't reduce dimensions — it just reorganizes W into a form where **truncation becomes meaningful** (drop the low-variance directions). The reduction is the `[:, :k]` slice you apply afterward.
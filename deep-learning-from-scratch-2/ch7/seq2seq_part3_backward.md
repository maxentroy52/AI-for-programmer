好问题。你的直觉「backward 是 forward 的反过程」**大体对**，但有个关键细节：**forward 只返回了最后一步 `C`，但内部 `hs` 是有所有时间步的。backward 要「补全」这个梯度。**

## 先看 forward 的输入输出

```python
def forward(self, xs):
    xs = self.embed_layer.forward(xs)      # (N, T, D)
    hs = self.lstm_layer.forward(xs)       # (N, T, H)
    self.hs = hs
    return hs[:, -1, :]                    # (N, H)  ← 只返回最后一步
```

- **输入**：`xs`，形状 `(N, T)`（id）
- **输出**：`hs[:, -1, :]`，形状 `(N, H)`（只返回最后一步）

**注意：forward 对外只暴露了「最后一步」`(N, H)`，但内部 `hs` 是完整的 `(N, T, H)`。**

## 所以 backward 的输入应该是「输出对应的梯度」

forward 的输出是 `(N, H)`，所以**上游传回来的梯度 `dh` 也是 `(N, H)`**——只对应「最后一步」的梯度。

```python
def backward(self, dh):        # dh: (N, H)，只对应最后一步
```

**这就是为什么 backward 的输入不是 `dhs (N, T, H)`，而是 `dh (N, H)`**——因为 forward 只输出了最后一步，上游只能对最后一步求导。

## 但 LSTM 内部需要所有时间步的梯度

`TimeLSTM.backward` 需要的是 `dhs (N, T, H)`——**每个时间步的梯度**。可是上游只给了最后一步的 `dh (N, H)`。

所以要做**补全**：

```python
dhs = np.zeros_like(self.hs)    # (N, T, H) 全 0
dhs[:, -1, :] = dh              # 只把最后一步填上
```

- `dhs` 初始全 0：**其他时间步没有来自上游的梯度**（因为 forward 没输出它们）。
- `dhs[:, -1, :] = dh`：最后一步的梯度 = 上游传回的 `dh`。

**这就是「把 forward 只暴露的最后一步，还原成完整的 `dhs`」的过程。**

## 为什么其他时间步是 0

forward 只 `return hs[:, -1, :]`，**前面的 `hs[:, 0, :], hs[:, 1, :]` 没有输出**。所以它们**对 loss 没有直接贡献**，上游不会给它们梯度。

- `hs[:, 0, :]` 的梯度 = 0（没被用到）
- `hs[:, 1, :]` 的梯度 = 0（没被用到）
- `hs[:, -1, :]` 的梯度 = `dh`（被用到，上游传回）

**但注意**：虽然前面的 `hs` 没被直接用到，它们**通过 LSTM 的时间步传递，间接影响了最后一步**。所以 `TimeLSTM.backward` 会**沿时间步回传**，把最后一步的梯度一路传到前面。这就是 `dhs` 全 0 但 backward 仍能算出前面时间步梯度的原因——**梯度是通过 LSTM 内部的 BPTT 传播的，不是通过 `dhs` 直接给的**。

## 所以「backward 是 forward 的反过程」准确说

| | forward | backward |
|---|---|---|
| 输入 | `xs (N, T)` | `dh (N, H)`（对应 forward 的输出） |
| 输出 | `hs[:, -1, :] (N, H)` | `dxs (N, T)`（对应 forward 的输入） |
| 中间 | 内部 `hs (N, T, H)` | 补全成 `dhs (N, T, H)` 再回传 |

**backward 的输入/输出，对应的是 forward 的「对外输入/输出」，不是内部中间量。**

- forward **对外输出** `(N, H)` → backward **对外输入** `dh (N, H)`。
- forward **对外输入** `xs (N, T)` → backward **对外输出** `dxs (N, T)`。

**`hs (N, T, H)` 是 forward 的内部中间量，不是对外输出。** 所以 backward 的输入不是 `dhs (N, T, H)`，而是 `dh (N, H)`——然后内部补全成 `dhs`。

## 一个类比

想象一个函数：

```python
def f(x):
    y = internal_compute(x)     # y 是 (N, T, H) 的中间量
    return y[:, -1, :]          # 只返回最后一步
```

- 调用方只看到「输入 `x`，输出最后一步」。
- 所以调用方只能对「最后一步」求导，传回 `dh (N, H)`。
- 但 `f` 内部要回传梯度给 `internal_compute`，需要 `dhs (N, T, H)`——**补全**：最后一步填 `dh`，其余 0。

**这就是 `dhs = np.zeros_like(self.hs); dhs[:, -1, :] = dh` 在做的事。**

## 一句话

> forward **对外只输出最后一步** `(N, H)`，所以 backward **对外只接收最后一步的梯度** `dh (N, H)`——不是 `dhs (N, T, H)`。内部 `dhs` 全 0，只把最后一步填成 `dh`，然后交给 `TimeLSTM.backward` 沿时间步回传。**前面的时间步梯度不是上游给的，是 LSTM 内部 BPTT 算出来的。** 「backward 是 forward 的反过程」指的是**对外接口**，不是内部中间量。

## 顺带：这行 `dhs[:, -1, :] = dh` 为什么对

forward：`return hs[:, -1, :]`。对 `hs[:, -1, :]` 求导，梯度就是 `dh`，直接写到 `dhs` 的最后一步。前面没被用到，梯度 0。**这是链式法则的直接应用。**

## 再顺带：`self.hs` 为什么要存

```python
self.hs = hs    # forward 时存下来
```

backward 要用 `self.hs` 的形状来建 `dhs = np.zeros_like(self.hs)`。**不存的话，backward 不知道 `(N, T, H)` 是多少。** 所以 forward 存 `self.hs`，backward 拿来建 `dhs`。这也是为什么 forward 里 `self.hs = hs` 是必要的。
你抓到一个**矛盾**了，而且抓得对。我之前的说法需要修正。

## 先看代码事实

书上的 `Decoder`：

```python
self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)   # ← stateful=True
```

**书上的 Decoder 确实是 `stateful=True`。** 我之前说「decoder 也通常是 `stateful=False`」——那是我讲错了，至少和这本书的代码不符。

## 为什么书上用 `stateful=True`

因为 **decoder 是自回归的**——它逐个时间步生成目标词，**上一步的 `h`/`c` 必须传给下一步**：

```python
h_0 = C                     # 初始状态来自 encoder
h_1, y_1 = step(<sos>, h_0)
h_2, y_2 = step(y_1,   h_1)  # ← 需要 h_1
h_3, y_3 = step(y_2,   h_2)  # ← 需要 h_2
```

`stateful=True` 让 `self.h`/`self.c` **在多次 `forward` 调用之间保留**。这样：

```python
self.lstm.set_state(h)      # forward 开头，把 encoder 的 C 设为初始状态
out = self.lstm.forward(out) # 内部沿 T 步循环，状态逐步传递
```

**decoder 需要状态在时间步之间连续传递**，所以 `stateful=True` 是合理的。

## 那我之前为什么说 `stateful=False`

我之前讲的是「**decoder 每次处理新句子时，状态应该由 `C` 显式设置，而不是靠隐式延续**」。这个点本身没错，但**实现方式**我讲错了：

- 正确做法不是 `stateful=False`，而是 **`stateful=True` + 每次 forward 前 `set_state(h)` 显式覆盖初始状态**。
- `set_state(h)` 把 `self.h` 设成 `C`，覆盖掉上一次的旧状态。所以即使 `stateful=True`，**每个新句子开始时状态也是干净的**（因为被显式覆盖了）。

书上就是这么做的：

```python
def forward(self, xs, h):
    self.lstm.set_state(h)     # ← 显式设置初始状态 = C
    ...
```

## 所以矛盾的解释

| 说法 | 对错 | 说明 |
|---|---|---|
| 「decoder 每次用 C 显式设初始状态」 | ✅ 对 | `set_state(h)` |
| 「decoder 用 `stateful=False`」 | ❌ 错 | 书上用 `stateful=True` |
| 「decoder 状态不跨句子延续」 | ✅ 对 | 但靠 `set_state` 覆盖，不是靠 `stateful=False` |

**「状态不延续」和「`stateful=False`」是两件事。** 书上用 `stateful=True`（为了时间步内连续），但用 `set_state` 保证每个新句子从 `C` 开始（不延续旧句）。**我之前把这两件事混了，说错了。**

## 为什么 `stateful=True` + `set_state` 比 `stateful=False` 更合适

如果用 `stateful=False`：

```python
def forward(self, xs, h):
    # stateful=False 会强制 self.h = zeros，忽略传入的 h
    out = self.lstm.forward(out)   # ← 初始状态被强制清零，C 丢了！
```

`stateful=False` 会让 `TimeLSTM.forward` 里的 `if not self.stateful or self.h is None: self.h = zeros` **强制清零**，把 encoder 传来的 `C` 覆盖掉。**这样 decoder 就拿不到 `C` 了。**

所以 **decoder 必须 `stateful=True`**，才能用 `set_state(h)` 把 `C` 设进去，并且在时间步之间保留。

## 修正后的准确说法

> **Decoder 用 `stateful=True`**，因为：
> 1. 自回归生成需要状态在时间步之间连续传递。
> 2. `set_state(h)` 把 encoder 的 `C` 设为初始状态（显式覆盖，不靠隐式延续）。
> 3. 每个新句子 forward 前都 `set_state(h)`，所以**不会延续上一个句子的状态**——「不延续」靠 `set_state` 覆盖实现，不是靠 `stateful=False`。
>
> **我之前说「decoder 用 `stateful=False`」是错的。** 正确的是 `stateful=True` + `set_state`。

## 一句话

> 你抓得对，**书上的 Decoder 是 `stateful=True`**。我之前说 `stateful=False` 讲错了。「decoder 不延续旧句状态」是对的，但实现方式是 **`stateful=True` + 每次 `set_state(C)` 显式覆盖**，不是 `stateful=False`。`stateful=False` 会强制清零，反而把 `C` 冲掉——decoder 必须 `stateful=True` 才能用 `set_state` 设初始状态。

## 顺带：Encoder 为什么 `stateful=False`

`Encoder` 没有「初始状态来自别处」的需求，它每次从零开始读源句就行。所以 `stateful=False` 合适：

```python
self.lstm_layer = TimeLSTM(..., stateful=False)   # 每次 forward 从 0 开始
```

**Encoder 和 Decoder 的 `stateful` 不同，是因为职责不同：**

- **Encoder**：读源句，从零开始，不需要外部状态 → `stateful=False`。
- **Decoder**：生成目标句，初始状态来自 `C`，需要 `set_state` → `stateful=True`。
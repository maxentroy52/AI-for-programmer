你描述的**大方向对，但细节有两处需要修正**：

1. **不是「给到每一个 LSTM layer 和 Affine layer」**，而是**给到 decoder 的每一个时间步的 LSTM**。
2. **不是把 `h` 直接拼到 LSTM 内部**，而是**把 `h` 作为额外的输入，拼接到 decoder 每个时间步的输入上**。

我按书上的 peephole / peaky 说法讲清楚。

## 先澄清名字

书上叫 **Peeky Decoder**（不是 peaky，是 **peeky**，来自 **peek**「偷看」）。它的意思是：**decoder 在每一步都「偷看」一下 encoder 的输出 `h`**。

## 核心思想

普通 seq2seq：

```
decoder 只在初始状态看到 h（h_0 = C），之后 h 就不再直接可见
```

Peeky decoder：

```
decoder 的每一个时间步，都把 h 拼接到输入上，一起喂给 LSTM 和 Affine
```

**「每一步都能偷看 `h`」，所以叫 peeky。**

## 具体怎么做

### 普通 decoder

```python
# 每个时间步
out = embed(x_t)           # (N, D)
h_t, c_t = lstm(out, h_{t-1}, c_{t-1})
score = affine(h_t)        # (N, V)
```

### Peeky decoder

```python
# 每个时间步
out = embed(x_t)                    # (N, D)
out = concat(out, h)                # (N, D + H)  ← 把 encoder 的 h 拼进来
h_t, c_t = lstm(out, h_{t-1}, c_{t-1})
score = affine(concat(h_t, h))      # (N, H + H) ← 再拼一次，给 affine
```

**两处拼接：**

1. **LSTM 的输入**：`embed(x_t)` 拼上 `h` → `(N, D + H)`。
2. **Affine 的输入**：`h_t` 拼上 `h` → `(N, H + H)`。

所以你说的「给到每一个 LSTM layer 和 Affine layer」——**准确说是「每个时间步的 LSTM 输入」和「每个时间步的 Affine 输入」**，不是「层与层之间」。

## 为什么这样设计

**普通 decoder 的问题**：`h` 只在 `h_0 = C` 时进入 LSTM，之后信息一路传递、可能衰减。到后面几步，`h` 的信息可能已经很弱了。

**Peeky 的做法**：**每一步都直接把 `h` 拼到输入里**，不依赖「一路传下来」。这样：

- LSTM 每一步都能**直接访问 `h`**，不用靠状态传递。
- Affine 也能**直接看到 `h`**，输出时参考源句信息。

**效果**：decoder 对源句信息的利用率更高，翻译/加法效果更好。

## 代价：参数变多

因为输入维度变了：

- LSTM 的 `Wx`：`(D, 4H)` → `(D + H, 4H)`。
- Affine 的 `W`：`(H, V)` → `(H + H, V)`。

**参数增加，计算变慢**，但效果通常更好。

## 一句话

> **Peeky decoder**（peek = 偷看）：**decoder 的每个时间步，都把 encoder 的 `h` 拼接到 LSTM 输入和 Affine 输入上**，让 decoder 每一步都能「偷看」源句信息，而不是只靠初始状态 `h_0`。你说的「给到 LSTM 和 Affine」方向对，但准确说是「**每个时间步的输入**」拼接，不是「层与层之间」。代价是参数变多。

## 顺带：peeky 和 reverse、attention 的关系

| 技巧 | 解决什么 | 方式 |
|---|---|---|
| **reverse** | `C` 对开头信息弱 | 翻转输入，让开头对齐 |
| **peeky** | `h` 只在初始状态可见，后面衰减 | 每一步拼接 `h` |
| **attention** | `C` 是固定长度瓶颈 | 每一步动态「看」encoder 所有输出 |

**三者都是「让 decoder 更好地利用 encoder 信息」**，但方式不同：

- reverse：调输入顺序。
- peeky：每一步拼接 `h`。
- attention：每一步加权看所有 `h_t`。

**attention 是最终的通用解**，peeky 和 reverse 是它的前身/简化版。
好，把 LSTM 内部完整代码补全，不省略。

## LSTM.forward 完整代码

```python
def forward(self, x, h_prev, c_prev):
    Wx, Wh, b = self.params
    N, H = h_prev.shape

    # 1. 四个门的预激活，拼成一个大矩阵
    A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
    #   x:      (N, D)  = (2, 128)
    #   Wx:     (D, 4H) = (128, 256)
    #   x @ Wx: (N, 4H) = (2, 256)
    #   h_prev: (N, H)  = (2, 64)
    #   Wh:     (H, 4H) = (64, 256)
    #   h @ Wh: (N, 4H) = (2, 256)
    #   b:      (4H,)   = (256,)
    #   A:      (N, 4H) = (2, 256)

    # 2. 切片：把 A 按 H 切成四段，每段对应一个门
    f = A[:, :H]            # (N, H) = (2, 64)   遗忘门预激活
    g = A[:, H:2*H]         # (N, H) = (2, 64)   候选记忆预激活
    i = A[:, 2*H:3*H]       # (N, H) = (2, 64)   输入门预激活
    o = A[:, 3*H:]          # (N, H) = (2, 64)   输出门预激活

    # 3. 各自过激活函数
    f = sigmoid(f)          # (N, H) = (2, 64)   遗忘门：0~1
    g = np.tanh(g)          # (N, H) = (2, 64)   候选：-1~1
    i = sigmoid(i)          # (N, H) = (2, 64)   输入门：0~1
    o = sigmoid(o)          # (N, H) = (2, 64)   输出门：0~1

    # 4. 更新记忆单元 c
    c_next = f * c_prev + g * i
    #   f * c_prev: (N, H) = (2, 64)   保留的旧记忆
    #   g * i:      (N, H) = (2, 64)   写入的新信息
    #   c_next:     (N, H) = (2, 64)

    # 5. 计算隐藏状态 h
    h_next = o * np.tanh(c_next)
    #   tanh(c_next): (N, H) = (2, 64)
    #   o * ...:      (N, H) = (2, 64)
    #   h_next:       (N, H) = (2, 64)

    # 6. 缓存 backward 需要的中间量
    self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)

    return h_next, c_next
    #   h_next: (N, H) = (2, 64)
    #   c_next: (N, H) = (2, 64)
```

## 维度变化全流程

```
x:        (N, D)  = (2, 128)
h_prev:   (N, H)  = (2, 64)
c_prev:   (N, H)  = (2, 64)

A = x@Wx + h@Wh + b:   (N, 4H) = (2, 256)     ← 四个门拼一起
  ↓ slice
f, g, i, o:            各 (N, H) = (2, 64)     ← 切成四段
  ↓ 激活
f, g, i, o:            各 (N, H) = (2, 64)     ← sigmoid/tanh

c_next = f*c_prev + g*i:  (N, H) = (2, 64)     ← 新记忆
h_next = o*tanh(c_next):  (N, H) = (2, 64)     ← 新隐藏状态

return h_next, c_next:   各 (N, H) = (2, 64)
```

## 关键点：`4H` 在哪里消失

```
A: (N, 4H) = (2, 256)       ← 四个门拼一起，4H 维
  ↓ slice 成 4 段，每段 H 维
f, g, i, o: 各 (N, H)       ← 每段 64 维
  ↓ 组合
c_next, h_next: (N, H)      ← 输出 H 维
```

**`4H` 只在 `A` 这一步存在。slice 后每个门是 `H` 维，组合出的 `c_next`/`h_next` 也是 `H` 维。** 所以 LSTM 对外输出是 `H`，不是 `4H`。

## 回到你的问题

> 「`(H, V) = (64, 1024)` 这不是没有对齐吗？为什么把 `(N, 4H)` 变成了 `(N, H)`？」

- **`4H` 是 `A` 的维度**，LSTM **内部**中间量。
- **`H` 是 `h_next` 的维度**，LSTM **对外输出**。
- **`4H → H` 发生在 LSTM 内部**（slice + 门控），**不是 Affine 做的**。
- Affine 拿到的是 `h_next (N, H) = (2, 64)`，和 `affine_W (H, V) = (64, 1024)` **对齐**（64 = 64）。

**所以「对齐」没问题**：Affine 的输入是 `H=64`，`affine_W` 的第 1 维也是 `H=64`。`4H=256` 是 LSTM 内部的，和 Affine 无关。
## Transformer Encoder 完整流程

```
输入 token id
  ↓
1. Input Embedding           id → 向量 (T, D)
  ↓
2. Positional Embedding      + 位置信息（self-attention 无视顺序，必须加）
  ↓
3. Multi-Head Attention      n_heads 组 Q/K/V，各自 attention，拼接 + W_O
  ↓
4. Add & Norm                LayerNorm(x + MHA(x))     ← 残差 + 归一化
  ↓
5. FFN                       每个位置独立：升维 4D → ReLU → 降维 D
  ↓
6. Add & Norm                LayerNorm(x + FFN(x))     ← 残差 + 归一化
  ↓
输出 (T, D)   ← 每个位置融合上下文的新编码
```

## 每个组件解决什么

| 组件 | 解决什么 |
|---|---|
| **Input Embedding** | id → 向量 |
| **Positional Embedding** | self-attention 无视顺序 → 加位置信息 |
| **Multi-Head Attention** | 每个位置**看所有位置**，多 head 学多种相关性模式 |
| **Add & Norm** | 残差保梯度、LayerNorm 稳定尺度 |
| **FFN** | attention 是线性的 → FFN 提供**非线性加工** |

## 一句话

> **Encoder = Embedding + 位置编码 → [Multi-Head Attention → Add&Norm → FFN → Add&Norm] × N 层。** **MHA 管「跨位置混合信息」，FFN 管「每个位置独立非线性加工」，Add&Norm 保训练稳定。** 输出是「每个位置融合了全局上下文的新编码」。

## 和 LSTM encoder 的对比

| | LSTM encoder | Transformer encoder |
|---|---|---|
| 编码方式 | 串行，`h_t` 依赖 `h_{t-1}` | **并行**，每个位置直接看所有位置 |
| 路径长度 | `O(T)` | **`O(1)`** |
| 长距离依赖 | 弱（靠传递） | **强（直接看）** |
| 并行 | ❌ | ✅ |

**Transformer encoder 用「self-attention + FFN」替代了 LSTM 的「循环」——并行、长距离、可堆叠。**

## 下一步

Encoder 走完了。接下来是 **Decoder**（masked self-attention + cross-attention），或者**位置编码的具体公式**，或者**进代码拼完整 Transformer**。你想哪个？
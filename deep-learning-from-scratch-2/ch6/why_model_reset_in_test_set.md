因为 **LSTM 是有状态的**，`eval_perplexity` 又要从语料开头开始评估——所以必须先清掉训练留下的隐藏状态。

## 具体原因

训练结束时，`model.lstm_layer` 里的 `self.h` 和 `self.c` 还留着**训练最后一个 batch 的状态**。如果直接拿这个状态去评估测试集，会发生：

```
测试集的第一个 token  →  用「训练集最后一刻的记忆」去理解它
```

这显然是错的——测试集和训练集是**不同的语料**，不应该共享隐藏状态。

`reset_state()` 做的事：

```python
def reset_state(self):
    self.lstm_layer.reset_state()   # self.h = None, self.c = None
```

清空后，`TimeLSTM.forward` 里会重新初始化：

```python
if not self.stateful or self.h is None:
    self.h = np.zeros((N, H))
```

于是评估**从零状态开始**，干净地处理测试集。

## 为什么训练时不用每次 reset

训练时**故意**不 reset——lane 的状态要跨 batch 延续，这才是 stateful RNN 的设计。reset 只在**切换语料**时做：

| 场景 | 是否 reset |
|---|---|
| 训练中，batch 之间 | ❌ 不 reset（状态延续） |
| 训练结束 → 评估开始 | ✅ reset（换语料） |
| 评估完 → 再做别的评估 | ✅ 再 reset |
| 生成新句子 | ✅ reset（新序列） |

## 不 reset 会怎样

测试集的第一个位置会拿到训练集最后的状态。影响大小取决于语料差异：

- 如果训练集和测试集风格相似，影响可能不大。
- 但**逻辑上就是错的**——测试集应该独立评估。
- 严格来说，这会让 ppl 有一点偏差。

## 一句话

> `reset_state()` 是为了**清掉训练留下的隐藏状态**，让测试集评估从**零状态**开始，独立、干净地衡量模型在测试集上的表现。不 reset 的话，测试集的第一个 token 会用训练集最后的记忆去理解，逻辑上错误。

## 顺带：`eval_perplexity` 内部也没 reset

`eval_perplexity` 本身不调 `reset_state`，所以**调用方必须负责 reset**。这是设计选择——评估函数保持纯粹，状态管理交给调用方。所以每次评估前都要手动 `model.reset_state()`。
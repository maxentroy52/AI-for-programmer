## Details

perceptron的概念，我同步到另一个文档。naive_perceptron的实现，就是按照标准定义实现的，包括参数学习的策略。

这里我补充一点：
>In the modern sense, the perceptron is an algorithm for learning a binary classifier called a threshold function: 
> a function that maps its input to an output value(a single binary value)

- 多元一次线性函数，本质是一个超平面。分类的作用是找到这个超平面，不是拟合。
- 分类的具体过程是，把样本带入超平面，看其在超平面上下的位置从而进行分类。
- 如果从NN的角度来看，没有hidden layer. 其实就是2层，一层输入，一层输出。当然，说成是single layer，也没问题。

## Sgd implementation

### Cost function

$$ Mean\, Square \, Error = \frac{ \sum [y_{i} - ( wx_{i} + b )]^{2} }{N} $$

### Gradient descent

$$ \frac{ \partial f }{ w } = \frac{2}{N} \cdot \sum x_{i} \cdot (wx_{i} + b - y_{i})$$

$$ \frac{ \partial f }{ b } = \frac{2}{N} \cdot \sum (wx_{i} + b - y_{i})$$


实现上，没有问题。debug的问题在于
- 特征输入没有对齐。训练的样本和在线输入的对象，特征没有对齐。所以，特征在离线一致是需要的。deepseek发现的，我这个点没发现，是因为压根没意识到。
- learning rate这个我觉得说的过去，这个我确实不懂。
- 还有，函数设计这里。传入，传出没有搞明白。本来写对了，后来写错了。
- 还有一点，本次迭代，一定是当前迭代的参数。计算的也是当前参数的梯度，从而更新下次的参数。
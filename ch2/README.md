## Details

perceptron的概念，我同步到另一个文档。naive_perceptron的实现，就是按照标准定义实现的，包括参数学习的策略。

这里我补充一点：
>In the modern sense, the perceptron is an algorithm for learning a binary classifier called a threshold function: 
> a function that maps its input to an output value(a single binary value)

- 多元一次线性函数，本质是一个超平面。分类的作用是找到这个超平面，不是拟合。
- 分类的具体过程是，把样本带入超平面，看其在超平面上下的位置从而进行分类。
- 如果从NN的角度来看，没有hidden layer. 其实就是2层，一层输入，一层输出。当然，说成是single layer，也没问题。
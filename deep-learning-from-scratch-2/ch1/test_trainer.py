import sys

from common.trainer import Trainer

sys.path.append('..')
from dataset import spiral

from common.optimizers import SGD
from two_layer_net import TwoLayerNet

# 1. 设定超参数
max_epoch = 300
batch_size = 30
eval_interval = 5
learning_rate = 1.0

# 2.读入数据
x, t = spiral.load_data()

# 3.构建模型和优化器
input_size = 2
hidden_size = 10
output_size = 3
model = TwoLayerNet(input_size, hidden_size, output_size)
optimizer = SGD(learning_rate)

# 4.模型训练
trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval)
trainer.plot()

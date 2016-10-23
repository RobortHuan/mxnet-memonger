# MXNet Memory Monger

This project contains a 150 lines of python script to give sublinear memory plans of deep neural networks.
This allows you to trade computation for memory and get sublinear memory cost,
so you can train bigger/deeper nets with limited resources.
  这个项目包含150行python代码用于sublinear memory plans的深度神经网络
## Reference Paper

[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) Arxiv 1604.06174
使用sublinear memory plans训练神经网络
## How to Use

This code is based on [MXNet](https://github.com/dmlc/mxnet), a lightweight, flexible and efficient framework for deep learning.
改程序基于mxnet，一款轻量级、灵活、高效的深度学习框架
- Configure your network as you normally will do using symbolic API  配置网络用于你以后使用symbolic 的API接口
- Give hint to the allocator about the possible places that we need to bookkeep computations.  给予分类器提示关于我们可能用到bookkeep computations的地方
  - Set attribute ```mirror_stage='True'```, see [example_resnet.py](example_resnet.py#L25)
  - The memonger will try to find possible dividing points on the nodes that are annotated as mirror_stage. memonger将试图寻找可能的分类点在标注为 mirror_stage的节点上
- Call ```memonger.search_plan``` to get an symbolic graph with memory plan.

```python
import mxnet as mx
import memonger

# configure your network
net = my_symbol()

# call memory optimizer to search possible memory plan.
net_planned = memonger.search_plan(net)

# use as normal
model = mx.FeedForward(net_planned, ...)
model.fit(...)
```

## Write your Own Memory Optimizer
写下自己的优化程序
MXNet's symbolic graph support attribute to give hint on whether (mirror attribute) a result
can be recomputed or not. You can choose to re-compute instead of remembering a result
for less memory consumption. To set output of a symbol to be re-computable, use
```python
sym._set_attr(force_mirroring='True')
```

mxnet-memonger actually use the same way to do memory planning. You can simply write your own memory
allocator by setting the force_mirroring attribute in a smart way.

# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import tensorflow as tf
from numpy.random import RandomState
import numpy as np

batch_size = 8  # 定义训练数据batch大小

# 神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x_input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y_input")  # 真实标签

# 前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)  # 预测值

# 定义损失函数与反向传播过程
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1))) # 定义损失函数时，只需关注预测值跟真实值
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy) # 定义优化算法时，只需关注特征跟真实值，因为特征知道了，可以通过前向传播过程自动计算出预测值

# 生成模拟数据集
rdm = RandomState(1)
data_size = 128
data_x = rdm.rand(data_size, 2)

# 定义规则给出样本的标签, x1,x2是样本的两个特征
Y = np.array([[int(x1 + x2 < 1)] for (x1, x2) in data_x])
# 创建一个会话来运行tensorflow程序
with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("初始参数：")
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))

    # print(type(data_x[0:8,:]))
    # print(type(data_x[0:8]))
    # exit()
    STEPS = 5000  # 迭代次数
    # print(type(data_x))
    for i in range(STEPS):
        start = (i * batch_size) % data_size
        end = min(start + batch_size, data_size)
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: data_x[start:end,:], y_: Y[start:end,:]})

        # 每1000次迭代输出在所有数据上的交叉熵
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:data_x, y_:Y})
            print("第%d次迭代后的交叉熵为%f" % (i, total_cross_entropy))
    print("最终参数：")
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))

# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("../data/")
# print(mnist.train)
import numpy as np
import os
import matplotlib.pyplot as plt


class Mnist_NN(object):
    def __init__(self):
        self.train_x, self.train_y, self.validate_x, self.validate_y, self.test_x, self.test_y = self.load_mnist()
        self.input_node = 784  # 输入层节点个数
        self.output_node = 10  # 输出层节点个数
        self.hidden_node = 500  # 隐藏层节点个数
        self.batch_size = 100  # batch
        self.learning_rate_dacay = 0.99  # 学习率衰减系数
        self.moving_avg_decay = 0.99 # 滑动平均衰减系数
        self.global_step = tf.Variable(0, trainable=False)

        self.w1 = tf.Variable(tf.truncated_normal([self.input_node, self.hidden_node], dtype=tf.float32, stddev=0.1, seed=1))
        self.bias1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_node], dtype=tf.float32))
        self.w2 = tf.Variable(tf.truncated_normal([self.hidden_node, self.output_node], dtype=tf.float32, stddev=0.1, seed=2))
        self.bias2 = tf.Variable(tf.constant(0.1, shape=[self.output_node], dtype=tf.float32))
        self.train_step = 5000  # 迭代次数

    def load_mnist(self):
        train_x_path = "../data/train-images.idx3-ubyte"
        train_y_path = "../data/train-labels.idx1-ubyte"
        test_x_path = "../data/t10k-images.idx3-ubyte"
        test_y_path = "../data/t10k-labels.idx1-ubyte"

        with open(train_x_path) as f:
            # 利用np.fromfile语句将这个ubyte文件读取进来
            # X是从第16个字节开始的，前面的字节是文件信息
            loaded = np.fromfile(file=f, dtype=np.uint8)
            train_x = loaded[16:].reshape((60000, 28, 28)).astype(np.float)
            # print(train_x[0,:,:])

        with open(train_y_path) as f:
            loaded = np.fromfile(file=f, dtype=np.uint8)
            train_y = loaded[8:].reshape((60000, 1)).astype(np.float)

        with open(test_x_path) as f:
            loaded = np.fromfile(file=f, dtype=np.uint8)
            test_x = loaded[16:].reshape((10000, 28, 28)).astype(np.float)

        with open(test_y_path) as f:
            loaded = np.fromfile(file=f, dtype=np.uint8)
            test_y = loaded[8:].reshape((10000, 1)).astype(np.float)

        data_x = np.concatenate((train_x, test_x), axis=0)
        data_y = np.concatenate((train_y, test_y), axis=0)

        # 目的是为了打乱数据集
        # 这里随意固定一个seed，只要seed的值一样，那么打乱矩阵的规律就是一眼的
        seed = 666
        np.random.seed(seed)
        np.random.shuffle(data_x)
        np.random.seed(seed)
        np.random.shuffle(data_y)

        # 将标签转换为one-hot
        y = np.zeros((data_y.shape[0], 10))
        for i, label in enumerate(data_y):
            y[i, int(label)] = 1

        # 将数据分为训练集、验证集、测试集
        train_x = data_x[:55000, :].reshape(55000, 784)
        # test = train_x[0, :].reshape(28, 28)
        train_y = y[:55000, :].reshape(55000, 10)
        validate_x = data_x[55000:60000, :].reshape(5000, 784)
        validate_y = y[55000:60000, :].reshape(5000, 10)
        test_x = data_x[60000:70001, :].reshape(10000, 784)
        test_y = y[60000:70001, :].reshape(10000, 10)
        return train_x / 255.0, train_y, validate_x / 255.0, validate_y, test_x / 255.0, test_y

    # 前向传播
    def feedforward(self, input_tensor, w1, bias1, w2, bias2, avg_class=None):
        """
            前向传播
        :param input_tensor:
        :param w1:
        :param bias1:
        :param w2:
        :param bias2:
        :param avg_class: 是否对参数进行滑动平均
        :return:
        """
        if avg_class == None:
            hidden_output = tf.nn.relu(tf.matmul(input_tensor, w1) + bias1)
            y = tf.matmul(hidden_output, w2) + bias2
        else:
            hidden_output = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(w1)) + avg_class.average(bias1))
            y = tf.matmul(hidden_output, avg_class.average(w2)) + avg_class.average(bias2)

        return y

    # 定义损失函数
    def get_loss(self, y, y_):
        """

        :param y: 预测值
        :param y_: 真实值
        :return:
        """
        # 当每个样本只属于一类时，使用这个函数可以加速交叉熵的运算
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits()
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1), logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 当前batch中样本交叉熵的平均值

        regularizer = tf.contrib.layers.l2_regularizer(0.001)  # L2正则

        regularization = regularizer(self.w1) + regularizer(self.w2)

        return cross_entropy_mean + regularization

    # 训练
    def train(self):

        x = tf.placeholder(tf.float32, [None, self.input_node], name="input_x")
        y_ = tf.placeholder(tf.float32, [None, self.output_node], name="input_y")

        variable_avg = tf.train.ExponentialMovingAverage(self.moving_avg_decay, self.global_step)  # 滑动平均的类
        variable_avg_op = variable_avg.apply(tf.trainable_variables())  # 对所有的可以训练的变量执行滑动平均操作

        learning_rate = tf.train.exponential_decay(0.8, self.global_step, self.train_x.shape[0] / self.batch_size,
                                                        self.learning_rate_dacay)  # 指数衰减学习率
        y_pred = self.feedforward(x, self.w1, self.bias1, self.w2, self.bias2, None)  # 不对参数进行滑动平均

        # 对参数进行滑动平均
        y_avg = self.feedforward(x, self.w1, self.bias1, self.w2, self.bias2, variable_avg)


        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.get_loss(y_pred, y_),
                                                                               global_step=self.global_step)
        train_op = tf.group(train_step, variable_avg_op) #

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1)) # 不使用滑动平均
        # correct_prediction = tf.equal(tf.argmax(y_avg, 1), tf.argmax(y_, 1))  # 使用滑动平均
        accurate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:

            valid_feed = {x: self.validate_x, y_: self.validate_y}
            test_feed = {x: self.test_x, y_: self.test_y}
            sess.run(tf.global_variables_initializer())  # 初始化所有变量

            for i in range(self.train_step):
                if i * self.batch_size % self.train_x.shape[0] == 0 and i != 0: # 遍历完一遍, 重新打乱数据
                    np.random.seed(i)
                    np.random.shuffle(self.train_x)
                    np.random.seed(i)
                    np.random.shuffle(self.train_y)
                # print(sess.run(self.))
                if i % 1000 == 0:
                    # print(sess.run(self.w1))
                    valid_accurate = sess.run(accurate, feed_dict=valid_feed)
                    print("第%d次迭代后，验证集准确率为%g" % (i, valid_accurate))
                # 产生每一轮迭代的训练数据
                # print("训练样本个数：%d" % self.train_x.shape[0] )
                # exit()
                start = i * self.batch_size % self.train_x.shape[0]
                end = min(start + self.batch_size, self.train_x.shape[0])
                train_x_batch = self.train_x[start:end, :]
                train_y_batch = self.train_y[start:end, :]

                sess.run(train_op, feed_dict={x: train_x_batch, y_: train_y_batch})
                # sess.run(variable_avg_op)

            # 训练结束后，输出测试精度
            test_accurate = sess.run(accurate, feed_dict=test_feed)
            print("%d次迭代后，测试精度为%g" % (self.train_step, test_accurate))
            print(sess.run(variable_avg.average(self.w1)))



if __name__ == "__main__":
    mnist = Mnist_NN()
    mnist.train()
    # print(mnist.validate_y[1,:])
    # print(mnist.train_x[0,:].shape)
    # plt.imshow(mnist.validate_x[1,:].reshape(28,28))
    # plt.show()


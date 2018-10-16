# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import tensorflow as tf
import numpy as np

a = tf.Variable(tf.constant([[1,2]]), name="a")
b = tf.Variable(tf.constant([[2,3]]), name="b")
# a = tf.constant([[1,2]])
# b = tf.constant([[2,3]])
# print(a)
# exit()
result = a + b # 23232323
# print(result)
# print(tf.Session().run(result))
sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# with sess.as_default():
# sess.run(a.initializer)
# sess.run(b.initializer)
print(result.eval())
# print(tf.global_variables())
# print(a)
exit()
tf.clip_by_value()
# print(a.graph is tf.get_default_graph())
# exit()
# sess = tf.Session()
#
# print(sess.run(result))
# print(tf.get_default_graph())
print(tf.random_normal([2,3]))
exit()
weights = tf.Variable(tf.random_normal([2,3], stddev=1))
print(weights.initial_value)
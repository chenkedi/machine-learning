#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/30 下午12:45
# @Author  : chenekdi
# @Email   : chenkedi@baidu.com
import tensorflow as tf

feat_value = tf.Constant([
                            [1, 1, 25],
                            [1, 1, 26]
]) # 输入特征的value，m * F
feat_index = tf.Constant([
                    [1, 3, 5],
                    [2, 4, 5]
])

one_order_weight = tf.Constant([[0.1],
                               [0.2],
                               [0.3],
                               [0.4],
                               [0.5]]) # feat_size * 1

# 一阶特征unit
one_order = tf.nn.embedding_lookup(one_order_weight, feat_index) # 应该是 m * F * 1
with tf.Session() as sess:
    print(sess.run(one_order))





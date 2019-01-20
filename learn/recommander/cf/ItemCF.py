#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : chenekdi

from collections import defaultdict
from operator import itemgetter
import math
import time


class ItemCF:
    def __init__(self, train_set, K=50):
        self.train_set = train_set
        self.K = K
        self.W = None
        self.reverse_train = None

    def train_model(self):
        print("生成物品相似度矩阵....")
        start = time.time()
        self.sim()
        print(f"物品相似度矩阵生成完成，耗时{time.time() - start}")
        print("训练完成")

    def sim(self):
        """
        计算物品相似度矩阵
        :return:
        """
        # 先存储物品相似度的分子（即两个物品对应的用户列表交集的个数）
        # 再除以各自用户列表长度的乘积的平方根
        W = dict()

        # 存储每个物品对应的用户列表长度
        N = defaultdict(int)
        # 开始统计任意两个物品i和j之间用户列表交集的个数
        for user, items in self.train_set.items():
            for i in items.keys():
                N[i] += 1
                for j in items.keys():
                    if i == j:
                        continue
                    W.setdefault(i, defaultdict(int))
                    W[i][j] += 1

        for i, related_items in W.items():
            for j, intersection_count in related_items.items():
                if i == j:
                    continue
                W[i][j] /= math.sqrt(N[i] * N[j])

        self.W = W

    # def reverse_train(self):
    #     reverse_train = dict()
    #     for user, items:
    #         for item in items.keys():
    #             reverse_train.setdefault(item)

    def recommend(self, u, N):
        rank = dict()
        interest_items = self.train_set[u].keys()
        for i in interest_items:
            for j, wij in sorted(self.W[i].items(), key=itemgetter(1), reverse=True)[0:self.K]:
                if j in interest_items:
                    continue
                rank.setdefault(j, 0)
                rank[j] += wij
        return dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N])

    def recommends(self, users, N):
        recommends = dict()

        for user in users:
            recommends[user] = list(self.recommend(user, N).keys())
        return recommends

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/4 下午10:26
# @Author  : chenekdi
# @Email   : chenkedi@baidu.com

import numpy as np
import math
import time
from collections import defaultdict
from operator import itemgetter


class UserCF:

    def __init__(self, train_set, sim_type='IIF', K=80):
        """
        @:param type : 指定定义用户相似度度量的类型，IIF表示带流行物品惩罚度的用户相似度量
        @:param K : 选择前多少个与给定用户相似的候选用户进行推荐
        @:param train 训练数据集, {user:{item:rating}}
        """
        self.sim_type = sim_type
        self.K = K
        self.train_set = train_set
        self.invert_train = None
        self.W = None

    # todo attention!!!!!!!!!!!!
    # TODO 超级大坑，由于类变量有一个名为train，此处方法名也为train，所以出现了调用方法变成调用类变量的错误
    def train_model(self):
        print('生成物品倒排表......')
        start = time.time()
        self.invert_train = self.__invert_train()
        print(f'生成倒排表耗时:{time.time() - start}')

        print('生成用户相似度矩阵.....')
        start = time.time()
        self.W = self.sim()
        print(f'生成用户相似度矩阵耗时:{time.time() - start}')

        print('训练完成')

    def recommand(self, u, N):
        """
        根据给定的用户u，返回u的推荐列表
        :param u: 被推荐的用户
        :param N: 推荐的商品个数
        :return:
        """
        rank = dict()
        u_items = self.train_set[u]
        for v, wuv in sorted(self.W[u].items(), key=itemgetter(1), reverse=True)[0:self.K]:
            for item in self.train_set[v].keys():
                # 寻找K个最相似的用户v的物品列表中，不在用户u中的物品进行推荐评分
                if item in u_items:
                    continue
                # 将K个候选推荐用户v中，不在u的物品列表中的物品，进行累加评分（因为不同的v有可能有共同的物品i，这些得分需要累加）
                # 对于每个相似用户v，预测u对其物品列表中物品i的感兴趣度为：Wuv * rvi
                # 由于有K个相似用户v，所以u对某个一i的感兴趣程度的得分需要累加多个物品列表中有物品i的用户v的得分，所以这里的rank[item]后面是+=
                # rank[item] += wuv * rvi
                # 原文假设用的是隐性反馈数据，所以rvi都等于1
                rank.setdefault(item, 0)
                rank[item] += wuv
        return dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N])

    def recommands(self, users, N):
        """
        根据给定的用户u，返回u的推荐列表
        :param users: 需要推荐的所有用户
        :param N: 推荐的商品个数
        :return:
        """
        recommands = dict()
        for user in users:
            user_recommand = list(self.recommand(user, N).keys())
            recommands[user] = user_recommand
        return recommands

    def sim_raw(self):

        """
        使用原始方法求用户相似度矩阵, 复杂度为 M * M，M为用户数量
        :return:
        """
        # 理论上不会产生key error，因为推荐时是直接取W[u]的，只有与u有关系的u才能被取出
        W = dict()
        # todo 注意，书中使用的普通dict由于没有默认值，所以若取值到未赋值的部分有可能有KeyError的风险, 可以使用collections.defaultdict(float)来代替
        # 参考http://kodango.com/understand-defaultdict-in-python
        for u in self.train_set.keys():
            for v in self.train_set.keys():
                if u == v:
                    continue
                else:
                    if self.sim_type == 'IIF':
                        W[u][v] = self.__iif_sim(self.train_set[u], self.train_set[v])
                    else:
                        W[u][v] = self.__cos_sim(self.train_set[u], self.train_set[v])
        return W

    def sim(self):
        """
        因为整个用户-物品矩阵很稀疏，所以很多时候，两个用户的列表交集为0，sim_raw的循环中大量的时间耗费在了此种类型的相似度计算上
        :return:
        """
        # 任意两个用户u,v 物品交集的个数
        C = dict()
        # 某个用户产生行为的物品的个数
        # 注意，此处只能用defaultdict，因为对于任何元素，都是直接+=
        N = defaultdict(int)
        for item, userlist in self.invert_train.items():
            for u in userlist:
                N[u] += 1
                for v in userlist:
                    if u == v:
                        continue
                    else:
                        C.setdefault(u, defaultdict(float))
                        if self.sim_type == 'IIF':
                            # 此处对每个u,v相交的物品，均进行了流行度的惩罚（流行度就是一个物品被多少用户产生过行为）
                            # 所以，当u,v 共同的兴趣物品的流行度越高（也就是越热门时），相应的惩罚也越高
                            # 这样可以在一定程度上规避类似于《新华字典》这种大多数中国人小时候都买过的及其热门的物品对用户相似度造成的"伪增益"
                            C[u][v] += 1.0 / math.log10(1.0 + len(userlist))
                        else:
                            C[u][v] += 1
        # 其实这里的W和C可以共用
        # W = dict()
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                C[u][v] = cuv / math.sqrt(N[u] * N[v])
        return C

    def __cos_sim(self, u, v):
        """
        :param u: 用户u产生行为的物品列表
        :param v: 用户v产生行为的物品列表
        :return:
        """
        return len(u & v) / math.sqrt(len(u) * len(v) * 1.0)

    def __iif_sim(self, u, v):
        """
        改进的用户相似度定义，城府阿勒用户u和用户v共同兴趣列表中热门物品对他们相似度的影响
        :param u: 用户u产生行为的物品列表
        :param v: 用户v产生行为的物品列表
        :return:
        """
        join = np.array([len(self.invert_train[x]) for x in (u & v).keys()])

        nominator = np.sum(1.0 / np.log10(1.0 + join))

        return nominator / math.sqrt(len(u) * len(v))

    def __invert_train(self):
        """
        从（user，items）生成 倒排索引：（item， users）
        :return:
        """
        invert_train = dict()
        for user, items in self.train_set.items():
            for item in items.keys():
                # if item not in invert_train:
                #     invert_train[item] = set()
                # else:
                #     invert_train[item].add(user)
                # todo 上面的语句可以替代为
                invert_train.setdefault(item, set())
                invert_train[item].add(user)
        return invert_train

if __name__ == '__main__':
    import util.movielen_reader as mr
    movielens_train, movie_lens_test = mr.read_ratings("../data/ml-1m/ratings.dat", pivot=0.1)

    ucf = UserCF(movielens_train, sim_type='IIF')
    ucf.train_model()
    print(ucf.recommand('10' , 10))

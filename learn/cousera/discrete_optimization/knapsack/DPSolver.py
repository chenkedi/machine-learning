#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 上午12:15
# @Author  : chenekdi

from collections import defaultdict

class DPSolver:
    def __init__(self, item_count, capacity, items):
        self.item_count = item_count
        self.capacity = capacity
        self.items = items

    def solve(self):
        """

        :return:(value, weight, taken) taken是一个具有item个元素的数组
        """
        weight = 0
        value = 0
        taken = [0]*len(self.items)
        table = [[0 for i in range(self.item_count + 1)] for k in range(self.capacity + 1)]

        items_map = {item.index:item for item in self.items}

        for i in range(1, self.item_count + 1):
            wi = items_map[i].weight
            vi = items_map[i].value
            for k in range(1, self.capacity + 1):
                if wi <= k:
                    table[k][i] = max(table[k][i-1], vi + table[k - wi][i - 1])
                else:
                    table[k][i] = table[k][i-1]

        value = table[self.capacity][self.item_count]
        k = self.capacity
        for i in reversed(range(1, self.item_count + 1)):
            wi = items_map[i].weight
            if table[k][i] != table[k][i - 1]:
                taken[i - 1] = 1
                k = k - wi

        # print(table[self.capacity][self.item_count])
        # print(taken)

        return value, weight, taken



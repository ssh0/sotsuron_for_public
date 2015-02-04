#!/usr/bin/env python
# coding: utf-8

## model 3-4:近距離の点をクラスター化するモデル

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean as euc
import collections
import operator
import random
import bisect
from itertools import chain
from scipy.optimize import leastsq

__author__ = "Shotaro Fujimoto"


def uniq_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]

def accumulate(iterable, func=operator.add):
    """Return running totals

    Usage:
    accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def weighted_choice(d):
    choices, weights = zip(*d)
    cumdist = list(accumulate(weights))
    x = random.random() * cumdist[-1]
    return choices[bisect.bisect(cumdist, x)]

class Person:

    def __init__(self, master, id, ideas, w):
        """Initialize argmunets.

        Keyword arguments:
        master    : Master class (call from "Meeting")
        self.id   : Id for each person [0, 1, ..., N-1]
        self.ideas: ideas in space [0,1] × [0,1]
        self.w    : probability weight for the person to speak
        """
        self.id = id
        self.ideas = ideas
        self.w = w
        # add_ideas : place, tag : (x, y), [person_id, cluster_id]
        master.ideas += [[(i1, i2), [self.id, 0, self.w]] for i1, i2 in self.ideas]


class Cluster:

    def __init__(self, ideas, r):
        """make cluster with self.r

        cluster_link:
        """
        self.ideas = ideas
        self.r = r
        self.l = 0
        self.cluster_link = []
        self.clustering()

    def clustering(self):
        self.cell_num = int(1./self.r)
        lr = 1./self.cell_num

        self.cell = dict() # key: (cellx,celly), value: list of ids
        self.rcell = []
        for i, idea in enumerate(self.ideas):
            cellx = int(idea[0][0]/lr)
            celly = int(idea[0][1]/lr)
            if self.cell.has_key((cellx, celly)):
                self.cell[(cellx, celly)] += [i]
            else:
                self.cell[(cellx, celly)] = [i]
            self.rcell.append((cellx, celly))
        num = 1
        for i in range(len(self.ideas)):
            num += self.find_nearest(i, num)
        return self.cluster_link

    def find_nearest(self, idea_id, num):
        """find nearest idea

        idea_id: index in self.ideas
        """
        cx, cy = self.rcell[idea_id]
        place = self.ideas[idea_id][0]
        CX = uniq_list([max(0, cx - 1), cx, min(cx + 1, self.cell_num - 1)])
        CY = uniq_list([max(0, cy - 1), cy, min(cy + 1, self.cell_num - 1)])
        tmp = [self.cell[(i, j)] for i in CX for j in CY if self.cell.has_key((i, j))]
        tmp = list(chain.from_iterable(tmp))
        tmp.remove(idea_id)
        if len(tmp) == 0:
            self.ideas[idea_id][1][1] = num
            return 1

        nearest = []
        cid = [num]
        for k in tmp:
            if euc(self.ideas[k][0], place) > self.r:
                continue
            nearest.append(k)
            prenum = self.ideas[k][1][1]
            if prenum == 0:
                cid.append(num)
                self.cluster_link.append((idea_id, k))
            elif prenum < num:
                cid.append(prenum)
                if not (k, idea_id) in self.cluster_link:
                    self.cluster_link.append((idea_id, k))
        self.l += len(nearest)
        cluster_id = min(cid)
        if cluster_id < num:
            ans = 0
        else:
            ans = 1
        self.ideas[idea_id][1][1] = cluster_id
        for i in nearest:
            self.ideas[i][1][1] = cluster_id
        cid.remove(num)
        if len(cid) == 0:
            return ans
        cid.remove(cluster_id)
        if len(cid) == 0:
            return ans
        for i in cid:
            for x in self.ideas:
                if x[1][1] == i:
                    x[1][1] = cluster_id
        return ans


class Meeting:

    def __init__(self, K, N, S=20, r=0.06, draw=True):
        self.K = K
        self.N = N
        self.S = S
        self.r = r
        self.ideas = []
        self.minutes = []
        self.ave_l = 0
        self.draw = draw

    def gather_people(self, ideass=None, weights=None):
        """Gather participants.

        Keyword arguments:
        ideas  : list of ideas for each person
               ex) [((0.3,0.1),(0.2,0.5)), ((0.5,0.6))] when N = 2
        weights: list of weights for the probability of the person to speak
        """
        if not ideass:
            x = np.random.rand(self.N, self.S*2)
            ideass = []
            for _x in x:
                ideass.append([(i,j) for i,j in zip(_x[::2], _x[1::2])])
        if not weights:
            weights = [1.] * self.N
        for i, ideas, w in zip(range(self.N), ideass, weights):
            Person(self, i, ideas, w)

    def init(self):
        self.gather_people()
        cluster = Cluster(self.ideas, self.r)
        self.cluster_link = cluster.cluster_link
        self.ave_l = cluster.l/float(len(self.ideas))
        if self.draw:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            self.fig = plt.figure(figsize=(9, 9))
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.labels = []
            self.s1 = []
            for idea, tag in self.ideas:
                x = idea[0]
                y = idea[1]
                s = self.ax.scatter(x, y,
                                    c=colors[tag[0]%len(colors)],
                                    alpha=0.2)
                self.s1.append(s)
            data = []
            for link in self.cluster_link:
                ix = self.ideas[link[0]][0][0]
                iy = self.ideas[link[0]][0][1]
                jx = self.ideas[link[1]][0][0]
                jy = self.ideas[link[1]][0][1]
                data += [(ix, jx), (iy, jy), 'k']
            self.ax.plot(*data, alpha=0.5)

    def progress(self):
        self.init()
        preidea = self.ideas[np.random.choice(range(len(self.ideas)))]
        self.minutes.append(preidea)
        l = list(self.ideas)
        self.k = 1

        while self.k < self.K + 1:

            # remove ideas in the same cluster
            l = [idea for idea in l if idea[1][1] != preidea[1][1]]

            # if no one can speak: meeting ends.
            if len(l) == 0:
                break

            # confirm cluster id which is nearest from the preidea
            distance = [(euc(preidea[0], i[0]), i) for i in l]
            minclusterid = min(distance)[1][1][1]

            # gather ideas in the cluster
            tmp = [idea for idea in l if idea[1][1] == minclusterid]
            d = dict()
            for t in tmp:
                d[t[1][0]] = d.get(t[1][0], 0) + t[1][2]
            d = [(k, v) for k, v in d.items()]
            # chose whose ideas to be chosed from the cluster
            whois = weighted_choice(d)

            # gather ideas
            who = [idea for idea in tmp if idea[1][0] == whois]
            p = [(idea, idea[1][2]) for idea in who]
            # chose the next idea from the id is "whois"
            idea = weighted_choice(p)

            self.minutes.append(idea)
            preidea = idea
            self.callback()
            self.k += 1
        self.after()

    def callback(self):
        if self.draw:
            ix = self.minutes[-2][0][0]
            iy = self.minutes[-2][0][1]
            jx = self.minutes[-1][0][0]
            jy = self.minutes[-1][0][1]
            l1 = self.ax.plot([ix, jx], [iy, jy], color='b', alpha=0.5)
            self.ax.text((ix+jx)/2, (iy+jy)/2, self.k)
        else:
            pass

    def after(self):
        if self.draw:
            plt.show()
        else:
            pass


if __name__ == '__main__':
    meeting = Meeting(K=20, N=4, S=20, r=0.07, draw=True)
    meeting.progress()

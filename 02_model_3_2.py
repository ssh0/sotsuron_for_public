#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""02_model_3_2.py

See also 02_model_3_2.ipynb
"""
# 距離$d$の確率への依存性として計算g(d) = e^{-d}
from Tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import collections
import operator

__author__ = "Shotaro Fujimoto"


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


class Person(object):

    def __init__(self, ideas_num=10, place=(0., 0.), **kwargs):
        # 意見は0~1の間の値を一様に取りうる
        self.ideas = list(np.random.random(ideas_num))
        # 発言者の実際の位置が2次元の座標として表せる
        self.place = place
        # その他の特徴量
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def distance(self, p):
        # 人pと自分との間の距離(ユークリッド距離)
        d = np.sqrt(
            (self.place[0] - p.place[0]) ** 2 + (self.place[1] - p.place[1]) ** 2)
        return d


class meeting(object):

    def __init__(self, N):
        # 会議の参加人数
        self.N = N
        # 意見の時系列
        self.ideas = []
        # 発言者の時系列
        self.speaker = []
        # 時刻
        self.k = 0
        self.K = 100
        # 張られたリンク(時刻, 時刻)のタプルで表現する
        self.links = []
        # リンクの数(各時刻)
        self.l = [0]
        # リンクの数(累計)
        self.L = [0]

    def g(self, x):
        # 発言者の物理的距離に対する関数
        return np.exp(-x)

    def p(self, i):

        # 参加者の中で話せる人のみを対象に
        _N = []
        for k in range(1, self.N + 1):
            if len(self.members[k].ideas):
                _N.append(k)

        # それらの人たちに対し、関数gによる重み付けの確率を付与
        w = []
        for n in _N:
            d = self.members[n].distance(i)
            w.append(self.g(d))
        w = np.array(w)
        sum_ = np.sum(w)
        _p = list(w / sum_)
        p = list(accumulate(_p))
        rn = np.random.rand()
        nm = 0
        while True:
            if p[nm] > rn:
                break
            else:
                nm += 1
        # その確率で選ばれた人の名前を返す
        j = _N[nm]
        return j

    def q(self, j):
        # 発言者jが選ばれた時、持っている意見から等確率で意見を取り出す
        x_j = self.members[j]
        return np.random.rand()  # x_j.ideas.pop()

    def distance(self, x, y):
        # 意見の近さを絶対値で表現
        d = np.abs(x - y)
        if d == 0:
            return self.radius + 1
        else:
            return d

    def connect(self):
        l = 0
        for i, v in enumerate(self.ideas[:-1]):
            # k番目の意見と意見が近い時、それらノードの間にリンクを形成する
            if self.distance(v, self.ideas[self.k]) < self.radius:
                self.links.append((i, self.k))
                l += 1
        return l

    def check_agreement(self):
        # 合意チェック 参加人数Nによる関数
        def L(N):
            return N ** 2
        # if self.l[-1] > L(self.N):
        if self.k > self.K:
            return True
        else:
            return False

    def check_ideas(self):
        for k in range(1, self.N + 1):
            if len(self.members[k].ideas):
                return True
        return False

    def f_L(self):
        # リンクから会議の評価
        # 単純に会議終了時に得られたリンクの数を返す
        return self.L[-1]

    def f_T(self):
        # 会議に必要な時間の評価
        # 単純に必要な時間kを返す
        return self.k

    def f(self):
        # f_Lとf_Tを使った評価関数f
        return self.f_L() - self.f_T()

    def end(self):
        # 会議の通常終了、各定義量の計算や受け渡しなどはここで
        plt.ioff()
        plt.show()

        # ネットワーク図を描画
        link_s = [(a, b) for a, b in zip(self.speaker[:-1], self.speaker[1:])]
        counter_links = collections.Counter(link_s)
        for link, lw in counter_links.items():
            ix = self.members[link[0]].place[0]
            iy = self.members[link[0]].place[1]
            jx = self.members[link[1]].place[0]
            jy = self.members[link[1]].place[1]
            _x, _y = ((ix + jx) / 2, (iy + jy) / 2)
            if link[0] == link[1]:
                continue
            elif link[0] < link[1]:
                color = 'black'
                va = 'bottom'
            else:
                color = 'red'
                va = 'top'

            plt.plot([ix, jx], [iy, jy], color=color, lw=lw * 4 / self.k + 1)
            plt.text(_x, _y, '(%d,%d)' % (link[0], link[1]),
                     color=color, va=va)

        counter = collections.Counter(self.speaker)

        for key, i in self.members.items():
            x = i.place[0]
            y = i.place[1]
            size = counter[key] * 30
            plt.scatter(x, y, s=size)
            plt.text(x, y, str(key), color='green')
        plt.show()

        # 各時刻に追加されたリンク数のグラフ
        r = self.radius
        k = np.arange(self.k + 1)
        y = (-r ** 2 + 2 * r) * k
        delta = np.sqrt((-r ** 4 + 4 * r ** 3 - 5 * r ** 2 + 2 * r) * k)
        y1 = y + delta
        y2 = y - delta
        plt.fill_between(k, y1, y2, facecolor='green', alpha=0.2)
        plt.plot(k, self.l)
        plt.plot(k, y)
        plt.xlabel(r"Time: $k$")
        plt.ylabel(r"A number of edges for each time: $l$")
        plt.show()

        # リンク数の累積グラフ
        plt.plot(k, self.L)
        plt.plot(k, (-self.radius ** 2 + 2 * self.radius) * k ** 2 / 2.)
        plt.xlabel(r"Time: $k$")
        plt.ylabel(r"A number of edges: $L$")
        plt.show()

        # 時系列で発言者の表示
        # print 'self.speaker:', self.speaker

        # 評価関数を通した結果
        # print 'self.f', self.f()

    def end2(self):
        # 会議の異常終了(発言者が発言できなくなる)
        pass

    def init(self):
        x = [i.place[0] for i in self.members.values()]
        y = [i.place[1] for i in self.members.values()]
        plt.scatter(x, y)
        plt.ion()
        plt.draw()

    def callback(self):
        # print 'speaker:', self.speaker[-1]
        # print 'link:', self.l[-1]
        ix = self.members[self.speaker[-2]].place[0]
        iy = self.members[self.speaker[-2]].place[1]
        jx = self.members[self.speaker[-1]].place[0]
        jy = self.members[self.speaker[-1]].place[1]
        plt.plot([ix, jx], [iy, jy])
        plt.text((ix + jx) / 2, (iy + jy) / 2, '%d:(%d,%d)'
                 % (self.k, self.speaker[-2], self.speaker[-1]))
        plt.draw()

    def progress(self):
        self.init()
        # はじめに1が発言するとする
        self.ideas.append(self.q(1))
        self.speaker.append(1)
        while True:
            j = self.p(self.members[self.speaker[-1]])
            self.ideas.append(self.q(j))
            self.speaker.append(j)
            self.k += 1
            self.l.append(self.connect())
            self.L.append(len(self.links))
            self.callback()
            if self.check_agreement():
                print "\nnormal end"
                self.end()
                break
            if not self.check_ideas():
                print "\nno one can speak"
                self.end2()
                break


class Main:

    def __init__(self, radius=0.3):
        N = 6
        self.app = meeting(N)
        self.app.radius = radius
        window = Window(N, main=self.app)
        window.display()


class Window(object):

    def __init__(self, N, main):
        self.root = Tk()
        self.main = main
        self.width = 640
        self.height = 480
        self.canvas = Canvas(self.root, width=self.width, height=self.height)
        self.var = StringVar()
        self.oval(self.canvas, N)
        self.canvas.bind('<Motion>', self.pointer)
        self.canvas.pack()
        label = Label(self.root, textvariable=self.var, font='Ubuntu 9')
        label.pack(side='left')
        b1 = Button(self.root, text='start', command=self.b1_clicked)
        b1.pack(side='right')
        b2 = Button(self.root, text='save', command=self.b2_clicked)
        b2.pack(side='right')

    def oval(self, canvas, N=6):
        self.members = dict()
        deg = np.linspace(0., 360., N, endpoint=False)
        radius = 20
        self.r = int((min(self.height, self.width) / 2 - radius) * 0.9)
        self.centerx = int(self.width / 2)
        self.centery = int(self.height / 2)
        for n in range(1, N + 1):
            rad = np.radians(deg[n - 1])
            self.members[n] = Oval(canvas, n,
                                   self.centerx + self.r * np.cos(rad),
                                   self.centery + self.r * np.sin(rad),
                                   radius, self.var)

    def pointer(self, event):
        self.var.set("(%d,%d)" % (event.x, event.y))

    def b1_clicked(self):
        self.main.members = dict()
        for n in range(1, self.main.N + 1):
            x = (self.members[n].x - self.centerx) / float(self.r)
            y = (self.members[n].y - self.centery) / float(self.r)
            self.main.members[n] = Person(place=(x, y))
        self.main.progress()

    def b2_clicked(self):
        import tkFileDialog
        import os

        fTyp = [('eps file', '*.eps'), ('all files', '*')]
        filename = tkFileDialog.asksaveasfilename(filetypes=fTyp,
                                                  initialdir=os.getcwd(),
                                                  initialfile='figure_1.eps')

        if filename is None:
            return
        try:
            self.canvas.postscript(file=filename)
        except TclError:
            print """
            TclError: Cannot save the figure.
            Canvas Window must be alive for save."""
            return 1

    def display(self):
        self.root.mainloop()


class Oval:

    def __init__(self, canvas, id, x, y, r, var):
        self.c = canvas
        self.x = x
        self.y = y
        self.var = var
        self.tag = str(id)
        self.c.create_oval(
            x - r, y - r, x + r, y + r, outline='', fill='#069', tags=self.tag)

        self.c.tag_bind(self.tag, '<Button-1>', self.pressed)
        self.c.tag_bind(self.tag, '<Button1-Motion>', self.dragging)

    def pressed(self, event):
        self.x = event.x
        self.y = event.y

    def dragging(self, event):
        self.c.move(self.tag, event.x - self.x, event.y - self.y)
        self.x = event.x
        self.y = event.y


if __name__ == '__main__':
    main = Main(radius=1 / 3.)

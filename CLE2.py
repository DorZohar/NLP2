from copy import deepcopy
from heapq import heappush, heappop, heapify
from time import time

import numpy as np



class vertices:
    def __init__(self, n, G):
        self.in_arr = {}
        self.const = {i: 0.0 for i in range(1, n)}
        self.prev = {}
        self.parent = {}
        self.children = {i: [] for i in range(1, n)}
        self.P = {i: [] for i in range(1, n)}
        for u, neighbors in G.items():
            for v, w in neighbors.items():
                heappush(self.P[v], (w, (u, v)))

    def add_vertex(self, v):
        self.const[v] = 0.0
        self.children[v] = []
        self.P[v] = []



def find(vert, u):
    while u in vert.parent:
        u = vert.parent[u]
    return u


def dismantle(vert, u, R):
    while u in vert.parent:
        for v in vert.children[vert.parent[u]]:
            if v != u:
                del vert.parent[v]
                if vert.children[v]:
                    R.append(v)
        u = vert.parent[u]

    return R, vert


def weight(G, vert, u, v):
    w = G[u][v]
    while v in vert.parent:
        w += vert.const[v]
        v = vert.parent[v]

    return w


def mst(r, G):
    n = len(G)
    root_neighbors = G[0]
    del G[0]
    vert = vertices(n+1, G)
    a = 1 # np.random.choice(list(G.keys()))
    while vert.P[a]:
        w, (u, v) = heappop(vert.P[a])
        b = find(vert, u)
        if a != b:
            vert.in_arr[a] = (u, v)
            vert.prev[a] = b
            if u not in vert.in_arr:
                a = b
            else:
                c = n
                n += 1
                vert.add_vertex(c)
                while a not in vert.parent:
                    vert.parent[a] = c
                    vert.in_arr[c] = vert.in_arr[a]
                    s, t = vert.in_arr[a]
                    vert.const[a] = -weight(G, vert, s, t)
                    vert.children[c].append(a)
                    vert.P[c] += vert.P[a]
                    a = vert.prev[a]
                heapify(vert.P[c])
                a = c

    min_sum = float('inf')

    for ne, we in root_neighbors.items():
        R = []
        temp_vert = deepcopy(vert)
        R, temp_vert = dismantle(temp_vert, ne, R)
        sum = we
        while R:
            c = R.pop()
            u, v = temp_vert.in_arr[c]
            temp_vert.in_arr[v] = (u, v)
            R, temp_vert = dismantle(temp_vert, v, R)

        new_G = {i: {} for i in range(1, len(G) + 1)}
        for u, (s, t) in temp_vert.in_arr.items():
            if u != ne and u in G:
                new_G[s][t] = G[s][t]
                sum += G[s][t]

        if sum <= min_sum:
            min_sum = sum
            new_G[0] = {ne: we}
            min_tree = new_G


    return min_tree


if __name__ == "__main__":
    print(mst(0, {0: {1: 0.0, 2: 0, 3: 0}, 1: {2: 0, 3: 0}, 2: {1: 0, 3: 0}, 3: {1: 0, 2: 0}}))

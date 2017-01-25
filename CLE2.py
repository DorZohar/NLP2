from heapq import heappush, heappop, heapify

import numpy as np



class vertices:
    def __init__(self, n, G):
        self.in_arr = {}
        self.const = {i: 0 for i in range(n)}
        self.prev = {}
        self.parent = {}
        self.children = {i: [] for i in range(n)}
        self.P = {i: [] for i in range(n)}
        for u, neighbors in G.items():
            for v, w in neighbors.items():
                heappush(self.P[v], (-w, (u, v)))

    def add_vertex(self, v):
        self.const[v] = 0
        self.children[v] = []
        self.P[v] = []


def find(vert, u):
    while u in vert.parent:
        u = vert.parent[u]
    return u


def dismantle(vert, r, R):
    while r in vert.parent:
        r = vert.parent[r]
        for v in vert.children[r]:
            del vert.parent[v]
            if vert.children[v]:
                R.append(v)

    return R, vert


def weight(G, vert, u, v):
    w = G[u][v]
    while v in vert.parent:
        w += vert.const[v]
        v = vert.parent[v]

    return w


def mst(r, G):
    n = len(G)
    vert = vertices(n, G)
    a = 0  # np.random.choice(list(G.keys()))
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
                while a not in vert.parent and a in vert.in_arr:
                    vert.parent[a] = c
                    s, t = vert.in_arr[a]
                    vert.const[a] = -weight(G, vert, s, t)
                    vert.children[c].append(a)
                    vert.P[c] += vert.P[a]
                    heapify(vert.P[c])
                    a = vert.prev[a]

    R = []
    R, vert = dismantle(vert, r, R)
    while R:
        c = R.pop()
        u, v = vert.in_arr[c]
        vert.in_arr[v] = (u, v)
        R, vert = dismantle(vert, r, R)

    new_G = {i: {} for i in range(len(G))}
    for u, (s, t) in vert.in_arr.items():
        if u != r:
            new_G[s][u] = G[s][u]

    return new_G


if __name__ == "__main__":
    print(mst(0, {0: {1: 1, 2: 2, 3: 1}, 1: {0: 1, 2: 1, 3: 1}, 2: {0: 5, 1: 0, 3: 2}, 3: {}}))

import copy
import numpy as np
from multiprocessing import Pool

def reverse(G):
    # reverse the direction of the edges in the graph for conevnience
    reversed_G = {}
    for src in G:
        neighbours = G[src]
        for dst in neighbours:
            if dst not in reversed_G:
                reversed_G[dst] = {}
            reversed_G[dst][src] = neighbours[dst]
    return reversed_G


def keep_only_max_parents(G):
    possible_tree = {}
    reversed_G = reverse(G)
    for node in reversed_G:
        parent = -1
        min_weight = float("inf")
        for src in reversed_G[node]:
            weight = reversed_G[node][src]
            if weight < min_weight:
                parent = src
                min_weight = weight
        if parent not in possible_tree:
            possible_tree[parent] = {}
        possible_tree[parent][node] = min_weight

    return possible_tree


def dfs(node, G, visited, cycle):
    if node in visited:
        return cycle

    visited.append(node)
    curr_cycle = list(cycle)
    curr_cycle += [node]

    if node in G:
        for child in G[node]:
            new_cycle = dfs(child, G, visited, curr_cycle)
            if new_cycle != []:
                return new_cycle

    return []


def find_cycle(G):
    for node in G:
        cycle = set(dfs(node, G, [], []))
        if cycle != set([]):
            return cycle
    return []


def contract_cycle(G, cycle):
    contracted_G = {}

    # build a sub-graph including only the cycle and find the maximum edge in it
    cycle_graph = {}
    max_edge = (-1, -1)
    max_weight = -float("inf")
    for src in cycle:
        cycle_graph[src] = {}
        for dst in G[src]:
            if dst in cycle:
                weight = G[src][dst]
                if weight > max_weight:
                    max_edge = (src, dst)
                    max_weight = weight
                cycle_graph[src][dst] = weight

    # keep all nodes outside of the cycle
    for src in G:
        for dst in G[src]:
            if src not in cycle or dst not in cycle:
                if src not in contracted_G:
                    contracted_G[src] = {}
                contracted_G[src][dst] = G[src][dst]

    # keep min outgoing edges
    # all nodes x where c->x is an edge and c is in cycle
    candidates = [node for node in set(sum([G[key].keys() for key in cycle],[])) if node not in cycle]
    for dst in candidates:
        min_weight = float("inf")
        min_edge = -1
        for src in cycle:
            if dst in G[src] and G[src][dst] < min_weight:
                min_weight = G[src][dst]
                min_edge = (src, dst)
        if min_edge[0] not in contracted_G:
            contracted_G[min_edge[0]] = {}
        contracted_G[min_edge[0]][min_edge[1]] = min_weight

    # caclculate s(C)
    s_cycle = 0
    for dst in cycle:
        for src in cycle:
            if dst != src and dst in G[src]:
                s_cycle += G[src][dst]

    # keep min incoming edges
    for src in G:
        if src not in cycle:
            min_weight = float("inf")
            min_dst = -1
            for dst in G[src]:
                if dst in cycle:
                    internal_parent_weight = 0
                    for node in cycle:
                        if dst in G[node]:
                            internal_parent_weight = G[node][dst]
                            break
                    weight = G[src][dst] - internal_parent_weight
                    if weight < min_weight:
                        min_weight = weight
                        min_dst = dst
            if min_dst != -1:
                if src not in contracted_G:
                    contracted_G[src] = {}
                contracted_G[src][min_dst] = min_weight + s_cycle

    # add the mst of the cycle
    for src in cycle_graph:
        for dst in cycle_graph[src]:
            if (src, dst) != max_edge:
                if src not in contracted_G:
                    contracted_G[src] = {}
                contracted_G[src][dst] = cycle_graph[src][dst]

    return contracted_G


def replace_weights(dst_G, src_G):
    for src in dst_G:
        for dst in dst_G[src]:
            dst_G[src][dst] = src_G[src][dst]

    return dst_G


def mst(G):
    while True:
        possible_tree = keep_only_max_parents(G)
        cycle = list(find_cycle(possible_tree))
        if cycle != []:
            new_G = contract_cycle(G, cycle)
            G = replace_weights(new_G, G)
        else:
            return possible_tree


def graph_weight(G):
    res = 0
    for src in G:
        for dst in G[src]:
            res += G[src][dst]
    return res


def graphs_with_single_edge_from_root(G, root, node):
    G_temp = copy.deepcopy(G)
    G_temp[root] = {node: G_temp[root][node]}
    return G_temp
#
# def mst(root, G):
#     root_nodes = G[root]
#     #p = Pool(56)
#     Gs = map(lambda node: graphs_with_single_edge_from_root(G, root, node), root_nodes)
#     trees = map(_mst, Gs)
#     vals = map(graph_weight, trees)
#     min_idx = np.argmin(vals)
#
#     return trees[min_idx]


def plot(G, name='graph'):
    from graphviz import Digraph

    dot = Digraph()

    for src in G:
        for dst in G[src]:
            dot.node(str(dst), str(dst))
            dot.edge(str(src), str(dst))

    dot.render(name, view=False)


if __name__ == '__main__':
    G = {'root': {'saw': -10, 'Mary': -9, 'John': -9}, 'Mary': {'saw': 0, 'John': -11}, 'saw': {'John': -30, 'Mary': -30}, 'John': {'saw': -20, 'Mary': -3}}
    #G = {0: {1: -38.0, 2: -19.0, 3: -11.0, 4: -17.0, 5: -3.0, 6: -3.0, 7: -19.0, 8: -3.0, 9: -19.0, 10: -19.0, 11: -19.0, 12: -19.0, 13: -51.0, 14: -3.0, 15: -35.0, 16: -23.0, 17: -3.0, 18: -19.0, 19: -15.0, 20: -7.0, 21: -21.0, 22: -3.0, 23: -3.0, 24: 5.0}, 1: {2: -32.0, 3: 2.0, 4: -15.0, 5: 4.0, 6: -11.0, 7: -24.0, 8: -5.0, 9: -18.0, 10: -4.0, 11: -4.0, 12: -4.0, 13: 6.0, 14: 8.0, 15: -10.0, 16: 8.0, 17: -7.0, 18: -13.0, 19: -2.0, 20: -3.0, 21: 7.0, 22: -8.0, 23: -14.0, 24: -28.0}, 2: {1: -14.0, 3: -26.0, 4: 8.0, 5: 7.0, 6: -30.0, 7: -15.0, 8: -0.0, 9: 1.0, 10: 1.0, 11: -9.0, 12: 1.0, 13: 3.0, 14: 6.0, 15: 3.0, 16: 6.0, 17: -39.0, 18: -0.0, 19: -21.0, 20: -0.0, 21: 4.0, 22: 4.0, 23: 2.0, 24: -1.0}, 3: {1: -2.0, 2: -21.0, 4: -8.0, 5: -0.0, 6: -8.0, 7: -21.0, 8: -0.0, 9: -14.0, 10: -14.0, 11: -14.0, 12: -14.0, 13: -0.0, 14: -13.0, 15: -16.0, 16: -39.0, 17: -8.0, 18: -21.0, 19: -5.0, 20: -14.0, 21: -13.0, 22: -10.0, 23: -17.0, 24: -7.0}, 4: {1: -27.0, 2: -26.0, 3: -25.0, 5: -53.0, 6: -21.0, 7: -26.0, 8: -32.0, 9: -10.0, 10: -10.0, 11: -10.0, 12: -10.0, 13: -16.0, 14: -13.0, 15: -22.0, 16: -20.0, 17: -25.0, 18: -26.0, 19: -22.0, 20: -28.0, 21: -22.0, 22: -15.0, 23: -15.0, 24: -8.0}, 5: {1: -2.0, 2: -2.0, 3: -10.0, 4: -8.0, 6: -2.0, 7: -2.0, 8: -2.0, 9: -2.0, 10: -2.0, 11: -2.0, 12: -2.0, 13: -2.0, 14: -2.0, 15: -2.0, 16: -2.0, 17: -2.0, 18: -2.0, 19: -2.0, 20: -2.0, 21: -2.0, 22: -2.0, 23: -2.0, 24: -2.0}, 6: {1: 11.0, 2: 10.0, 3: 1.0, 4: 16.0, 5: 10.0, 7: 10.0, 8: 10.0, 9: 10.0, 10: 10.0, 11: 10.0, 12: 10.0, 13: 5.0, 14: 10.0, 15: 1.0, 16: 4.0, 17: 9.0, 18: 10.0, 19: -2.0, 20: 10.0, 21: 10.0, 22: 5.0, 23: -0.0, 24: 10.0}, 7: {1: -25.0, 2: -18.0, 3: -13.0, 4: -0.0, 5: -1.0, 6: -43.0, 8: -18.0, 9: -7.0, 10: -7.0, 11: -17.0, 12: -7.0, 13: -5.0, 14: -2.0, 15: -26.0, 16: -2.0, 17: -52.0, 18: -8.0, 19: -32.0, 20: -8.0, 21: -4.0, 22: -4.0, 23: -6.0, 24: -9.0}, 8: {1: -3.0, 2: -3.0, 3: -3.0, 4: -3.0, 5: -3.0, 6: -3.0, 7: -3.0, 9: -3.0, 10: -3.0, 11: -3.0, 12: -3.0, 13: -3.0, 14: -3.0, 15: -3.0, 16: -17.0, 17: -3.0, 18: -3.0, 19: -3.0, 20: -3.0, 21: -3.0, 22: -3.0, 23: -3.0, 24: -3.0}, 9: {1: -2.0, 2: 2.0, 3: 15.0, 4: 9.0, 5: 5.0, 6: -23.0, 7: 2.0, 8: -6.0, 10: -30.0, 11: -38.0, 12: -17.0, 13: 4.0, 14: 13.0, 15: 22.0, 16: 17.0, 17: -9.0, 18: 2.0, 19: -16.0, 20: -1.0, 21: 10.0, 22: 11.0, 23: 11.0, 24: -7.0}, 10: {1: 17.0, 2: 4.0, 3: 17.0, 4: 11.0, 5: 7.0, 6: -21.0, 7: 4.0, 8: -4.0, 9: -32.0, 11: -28.0, 12: -7.0, 13: 6.0, 14: 15.0, 15: 24.0, 16: 19.0, 17: -7.0, 18: 4.0, 19: 3.0, 20: 1.0, 21: 12.0, 22: 13.0, 23: 13.0, 24: -5.0}, 11: {1: 17.0, 2: 4.0, 3: 17.0, 4: 11.0, 5: 7.0, 6: -21.0, 7: 4.0, 8: -4.0, 9: -32.0, 10: -20.0, 12: -7.0, 13: 6.0, 14: 15.0, 15: 24.0, 16: 19.0, 17: -7.0, 18: 4.0, 19: 3.0, 20: 1.0, 21: 12.0, 22: 13.0, 23: 13.0, 24: -5.0}, 12: {1: 17.0, 2: 4.0, 3: 17.0, 4: 11.0, 5: 7.0, 6: -21.0, 7: 4.0, 8: -4.0, 9: -32.0, 10: -20.0, 11: -28.0, 13: 6.0, 14: 15.0, 15: 24.0, 16: 19.0, 17: -7.0, 18: 4.0, 19: 3.0, 20: 1.0, 21: 12.0, 22: 13.0, 23: 13.0, 24: -5.0}, 13: {1: -64.0, 2: -43.0, 3: -15.0, 4: -24.0, 5: -30.0, 6: -15.0, 7: -49.0, 8: -32.0, 9: -29.0, 10: -29.0, 11: -29.0, 12: -29.0, 14: -66.0, 15: -55.0, 16: -17.0, 17: -15.0, 18: -26.0, 19: -16.0, 20: -22.0, 21: -18.0, 22: -13.0, 23: -16.0, 24: -80.0}, 14: {1: 4.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: -9.0, 7: -1.0, 8: -4.0, 9: -7.0, 10: -7.0, 11: -7.0, 12: -7.0, 13: -29.0, 15: -1.0, 16: -1.0, 17: -9.0, 18: -1.0, 19: -5.0, 20: 3.0, 21: -1.0, 22: -1.0, 23: -1.0, 24: -1.0}, 15: {1: -43.0, 2: -22.0, 3: -5.0, 4: -15.0, 5: -11.0, 6: -10.0, 7: -22.0, 8: -25.0, 9: -9.0, 10: -9.0, 11: -9.0, 12: -9.0, 13: -25.0, 14: -55.0, 16: -44.0, 17: -10.0, 18: -22.0, 19: -5.0, 20: -17.0, 21: -11.0, 22: -3.0, 23: -3.0, 24: -47.0}, 16: {1: -18.0, 2: -26.0, 3: -37.0, 4: -26.0, 5: -24.0, 6: 2.0, 7: -26.0, 8: -17.0, 9: -3.0, 10: -3.0, 11: -3.0, 12: -3.0, 13: -5.0, 14: -11.0, 15: -5.0, 17: 2.0, 18: -52.0, 19: -10.0, 20: -10.0, 21: -9.0, 22: -7.0, 23: -7.0, 24: -14.0}, 17: {1: 5.0, 2: 13.0, 3: 4.0, 4: 19.0, 5: 13.0, 6: 12.0, 7: 13.0, 8: 13.0, 9: 13.0, 10: 13.0, 11: 13.0, 12: 13.0, 13: 8.0, 14: 13.0, 15: 4.0, 16: 7.0, 18: 13.0, 19: -8.0, 20: 13.0, 21: 13.0, 22: 8.0, 23: 3.0, 24: 13.0}, 18: {1: -47.0, 2: -33.0, 3: -38.0, 4: -15.0, 5: -16.0, 6: -51.0, 7: -38.0, 8: -23.0, 9: -22.0, 10: -22.0, 11: -32.0, 12: -22.0, 13: -20.0, 14: -17.0, 15: -20.0, 16: -17.0, 17: -60.0, 19: -54.0, 20: -23.0, 21: -19.0, 22: -19.0, 23: -21.0, 24: -24.0}, 19: {1: 8.0, 2: -42.0, 3: -2.0, 4: -21.0, 5: -0.0, 6: -11.0, 7: -34.0, 8: -10.0, 9: -25.0, 10: -11.0, 11: -11.0, 12: -11.0, 13: -1.0, 14: -13.0, 15: -15.0, 16: 4.0, 17: -7.0, 18: -23.0, 20: -11.0, 21: -11.0, 22: -12.0, 23: -18.0, 24: -7.0}, 20: {1: -7.0, 2: 4.0, 3: -13.0, 4: -0.0, 5: -0.0, 6: -20.0, 7: 4.0, 8: -2.0, 9: 4.0, 10: 4.0, 11: 4.0, 12: 4.0, 13: -0.0, 14: -2.0, 15: -9.0, 16: 2.0, 17: -26.0, 18: 4.0, 19: -12.0, 21: -3.0, 22: -23.0, 23: -31.0, 24: -5.0}, 21: {1: -13.0, 2: -2.0, 3: -21.0, 4: -8.0, 5: -2.0, 6: -27.0, 7: -2.0, 8: -9.0, 9: 8.0, 10: 8.0, 11: 8.0, 12: 8.0, 13: -3.0, 14: -2.0, 15: -2.0, 16: -2.0, 17: -26.0, 18: -2.0, 19: -27.0, 20: -16.0, 22: -35.0, 23: -56.0, 24: -20.0}, 22: {1: 4.0, 2: 6.0, 3: -2.0, 4: -2.0, 5: -7.0, 6: -22.0, 7: 6.0, 8: -14.0, 9: 3.0, 10: 3.0, 11: 3.0, 12: 3.0, 13: -2.0, 14: 1.0, 15: -3.0, 16: -0.0, 17: -19.0, 18: 6.0, 19: -3.0, 20: -9.0, 21: -0.0, 23: -15.0, 24: 6.0}, 23: {1: -8.0, 2: 6.0, 3: -12.0, 4: -2.0, 5: -7.0, 6: -24.0, 7: 6.0, 8: -14.0, 9: -11.0, 10: -11.0, 11: -11.0, 12: -11.0, 13: -2.0, 14: 1.0, 15: -15.0, 16: -0.0, 17: -21.0, 18: 6.0, 19: -15.0, 20: -9.0, 21: -0.0, 22: -30.0, 24: 6.0}, 24: {1: 8.0, 2: 8.0, 3: 8.0, 4: 8.0, 5: 8.0, 6: 8.0, 7: 8.0, 8: 8.0, 9: 2.0, 10: 2.0, 11: 2.0, 12: 2.0, 13: 8.0, 14: 8.0, 15: 8.0, 16: 8.0, 17: 8.0, 18: 8.0, 19: 8.0, 20: 8.0, 21: 8.0, 22: 8.0, 23: 8.0}}
    #ref = {0: {13: -51.0}, 4: {8: -32.0, 20: -28.0, 5: -53.0, 21: -22.0}, 9: {10: -30.0, 11: -38.0}, 10: {9: -32.0}, 13: {1: -64.0, 2: -43.0, 7: -49.0, 12: -29.0, 14: -66.0, 15: -55.0, 24: -80.0}, 15: {16: -44.0}, 16: {18: -52.0, 4: -26.0}, 18: {19: -54.0, 17: -60.0, 3: -38.0, 6: -51.0}, 21: {22: -35.0, 23: -56.0}}
    #plot(G)
    res = mst('root', G)
    #plot(res)
    print(res)
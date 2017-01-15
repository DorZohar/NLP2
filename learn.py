import time

from chu_liu import mst
from features import parse_input_file, root_word, get_edge_features, total_feature_num
import numpy as np
import os


def sentence_to_graph(sentence, families):
    graph = {0: {}}
    for key in sentence.keys():
        if key != 0:
            graph[key] = {}
            graph[0][key] = get_edge_features(sentence[key], root_word, sentence, families)

    for child in sentence.keys():
        for parent in sentence.keys():
            if child == parent or child == 0 or parent == 0:
                continue
            graph[parent][child] = get_edge_features(sentence[child], sentence[parent], sentence, families)

    return graph


def sentence_to_features(sentence, families):
    features = []
    sentence[0] = root_word
    for idx, word in sentence.items():
        if idx == 0:
            continue
        features += get_edge_features(word, sentence[word.parent], sentence, families)

    return features


def graph_to_features(graph, sentence, families):
    features = []
    for vertex, edges in graph.items():
        for neigh in edges:
            features += get_edge_features(sentence[neigh], sentence[vertex], sentence, families)

    return features


def get_weighted_graph(graph, vec):
    weighted_graph = {}
    for vertex, neigh in graph.items():
        weighted_graph[vertex] = {child: -np.sum(vec[indices]) for child, indices in neigh.items()}

    return weighted_graph


def compare_labels_to_graph(sentence, graph):

    for key, word in sentence.items():
        if key != 0 and (word.parent not in graph or key not in graph[word.parent]):
            return False

    return True


def vector_location(families):
    return "vectors/%s" % '_'.join([str(family) for family in families])


def keep_vector(vec, families, iter):
    dir = vector_location(families)
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_name = dir + "/vector_%d.py" % iter
    with open(file_name, "w") as f:
        f.write("vec = %s\n" % vec.tolist())


def perceptron(file_path, iterations, families):
    n = max(iterations) + 1
    start_time = time.time()
    sentences = parse_input_file(file_path)
    graphs = list(map(lambda sentence: (sentence, sentence_to_features(sentence, families), sentence_to_graph(sentence, families)), \
                 sentences))
    vec = np.zeros(total_feature_num(families))
    print("%d sentences\n" % len(sentences))

    vecs_by_iter = {}

    for i in range(0, n):
        print("enter %d iteration at %f" % (i, time.time() - start_time))
        for sentence, features, graph in graphs:
            weighted_graph = get_weighted_graph(graph, vec)
            graph_mst = mst(0, weighted_graph)
            if not compare_labels_to_graph(sentence, graph_mst):
                graph_features = graph_to_features(graph_mst, sentence, families)
                for feature in features:
                    vec[feature] += 1
                for feature in graph_features:
                    vec[feature] -= 1
        if i in iterations:
            vecs_by_iter[i] = vec.tolist()
            keep_vector(vec, families, i)

    return vecs_by_iter


if __name__ == "__main__":
    perceptron("train.labeled", [5], [1, 2, 3, 4, 5, 6, 8, 10, 13, 15, 16, 17, 18, 19, 20])

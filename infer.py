import numpy as np

from chu_liu import mst
from features import parse_input_file
from learn import sentence_to_graph, get_weighted_graph
from vector import vec


def get_inference_accuracy(sentence_and_tree):
    sentence, tree = sentence_and_tree
    sum = 0
    for key, word in sentence.items():
        if word.parent in tree and key in tree[word.parent]:
            sum += 1

    return sum


def infer(file_path, families):
    sentences = parse_input_file(file_path)[1:-1]
    word_num = sum([len(sentence) for sentence in sentences])
    graphs = map(lambda sentence: sentence_to_graph(sentence, families), sentences)
    selected_vec = np.asarray(vec[len(vec) - 1])
    weighted_graphs = list(map(lambda graph: get_weighted_graph(graph, selected_vec), graphs))
    msts = map(lambda wgraph: mst(0, wgraph), weighted_graphs)
    accuracies = map(get_inference_accuracy, zip(sentences, msts))

    total_acc = sum(list(accuracies)) / word_num
    print(total_acc)



if __name__ == "__main__":
    infer("test.labeled", [1, 2, 3, 4, 5, 6, 8, 10, 13, 14, 15, 16])

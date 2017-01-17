import numpy as np
import time
from chu_liu import mst
from features import parse_input_file
from learn import sentence_to_graph, get_weighted_graph, perceptron, vector_location
from CLE import plot, graphs_with_single_edge_from_root, graph_weight
import argparse
import imp
#from vector import vec

from multiprocessing import Pool


def get_fails(sentences, msts, save_path, num_fails=10):
    count = 0

    fails = []
    for sentence, tree in zip(sentences, msts):

        # get fails and gold
        sentence_fails = {}
        gold_tree = {}
        gold_tree_verbal = {}
        for key, word in sentence.items():
            # tree with words
            if word.parent == 0:
                parent_word = '0.root'
            else:
                parent_word = str(word.parent) + '.' + sentence[word.parent].word
            if parent_word not in gold_tree_verbal:
                gold_tree_verbal[parent_word] = {}
            gold_tree_verbal[parent_word][str(word.idx) + '.' + word.word] = 1
            # tree with indices
            if word.parent not in gold_tree:
                gold_tree[word.parent] = {}
            gold_tree[word.parent][key] = 1

            if word.parent in tree and key in tree[word.parent]:
                pass
            else:
                sentence_fails[word.parent] = key

        fails += [sentence_fails]

        # plot result and gold
        if sentence_fails != {}:
            plot(tree, save_path + '/' + str(count))
            plot(gold_tree, save_path + '/' + str(count) + '_gold')
            plot(gold_tree_verbal, save_path + '/' + str(count) + '_gold_verbal')

            count += 1
        if count >= num_fails:
            break

    return fails


def get_inference_accuracy(sentence_and_tree):
    sentence, tree = sentence_and_tree
    sum = 0
    for key, word in sentence.items():
        if word.parent in tree and key in tree[word.parent]:
            sum += 1

    return sum


def infer_only(file_path, families, plot_fails=False, iter=-1, write_results=False):
    if iter == -1:
        iter = [20, 50, 80, 100]
    else:
        iter = [iter]
    results = {}
    for num_iter in iter:
        print("num iterations: " + str(num_iter))
        module = imp.load_source('vector', vector_location(families) + '/vector_' + str(num_iter) + '.py')
        results[num_iter] = infer(file_path, families, module.vec, plot_fails)

    if write_results:
        output_path = vector_location(families) + "/results.csv"
        with open(output_path, 'w') as file:
            file.write("Iteration,Accuracy,Infer_Time\n")
            for i, res in results.items():
                file.write("%d,%f,%f\n" % (i, 100*res[0], res[1]))

def mst_with_root(G):

    root = 0
    root_nodes = G[root]
    Gs = map(lambda node: graphs_with_single_edge_from_root(G, root, node), root_nodes)
    trees = map(lambda G: mst(root, G), Gs)
    vals = map(graph_weight, trees)
    min_idx = np.argmin(vals)

    return trees[min_idx]

def infer(file_path, families, vector, plot_fails=False):
    start_time = time.time()
    sentences = parse_input_file(file_path)
    word_num = sum([len(sentence) for sentence in sentences])
    graphs = map(lambda sentence: sentence_to_graph(sentence, families), sentences)
    selected_vec = np.asarray(vector)
    weighted_graphs = list(map(lambda graph: get_weighted_graph(graph, selected_vec), graphs))
    msts = map(mst_with_root, weighted_graphs)
    accuracies = map(get_inference_accuracy, zip(sentences, msts))
    if plot_fails:
        get_fails(sentences, msts, vector_location(families))

    total_acc = sum(list(accuracies)) / float(word_num)
    print("accuracy: " + str(total_acc*100) + "%")
    print("time: " + str(time.time()-start_time) + " seconds")
    return total_acc, time.time()-start_time


def train_and_infer(file_name, families, iterations = [20, 50, 80, 100]):
    vecs_by_iter = perceptron(file_name, iterations, families)
    results = {}
    for i, vec in vecs_by_iter.items():
        results[i] = infer('test.labeled', families, vec)

    output_path = vector_location(families) + "/results.csv"
    with open(output_path, 'w') as file:
        file.write("Iteration,Accuracy,Infer_Time\n")
        for i, res in results.items():
            file.write("%d,%f,%f\n" % (i, 100*res[0], res[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infer', action='store_true')
    parser.add_argument('-p', '--plot_fails', action='store_true')
    parser.add_argument('-it', '--iter', default=-1, type=int)
    parser.add_argument('-w', '--write_results', action='store_true')
    args = parser.parse_args()

    families = [8,10,11,12,13,14,16,17,18,19,20,21,22,24,25,26,28,29,33,34,38,40,41]
    print('families: ' + ','.join([str(f) for f in families]))
    if args.infer:
        infer_only("test.labeled", families, args.plot_fails, args.iter, args.write_results)
    else:
        train_and_infer("train.labeled", families)

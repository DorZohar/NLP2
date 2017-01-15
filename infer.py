import numpy as np
import time
from chu_liu import mst
from features import parse_input_file
from learn import sentence_to_graph, get_weighted_graph, perceptron, vector_location
#from vector import vec


def get_inference_accuracy(sentence_and_tree):
    sentence, tree = sentence_and_tree
    sum = 0
    for key, word in sentence.items():
        if word.parent in tree and key in tree[word.parent]:
            sum += 1

    return sum


def infer(file_path, families, vector):
    start_time = time.time()
    sentences = parse_input_file(file_path)
    word_num = sum([len(sentence) for sentence in sentences])
    graphs = map(lambda sentence: sentence_to_graph(sentence, families), sentences)
    selected_vec = np.asarray(vector)
    weighted_graphs = list(map(lambda graph: get_weighted_graph(graph, selected_vec), graphs))
    msts = map(lambda wgraph: mst(0, wgraph), weighted_graphs)
    accuracies = map(get_inference_accuracy, zip(sentences, msts))

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
#    infer("test.labeled", [1, 2, 3, 4, 5, 6, 8, 10, 13, 15, 16, 17, 18, 19, 20])
    train_and_infer("train.labeled", [1, 2, 3, 4, 5, 6, 8, 10, 13])

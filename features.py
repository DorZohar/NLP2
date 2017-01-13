from feature_families import family_indices

class Word:
    def __init__(self, line):
        params = line.split('\t')
        self.idx = int(params[0])
        self.word = params[1]
        self.pos = params[3]
        self.parent = int(params[6])
        self.edge_type = params[7]


root_word = Word("0\tROOT\t_\tROOT\t_\t_\t-1\tROOT\t_\t_")


class Feature:
    def __init__(self, family_id, func):
        self.family_id = family_id
        self.func = func
        self.keys_dict = family_indices[family_id]

    def get_indices(self, cword, pword, offset):
        return [self.keys_dict[key] + offset for key in self.func(cword, pword) if key in self.keys_dict]

    def create_indices(self, sentences):

        possible_values_set = set()
        for sentence in sentences:
            sentence[0] = root_word
            for word in sentence.values():
                if word.idx == 0:
                    continue
                possible_values_set.add(self.func(word, sentence[word.parent]))

        possible_values_list = list(possible_values_set)
        possible_values_list.sort()
        possible_values_dict = {}
        idx = 0
        for val in possible_values_list:
            possible_values_dict[val] = idx
            idx += 1

        return possible_values_dict


feature_families = [
    Feature(0, lambda cword, pword: [None]),
    Feature(1, lambda cword, pword: [(pword.word, pword.pos)]),
    Feature(2, lambda cword, pword: [pword.word]),
    Feature(3, lambda cword, pword: [pword.pos]),
    Feature(4, lambda cword, pword: [(cword.word, cword.pos)]),
    Feature(5, lambda cword, pword: [cword.word]),
    Feature(6, lambda cword, pword: [cword.pos]),
    Feature(7, lambda cword, pword: [(pword.word, pword.pos, cword.word, cword.pos)]),
    Feature(8, lambda cword, pword: [(pword.pos, cword.word, cword.pos)]),
    Feature(9, lambda cword, pword: [(pword.word, cword.word, cword.pos)]),
    Feature(10, lambda cword, pword: [(pword.word, pword.pos, cword.pos)]),
    Feature(11, lambda cword, pword: [(pword.word, pword.pos, cword.word)]),
    Feature(12, lambda cword, pword: [(pword.word, cword.word)]),
    Feature(13, lambda cword, pword: [(pword.pos, cword.pos)]),

]


def parse_input_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()

    sentences = []
    sentence = {}

    for line in lines:
        if line == "\n":
            sentences.append(sentence)
            sentence = {}
        else:
            new_word = Word(line)
            sentence[new_word.idx] = new_word

    if len(sentence) != 0:
        sentences.append(sentence)

    return sentences


def create_feature_indices(file_path):
    sentences = parse_input_file(file_path)

    file = open('feature_families.py', 'w')
    file.write("family_indices = {}\n\n")
    for feat in feature_families:
        indices = feat.create_indices(sentences)
        file.write("family_indices[%d] = %s\n" % (feat.family_id, indices))

    file.close()


def get_edge_features(cword, pword, families):
    offset = 0
    indices = []
    for family in families:
        indices += feature_families[family].get_indices(cword, pword, offset)
        offset += len(feature_families[family].keys_dict)

    return indices


def total_feature_num(families):
    num = 0
    for family in families:
        num += len(feature_families[family].keys_dict)

    return num


#if __name__ == "__main__":
    #create_feature_indices('train.labeled')

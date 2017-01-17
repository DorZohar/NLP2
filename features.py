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
max_arc_length = 10

class Feature:
    def __init__(self, family_id, func):
        self.family_id = family_id
        self.func = func
        if family_id in family_indices:
            self.keys_dict = family_indices[family_id]
        else:
            self.keys_dict = {}

    def get_indices(self, cword, pword, sentence, offset):
        return [self.keys_dict[key] + offset for key in self.func(cword, pword, sentence) if key in self.keys_dict]

    def create_indices(self, sentences):

        possible_values_set = set()
        for sentence in sentences:
            sentence[0] = root_word
            for word in sentence.values():
                if word.idx == 0:
                    continue
                for item in self.func(word, sentence[word.parent], sentence):
                    possible_values_set.add(item)

        possible_values_list = list(possible_values_set)
        possible_values_list.sort()
        possible_values_dict = {}
        idx = 0
        for val in possible_values_list:
            possible_values_dict[val] = idx
            idx += 1

        return possible_values_dict


feature_families = [
    Feature(0, lambda cword, pword, sentence: [None]),
    Feature(1, lambda cword, pword, sentence: [(pword.word, pword.pos)]),
    Feature(2, lambda cword, pword, sentence: [pword.word]),
    Feature(3, lambda cword, pword, sentence: [pword.pos]),
    Feature(4, lambda cword, pword, sentence: [(cword.word, cword.pos)]),
    Feature(5, lambda cword, pword, sentence: [cword.word]),
    Feature(6, lambda cword, pword, sentence: [cword.pos]),
    Feature(7, lambda cword, pword, sentence: [(pword.word, pword.pos, cword.word, cword.pos)]),
    Feature(8, lambda cword, pword, sentence: [(pword.pos, cword.word, cword.pos)]),
    Feature(9, lambda cword, pword, sentence: [(pword.word, cword.word, cword.pos)]),
    Feature(10, lambda cword, pword, sentence: [(pword.word, pword.pos, cword.pos)]),
    Feature(11, lambda cword, pword, sentence: [(pword.word, pword.pos, cword.word)]),
    Feature(12, lambda cword, pword, sentence: [(pword.word, cword.word)]),
    Feature(13, lambda cword, pword, sentence: [(pword.pos, cword.pos)]),
    Feature(14, lambda cword, pword, sentence:
        [(pword.pos, cword.pos)] if abs(pword.idx - cword.idx) < max_arc_length else []),
    Feature(15, lambda cword, pword, sentence:
        [(pword.pos, cword.pos)] if pword.idx > cword.idx else []),
    Feature(16, lambda cword, pword, sentence:
        [(pword.pos, cword.pos)] if pword.idx < cword.idx else []),
    Feature(17, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[i].pos) for i in range(min(pword.idx, cword.idx)+1, max(pword.idx, cword.idx))]),
    Feature(18, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, abs(pword.idx - cword.idx))] if abs(pword.idx - cword.idx) < max_arc_length else [(pword.pos, cword.pos, 0)]),
    Feature(19, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[pword.idx - 1].pos, sentence[cword.idx - 1].pos)] if pword.idx > 1 and cword.idx > 1 else []),
    Feature(20, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[pword.idx + 1].pos, sentence[cword.idx + 1].pos)] if pword.idx < len(sentence) - 1 and cword.idx < len(sentence) - 1 else []),
    Feature(21, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx + 1].pos, sentence[cword.idx + 2].pos)] if cword.idx < len(sentence) - 2 else []),
    Feature(22, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 1].pos, sentence[cword.idx - 2].pos)] if cword.idx > 2 else []),
    Feature(23, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 1].pos)] if cword.idx > 1 else []),
    Feature(24, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[pword.idx - 1].pos)] if pword.idx > 1 else []),
    Feature(25, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx + 1].pos)] if cword.idx < len(sentence) - 1 else []),
    Feature(26, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[pword.idx + 1].pos)] if pword.idx < len(sentence) - 1 else []),
    Feature(27, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[pword.idx - 1].pos, sentence[pword.idx + 1].pos)] if pword.idx < len(sentence) - 1 and pword.idx > 1 else []),
    Feature(28, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 1].pos, sentence[cword.idx + 1].pos)] if cword.idx < len(sentence) - 1 and cword.idx > 1 else []),
    Feature(29, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 1].word, sentence[cword.idx + 1].word)] if cword.idx < len(sentence) - 1 and cword.idx > 1 else []),
    Feature(30, lambda cword, pword, sentence:
        [(cword.word, sentence[cword.idx - 1].pos, sentence[cword.idx + 1].pos)] if cword.idx < len(sentence) - 1 and cword.idx > 1 else []),
    Feature(31, lambda cword, pword, sentence:
        [(cword.word, sentence[pword.idx - 1].pos, sentence[pword.idx + 1].pos)] if pword.idx < len(sentence) - 1 and pword.idx > 1 else []),
    Feature(32, lambda cword, pword, sentence:
        [(cword.pos, sentence[cword.idx - 1].pos, sentence[cword.idx + 1].pos)] if cword.idx < len(sentence) - 1 and cword.idx > 1 else []),
    Feature(33, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 2].pos, sentence[cword.idx + 2].pos)] if cword.idx < len(sentence) - 2 and cword.idx > 2 else []),
    Feature(34, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 2].word, sentence[cword.idx + 2].word)] if cword.idx < len(sentence) - 2 and cword.idx > 2 else []),
    Feature(35, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 3].pos, sentence[cword.idx + 3].pos)] if cword.idx < len(sentence) - 3 and cword.idx > 3 else []),
    Feature(36, lambda cword, pword, sentence:
        [(pword.pos, cword.pos, sentence[cword.idx - 3].word, sentence[cword.idx + 3].word)] if cword.idx < len(sentence) - 3 and cword.idx > 3 else []),
    Feature(37, lambda cword, pword, sentence:
        [(pword.word, cword.word)] if pword.idx > cword.idx else []),
    Feature(38, lambda cword, pword, sentence:
        [(pword.word, cword.word)] if pword.idx < cword.idx else []),
    Feature(39, lambda cword, pword, sentence:
        [(pword.word, cword.word)] if abs(pword.idx - cword.idx) < max_arc_length else []),
    Feature(40, lambda cword, pword, sentence:
        [(pword.pos, cword.pos)] if abs(pword.idx - cword.idx) < 2 else []),
    Feature(41, lambda cword, pword, sentence:
        [(pword.pos, cword.pos)] if abs(pword.idx - cword.idx) < 4 else []),
    Feature(42, lambda cword, pword, sentence:
        [(pword.pos, cword.pos)] if abs(pword.idx - cword.idx) < 8 else []),
    Feature(43, lambda cword, pword, sentence:
        [(pword.pos, cword.pos)] if abs(pword.idx - cword.idx) < 16 else []),
]


def parse_input_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()

    sentences = []
    sentence = {}

    for line in lines:
        if line == "\n" or line=="\r\n":
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


def get_edge_features(cword, pword, sentence, families):
    offset = 0
    indices = []
    for family in families:
        indices += feature_families[family].get_indices(cword, pword, sentence, offset)
        offset += len(feature_families[family].keys_dict)

    return indices


def total_feature_num(families):
    num = 0
    for family in families:
        num += len(feature_families[family].keys_dict)

    return num


if __name__ == "__main__":
    create_feature_indices('train.labeled')

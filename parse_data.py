import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle


def load_data():
    train = _parse_data(open('Train/data.data', 'rb'))

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = ['O', 'B-ASP', 'I-ASP', 'B-OPI', 'I-OPI']

    with open('model/config.pkl', 'wb') as out:
        pickle.dump((vocab, chunk_tags), out)

    train = _process_data(train, vocab, chunk_tags)
    return train, (vocab, chunk_tags)


def _parse_data(fh):
    split_text = '\r\n'
    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, max_len)

    y_chunk = pad_sequences(y_chunk, max_len, value=-1)

    y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, max_len=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], max_len)
    return x, length


def search_begin(single):
    train = _parse_data(open('Train/data.data', 'rb'))
    begin = set([row[0] for sample in train for row in sample if row[1][0] == 'B'])
    return single in begin


if __name__ == '__main__':
    print(search_begin('很'))
    print(search_begin('是'))

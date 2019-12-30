import os

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation, Bidirectional
from keras.models import Sequential
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences

from pre_process import get_asp_opi_combined_labels, read_task

max_len = 100
file_path_word2vec = 'model/word2vec_polarity.pkl'
file_path_lstm = 'model/polarity.h5'


def train_word2vec():
    asp_opi_combined, category_labels, polarity_labels, word_list = get_asp_opi_combined_labels()
    polarity_term = {'负面': 0, '中性': 1, '正面': 2}
    pol_labels = [polarity_term.get(word) for word in polarity_labels]
    if not os.path.exists(file_path_word2vec):
        model = Word2Vec(word_list, size=100, min_count=5, window=5)
        model.save(file_path_word2vec)
    else:
        model = Word2Vec.load(file_path_word2vec)
    return asp_opi_combined, pol_labels, model


def generate_id2vec(model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2index = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号

    w2vec = {word: model.wv.get_vector(word) for word in w2index.keys()}  # 词语的词向量
    n_vocabs = len(w2index) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2index.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2index, embedding_weights


def text_to_array(w2index, sent_list):  # 文本转为索引数字模式
    sentences_array = []
    for sen in sent_list:
        new_sen = [w2index.get(word, 0) for word in sen]  # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)


def prepare_data(w2index, combined, label_list):
    x_train = text_to_array(w2index, combined)
    x_train = pad_sequences(x_train, maxlen=max_len)
    return np.array(x_train), np_utils.to_categorical(label_list)


def train_lstm(w2index, embedding_weights, x_train, y_train):
    n_epoch = 10
    Embedding_dim = 100
    model = Sequential()
    model.add(Embedding(output_dim=Embedding_dim,
                        input_dim=len(w2index) + 1,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=100))
    model.add(Bidirectional(LSTM(50), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if not os.path.exists(file_path_lstm):
        model.fit(x_train, y_train, batch_size=32, epochs=n_epoch, verbose=1, validation_split=0.04)
        model.save(file_path_lstm)
        score = model.evaluate(x_train, y_train, batch_size=32)
        print(score)
    else:
        model.load_weights(file_path_lstm)
    return model


def predict(w2index, model, seg_list):
    seg2id = [w2index.get(word, 0) for sen in seg_list for word in sen]
    sen_input = pad_sequences([seg2id], max_len)
    res = model.predict(sen_input)[0]
    return np.argmax(res)


if __name__ == '__main__':
    asp_opi, labels, word2vec_model = train_word2vec()
    w2id, the_embedding_weights = generate_id2vec(word2vec_model)
    x_tra, y_tra = prepare_data(w2id, asp_opi, labels)
    lstm_model = train_lstm(w2id, the_embedding_weights, x_tra, y_tra)
    label_dic = {0: '负面', 1: '中性', 2: '正面'}
    output = []

    asp_opi_string, test_asp_opi = read_task(task='task2')
    for string, sent in zip(asp_opi_string, test_asp_opi):
        label = predict(w2id, lstm_model, sent)
        result = string + ',' + label_dic.get(int(label)) + '\n'
        print(result)
        output.append(result)
    with open('Test/task3_answer.csv', 'w', encoding='utf-8') as f:
        for sent in output:
            f.write(sent)

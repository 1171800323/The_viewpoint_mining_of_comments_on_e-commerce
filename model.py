from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
import parse_data
import pickle

EMBED_DIM = 200
BiRNN_UNITS = 200


def create_model(train=True):
    if train:S
        (train_x, train_y), (vocab, chunk_tags) = parse_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, train_y)
    else:
        return model, (vocab, chunk_tags)


if __name__ == '__main__':
    EPOCHS = 9
    model_, (tra_x, tra_y) = create_model()
    # train model
    model_.fit(tra_x, tra_y, batch_size=16, epochs=EPOCHS, validation_split=0.04)
    model_.save('model/crf.h5')
    score = model_.evaluate(tra_x, tra_y, batch_size=16)
    print(score)

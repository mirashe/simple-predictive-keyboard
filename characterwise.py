import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
import seaborn as sns
from pylab import rcParams
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.preprocessing.sequence import pad_sequences

# %matplotlib inline

should_save = True
should_load = not True
input_file_path = 'uniface-code-samples-01.txt'
model_files_title = 'characterwise-uniface-stateful'

np.random.seed(42)
tf.random.set_seed(42)

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5

text = open(input_file_path).read().lower()
text = re.sub(' +', ' ', text)
text = re.sub('( *[\r\n])+', '\r\n', text)

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

if should_load:
    model = load_model(model_files_title + '-model.h5')
    history = pickle.load(open(model_files_title + "-history.p", "rb"))
else:
    sentences = text.split('\r\noperation')
    sentences_max_len = 0
    for sentence_index, sentence in enumerate(sentences):
        if sentence_index != 0:
            sentences[sentence_index] = 'operation' + sentence
        if sentences_max_len < len(sentences[sentence_index]):
            sentences_max_len = len(sentences[sentence_index])

    #    data_gens = []
    #    for i, sentence in enumerate(sentences):
    #        tX = np.zeros((sentences_max_len, len(chars)), dtype=np.bool)
    #        for t, char in enumerate(sentence):
    #            tX[t, char_indices[char]] = 1
    #        data_gens.append(TimeseriesGenerator(tX, tX, batch_size=1, length=2))
    #
    #    model = Sequential()
    #    model.add(LSTM(128, input_shape=(None, len(chars)), batch_size=1, stateful=True))
    #    model.add(Dense(len(chars)))
    #    model.add(Activation('softmax'))
    #
    #    optimizer = RMSprop(lr=0.01)
    #    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #
    #    model.summary()
    #
    #    history = model.fit_generator(data_gens[0]
    #                        , epochs=1
    #                        , shuffle=False).history
    #


    X = np.zeros((len(sentences), sentences_max_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = True

    data_gen = TimeseriesGenerator(X, X, batch_size=1, length=2)

    model = Sequential()
    model.add(LSTM(128, input_shape=(None, len(chars)), batch_size=1, stateful=True))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    history = model.fit_generator(data_gen
                        , epochs=1
                        , shuffle=False).history

    if should_save:
        model.save(model_files_title + '-model.h5')
        pickle.dump(history, open(model_files_title + '-history.p', "wb"))


def predict_completions(sentence_40, prediction_length):
    sentence_matrix = np.zeros((1, SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
    for t, char in enumerate(sentence_40):
        sentence_matrix[0, t, char_indices[char]] = 1

    predicted_text = ""

    for prediction_index in range(prediction_length):
        predictions_for_first_sample = model.predict(sentence_matrix)[0]
        number_of_guesses = 1
        best_predictions_indices = heapq.nlargest(number_of_guesses, range(predictions_for_first_sample.size),
                                                  predictions_for_first_sample.take)
        best_predictions = np.array(chars)[best_predictions_indices]
        predicted_text += best_predictions[0]

        new_char_matrix = np.zeros((1, 1, len(chars)), dtype=np.bool)
        new_char_matrix[0, 0, char_indices[best_predictions[0]]] = 1

        sentence_matrix = np.concatenate((sentence_matrix[0:1, 1:SEQUENCE_LENGTH, :], new_char_matrix), axis=1)

    return predicted_text


sample_start_position = 32
input_sample = text[sample_start_position: SEQUENCE_LENGTH + sample_start_position]
print("Input sample: \r\n", input_sample)
print("Prediction sample: ", predict_completions(input_sample, 120))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

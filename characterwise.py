import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams

# %matplotlib inline

input_file_path = 'nietzsche.txt'
model_files_title = 'characterwise'

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5

text = open(input_file_path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print(f'num training examples: {len(sentences)}')

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history

model.save(model_files_title + '-model.h5')
pickle.dump(history, open(model_files_title + '-history.p', "wb"))

model = load_model(model_files_title + '-model.h5')
history = pickle.load(open(model_files_title + "-history.p", "rb"))


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


print("Prediction sample: ", predict_completions(text[0:40], 120))

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

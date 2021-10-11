import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
import seaborn as sns
from pylab import rcParams
import re

should_save = not True
should_load = True
input_file_path = 'uniface-code-samples-01.txt'
model_files_title = 'CW-U-SO'

np.random.seed(42)
tf.random.set_seed(42)

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5

text = open(input_file_path).read().lower()
text = re.sub(' +', ' ', text)
text = re.sub('( *[\r\n])+', '\r\n', text)

operations_texts = text.split('\r\noperation')
for operation_index, operation_text in enumerate(operations_texts):
    if operation_index != 0:
         operations_texts[operation_index] = 'operation' + operation_text

# print('trimmed text: ', text)

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

SENTENCES_MINIMUM_LENGTH = 1
step = 3
sentences = []
next_chars = []

SENTENCES_EQUALIZED_LENGTH = 40

for operation_index, operation_text in enumerate(operations_texts):
    for i in range(0, len(operation_text) - SENTENCES_MINIMUM_LENGTH, step):
        sentences.append(operation_text[max(0, i + SENTENCES_MINIMUM_LENGTH - SENTENCES_EQUALIZED_LENGTH):
                                        i + SENTENCES_MINIMUM_LENGTH])
        next_chars.append(operation_text[i + SENTENCES_MINIMUM_LENGTH])
print(f'num training examples: {len(sentences)}')

X = np.zeros((len(sentences), SENTENCES_EQUALIZED_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(reversed(sentence[-SENTENCES_EQUALIZED_LENGTH:])):
        X[i, SENTENCES_EQUALIZED_LENGTH-t-1, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

if should_load:
    model = load_model(model_files_title + '-model.h5')
    history = pickle.load(open(model_files_title + "-history.p", "rb"))
else:
    model = Sequential()
    model.add(LSTM(128, input_shape=(SENTENCES_EQUALIZED_LENGTH, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history

    if should_save:
        model.save(model_files_title + '-model.h5')
        pickle.dump(history, open(model_files_title + '-history.p', "wb"))


def predict_completions(sentence, prediction_length):
    sentence_matrix = np.zeros((1, SENTENCES_EQUALIZED_LENGTH, len(chars)), dtype=np.bool)
    for t, char in enumerate(reversed(sentence[-SENTENCES_EQUALIZED_LENGTH:])):
        sentence_matrix[0, SENTENCES_EQUALIZED_LENGTH-t-1, char_indices[char]] = 1

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

        sentence_matrix = np.concatenate((sentence_matrix[0:1, 1:SENTENCES_EQUALIZED_LENGTH, :], new_char_matrix), axis=1)

    return predicted_text


input_sample = operations_texts[3][0:40]
print("Input sample: \r\n", input_sample)
print("Prediction sample: \r\n", predict_completions(input_sample, 120))

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
